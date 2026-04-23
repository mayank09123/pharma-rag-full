"""
LLM Agents for Pharma Workflow Automation
- DrugApprovalSummaryAgent: summarizes new FDA approvals
- MarketingBriefAgent: generates compliant draft marketing briefs
"""

from __future__ import annotations

import json
import os
import requests
from datetime import datetime, timedelta
from typing import Optional


AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_new_approvals",
            "description": "Search FDA DailyMed for recently approved drug labels",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_back": {"type": "integer", "description": "How many days back to search"},
                    "drug_class": {"type": "string", "description": "Optional drug class filter"},
                },
                "required": ["days_back"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_label_sections",
            "description": "Retrieve specific sections from a drug label by set ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_id": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Section names like 'indications_and_usage', 'contraindications'",
                    },
                },
                "required": ["set_id", "sections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_review",
            "description": "Flag content for medical/legal review",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "reason": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["content", "reason", "priority"],
            },
        },
    },
]


def search_new_approvals(days_back: int, drug_class: Optional[str] = None) -> dict:
    try:
        params = {"pagesize": 20}
        resp = requests.get(
            "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
            params=params,
            timeout=15,
        )
        data = resp.json().get("data", [])
        if drug_class:
            data = [d for d in data if drug_class.lower() in d.get("title", "").lower()]
        return {"count": len(data), "approvals": data[:5]}
    except Exception as e:
        return {"error": str(e), "approvals": []}


def get_label_sections(set_id: str, sections: list) -> dict:
    from src.ingestion.fda_loader import FDALabelLoader
    loader = FDALabelLoader()
    try:
        chunks = loader.load_label_by_setid(set_id)
        filtered = {s: [] for s in sections}
        for chunk in chunks:
            if chunk.section_name in sections:
                filtered[chunk.section_name].append(chunk.text)
        return {k: " ".join(v)[:1000] for k, v in filtered.items()}
    except Exception as e:
        return {"error": str(e)}


def flag_for_review(content: str, reason: str, priority: str) -> dict:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "priority": priority,
        "reason": reason,
        "content_preview": content[:200],
        "status": "pending_review",
    }
    print(f"\n🚩 FLAGGED [{priority.upper()}]: {reason}")
    return record


TOOL_REGISTRY = {
    "search_new_approvals": search_new_approvals,
    "get_label_sections": get_label_sections,
    "flag_for_review": flag_for_review,
}


class PharmaAgent:
    """ReAct-style agent that loops: think → call tool → observe → repeat."""

    def __init__(self, system_prompt: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self._llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model
        self._system = system_prompt

    def run(self, task: str, max_iterations: int = 6) -> str:
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": task},
        ]

        for _ in range(max_iterations):
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                return msg.content or ""

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                print(f"  [Agent] → {fn_name}({fn_args})")

                fn = TOOL_REGISTRY.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        return "Max iterations reached."


class DrugApprovalSummaryAgent(PharmaAgent):
    SYSTEM = """You are a pharmaceutical intelligence analyst.
Find recent FDA drug approvals and produce structured summaries including:
drug name, approved indication, mechanism of action, key safety considerations.
Always use search_new_approvals first, then get_label_sections for details.
Flag any content requiring medical/regulatory review."""

    def __init__(self, **kwargs):
        super().__init__(self.SYSTEM, **kwargs)

    def summarize_recent(self, days_back: int = 30, drug_class: Optional[str] = None) -> str:
        task = f"Summarize FDA drug approvals from the last {days_back} days"
        if drug_class:
            task += f" in the {drug_class} space"
        task += ". For each drug found, get indications_and_usage and contraindications sections."
        return self.run(task)


class MarketingBriefAgent(PharmaAgent):
    SYSTEM = """You are a pharmaceutical marketing writer with regulatory expertise.
Create draft marketing briefs that are:
1. Grounded in FDA-approved labeling (use get_label_sections)
2. Compliant with OPDP promotional guidelines
3. Include fair balance safety language
4. Flag all efficacy claims via flag_for_review for MLR approval

Structure every brief as:
- Headline claim
- Indication statement
- Key efficacy data
- Safety/fair balance summary
- [MLR Review Notes]"""

    def __init__(self, **kwargs):
        super().__init__(self.SYSTEM, **kwargs)

    def generate_brief(self, drug_name: str, set_id: str, target_audience: str = "HCP") -> str:
        task = (
            f"Create a {target_audience}-facing marketing brief for {drug_name} "
            f"(set_id: {set_id}). Get indications_and_usage, contraindications, "
            f"and adverse_reactions sections. Flag any efficacy claims for MLR review."
        )
        return self.run(task)
