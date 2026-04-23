"""
Master Pharma Agent — All-in-One FDA Intelligence System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines 5 agents into one unified FastAPI endpoint:
  1. Drug Label Summarizer Agent
  2. Clinical Trial Search Agent
  3. Adverse Event Reporter Agent
  4. Drug Dosage Recommender Agent
  5. Regulatory Compliance Checker Agent

Run:
    uvicorn src.agents.master_agent:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import json
import os
import requests
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── Agent type enum ───────────────────────────────────────────────────────
class AgentType(str, Enum):
    SUMMARIZER  = "summarizer"
    CLINICAL    = "clinical_trial"
    ADVERSE     = "adverse_event"
    DOSAGE      = "dosage"
    COMPLIANCE  = "compliance"
    ALL         = "all"


# ── Result dataclass ──────────────────────────────────────────────────────
@dataclass
class AgentResult:
    agent:     str
    drug:      str
    result:    str
    sources:   list[dict] = field(default_factory=list)
    alerts:    list[str]  = field(default_factory=list)
    timestamp: str        = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "agent":     self.agent,
            "drug":      self.drug,
            "result":    self.result,
            "sources":   self.sources,
            "alerts":    self.alerts,
            "timestamp": self.timestamp,
        }


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — shared across all agents
# ══════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dailymed",
            "description": "Search FDA DailyMed for a drug label and return its set ID and title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string", "description": "Drug name e.g. 'warfarin'"},
                    "limit":     {"type": "integer", "description": "Max results (default 3)"},
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_label_section",
            "description": "Fetch a specific section from an FDA drug label by set ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "set_id":  {"type": "string"},
                    "section": {
                        "type": "string",
                        "enum": [
                            "indications_and_usage",
                            "dosage_and_administration",
                            "contraindications",
                            "warnings",
                            "adverse_reactions",
                            "drug_interactions",
                            "clinical_pharmacology",
                            "clinical_studies",
                            "patient_information",
                            "boxed_warning",
                        ],
                        "description": "Section name to fetch",
                    },
                },
                "required": ["set_id", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_clinical_trials",
            "description": "Search ClinicalTrials.gov for trials involving a drug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "status":    {
                        "type": "string",
                        "enum": ["RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING", "ALL"],
                        "description": "Trial status filter",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_adverse_events",
            "description": "Search FDA FAERS (adverse event reporting) for a drug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "limit":     {"type": "integer", "description": "Max results (default 5)"},
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_vector_store",
            "description": "Query the local ChromaDB vector store for drug information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":   {"type": "string"},
                    "section": {
                        "type": "string",
                        "description": "Optional section filter e.g. 'drug_interactions'",
                    },
                    "k": {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_compliance_issue",
            "description": "Flag a compliance issue found in drug information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue":    {"type": "string", "description": "Description of the issue"},
                    "severity": {"type": "string", "enum": ["LOW", "MODERATE", "HIGH"]},
                    "drug":     {"type": "string"},
                },
                "required": ["issue", "severity", "drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save the generated report to a JSON file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content":  {"type": "string"},
                    "drug":     {"type": "string"},
                    "agent":    {"type": "string"},
                },
                "required": ["filename", "content", "drug", "agent"],
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════

def search_dailymed(drug_name: str, limit: int = 3) -> dict:
    try:
        resp = requests.get(
            "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
            params={"drug_name": drug_name, "pagesize": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return {
            "found": len(data),
            "results": [
                {"set_id": d["setid"], "title": d.get("title", ""), "published": d.get("published", "")}
                for d in data
            ],
        }
    except Exception as e:
        return {"found": 0, "results": [], "error": str(e)}


def get_label_section(set_id: str, section: str) -> dict:
    SECTION_CODES = {
        "boxed_warning":           "34066-1",
        "indications_and_usage":   "34067-9",
        "dosage_and_administration":"34068-7",
        "contraindications":       "34070-3",
        "warnings":                "34071-1",
        "adverse_reactions":       "34084-4",
        "drug_interactions":       "34073-7",
        "clinical_pharmacology":   "34090-1",
        "clinical_studies":        "34092-7",
        "patient_information":     "42230-3",
    }
    try:
        import xml.etree.ElementTree as ET
        url  = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns   = {"v3": "urn:hl7-org:v3"}
        code = SECTION_CODES.get(section, "")
        for sec in root.findall(".//v3:section", ns):
            code_el = sec.find("v3:code", ns)
            if code_el is not None and code_el.get("code") == code:
                texts = []
                for el in sec.iter():
                    if el.text and el.text.strip():
                        texts.append(el.text.strip())
                    if el.tail and el.tail.strip():
                        texts.append(el.tail.strip())
                text = " ".join(texts)[:2500]
                return {"found": True, "section": section, "text": text, "source_url": url}
        return {"found": False, "section": section, "text": "Section not found in label."}
    except Exception as e:
        return {"found": False, "error": str(e)}


def search_clinical_trials(drug_name: str, status: str = "ALL", limit: int = 5) -> dict:
    try:
        params = {
            "query.intr":  drug_name,
            "pageSize":    limit,
            "format":      "json",
            "fields":      "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount",
        }
        if status != "ALL":
            params["filter.overallStatus"] = status

        resp = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data    = resp.json()
        studies = data.get("studies", [])
        results = []
        for s in studies:
            proto  = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            st_mod = proto.get("statusModule", {})
            de_mod = proto.get("designModule", {})
            results.append({
                "nct_id":     id_mod.get("nctId", ""),
                "title":      id_mod.get("briefTitle", ""),
                "status":     st_mod.get("overallStatus", ""),
                "phase":      de_mod.get("phases", ["N/A"])[0] if de_mod.get("phases") else "N/A",
                "enrollment": de_mod.get("enrollmentInfo", {}).get("count", "N/A"),
                "url":        f"https://clinicaltrials.gov/study/{id_mod.get('nctId','')}",
            })
        return {"found": len(results), "trials": results}
    except Exception as e:
        return {"found": 0, "trials": [], "error": str(e)}


def search_adverse_events(drug_name: str, limit: int = 5) -> dict:
    try:
        resp = requests.get(
            "https://api.fda.gov/drug/event.json",
            params={
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "limit":  limit,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data    = resp.json()
        results = data.get("results", [])
        events  = []
        for r in results:
            reactions = [
                rx.get("reactionmeddrapt", "Unknown")
                for rx in r.get("patient", {}).get("reaction", [])
            ]
            serious = r.get("serious", "1")
            events.append({
                "reactions":       reactions[:5],
                "serious":         "Yes" if serious == "1" else "No",
                "report_date":     r.get("receiptdate", "Unknown"),
                "patient_age":     r.get("patient", {}).get("patientonsetage", "Unknown"),
                "patient_sex":     "Male" if r.get("patient", {}).get("patientsex") == "1" else "Female",
            })
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        return {"total_reports": total, "sample_events": events}
    except Exception as e:
        return {"total_reports": 0, "sample_events": [], "error": str(e)}


def search_vector_store(query: str, section: Optional[str] = None, k: int = 5) -> dict:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        from src.retrieval.vector_store import DrugLabelVectorStore
        store   = DrugLabelVectorStore()
        results = store.hybrid_search(query, k=k)
        if section:
            results = [r for r in results
                       if r["metadata"].get("section_name") == section] or results
        return {
            "found":  len(results),
            "chunks": [
                {
                    "text":    r["text"][:500],
                    "drug":    r["metadata"].get("drug_name", ""),
                    "section": r["metadata"].get("section_name", ""),
                    "score":   round(r["score"], 3),
                }
                for r in results
            ],
        }
    except Exception as e:
        return {"found": 0, "chunks": [], "error": str(e)}


def flag_compliance_issue(issue: str, severity: str, drug: str) -> dict:
    record = {
        "drug":      drug,
        "issue":     issue,
        "severity":  severity,
        "timestamp": datetime.utcnow().isoformat(),
    }
    icon = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(severity, "⚪")
    print(f"\n{icon} COMPLIANCE [{severity}] — {drug}: {issue}")
    return record


def save_report(filename: str, content: str, drug: str, agent: str) -> dict:
    os.makedirs("reports", exist_ok=True)
    safe = filename.replace("/", "_").replace(" ", "_")
    if not safe.endswith(".json"):
        safe += ".json"
    path = os.path.join("reports", safe)
    payload = {
        "drug":      drug,
        "agent":     agent,
        "content":   content,
        "saved_at":  datetime.utcnow().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  💾 Report saved: {path}")
    return {"saved": True, "path": path}


TOOL_REGISTRY = {
    "search_dailymed":       search_dailymed,
    "get_label_section":     get_label_section,
    "search_clinical_trials":search_clinical_trials,
    "search_adverse_events": search_adverse_events,
    "search_vector_store":   search_vector_store,
    "flag_compliance_issue": flag_compliance_issue,
    "save_report":           save_report,
}


# ══════════════════════════════════════════════════════════════════════════
# AGENT SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════

PROMPTS = {

    AgentType.SUMMARIZER: """You are a pharmaceutical label summarizer.
Given a drug name, use search_dailymed to find its label, then use
get_label_section to retrieve: indications_and_usage, dosage_and_administration,
contraindications, and warnings.
Produce a clean structured summary with sections:
INDICATION | DOSAGE | CONTRAINDICATIONS | KEY WARNINGS
Save the result using save_report.
Base everything strictly on retrieved FDA label content.""",

    AgentType.CLINICAL: """You are a clinical trial research analyst.
Given a drug name, use search_clinical_trials to find active and completed trials.
Also use search_dailymed + get_label_section(clinical_studies) for label-based
clinical evidence.
Produce a structured report: ACTIVE TRIALS | COMPLETED TRIALS | KEY FINDINGS
Save using save_report. Cite NCT IDs for all trials mentioned.""",

    AgentType.ADVERSE: """You are a pharmacovigilance specialist.
Given a drug name:
1. Use search_adverse_events to find FAERS adverse event reports
2. Use search_dailymed + get_label_section(adverse_reactions) for label data
3. Cross-reference FAERS events with label-listed reactions
4. Flag any serious unlisted reactions using flag_compliance_issue(severity=HIGH)
Produce a report: TOP REACTIONS | SERIOUS EVENTS | LABEL vs REAL-WORLD
Save using save_report.""",

    AgentType.DOSAGE: """You are a clinical pharmacist specializing in dosing.
Given a drug name:
1. Use search_dailymed to get the label
2. Use get_label_section(dosage_and_administration) for standard dosing
3. Use get_label_section(clinical_pharmacology) for PK/PD data
4. Use search_vector_store for any additional dosing context
5. Flag unusual dosing considerations using flag_compliance_issue
Produce: STANDARD DOSE | SPECIAL POPULATIONS | RENAL/HEPATIC ADJUSTMENTS
Save using save_report.""",

    AgentType.COMPLIANCE: """You are an FDA regulatory compliance specialist.
Given a drug name:
1. Use search_dailymed to get the label
2. Use get_label_section(boxed_warning) — flag if present (HIGH severity)
3. Use get_label_section(contraindications) — flag critical ones
4. Use get_label_section(warnings) — flag REMS or serious warnings
5. Use search_vector_store to cross-check marketing compliance
6. Use flag_compliance_issue for each issue found
Produce: BOXED WARNINGS | CONTRAINDICATIONS | REMS | COMPLIANCE SCORE (0-100)
Save using save_report.""",
}


# ══════════════════════════════════════════════════════════════════════════
# BASE AGENT
# ══════════════════════════════════════════════════════════════════════════

class PharmaAgentBase:

    def __init__(self, agent_type: AgentType, model: str = "gpt-4o"):
        from openai import OpenAI
        self._llm        = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model      = model
        self._agent_type = agent_type
        self._system     = PROMPTS[agent_type]

    def run(self, drug_name: str) -> AgentResult:
        print(f"\n🤖 [{self._agent_type.value.upper()}] Running for: {drug_name}")

        messages = [
            {"role": "system",  "content": self._system},
            {"role": "user",    "content": f"Drug: {drug_name}. Run your full analysis now."},
        ]

        sources    = []
        alerts     = []
        final_text = ""

        for iteration in range(12):
            resp = self._llm.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = resp.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                final_text = msg.content or ""
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"  [{iteration+1}] → {fn_name}({list(fn_args.keys())})")

                fn     = TOOL_REGISTRY.get(fn_name)
                result = fn(**fn_args) if fn else {"error": f"Unknown: {fn_name}"}

                # Collect sources
                if fn_name == "search_dailymed":
                    for r in result.get("results", []):
                        sources.append({"type": "dailymed", "title": r["title"], "set_id": r["set_id"]})
                if fn_name == "get_label_section" and result.get("found"):
                    sources.append({"type": "label_section", "section": fn_args.get("section", ""), "url": result.get("source_url", "")})
                if fn_name == "search_clinical_trials":
                    for t in result.get("trials", []):
                        sources.append({"type": "clinical_trial", "nct_id": t["nct_id"], "title": t["title"]})
                if fn_name == "flag_compliance_issue":
                    alerts.append(f"[{result.get('severity','?')}] {result.get('issue','')}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result),
                })

        return AgentResult(
            agent=self._agent_type.value,
            drug=drug_name,
            result=final_text,
            sources=sources,
            alerts=alerts,
        )


# ══════════════════════════════════════════════════════════════════════════
# MASTER AGENT — runs all 5 in sequence
# ══════════════════════════════════════════════════════════════════════════

class MasterPharmaAgent:

    def __init__(self):
        self._agents = {
            AgentType.SUMMARIZER: PharmaAgentBase(AgentType.SUMMARIZER),
            AgentType.CLINICAL:   PharmaAgentBase(AgentType.CLINICAL),
            AgentType.ADVERSE:    PharmaAgentBase(AgentType.ADVERSE),
            AgentType.DOSAGE:     PharmaAgentBase(AgentType.DOSAGE),
            AgentType.COMPLIANCE: PharmaAgentBase(AgentType.COMPLIANCE),
        }

    def run_one(self, drug: str, agent_type: AgentType) -> AgentResult:
        return self._agents[agent_type].run(drug)

    def run_all(self, drug: str) -> dict:
        results = {}
        all_alerts = []
        for agent_type, agent in self._agents.items():
            r = agent.run(drug)
            results[agent_type.value] = r.to_dict()
            all_alerts.extend(r.alerts)

        return {
            "drug":        drug,
            "run_at":      datetime.utcnow().isoformat(),
            "agents_run":  list(results.keys()),
            "total_alerts":len(all_alerts),
            "all_alerts":  all_alerts,
            "results":     results,
        }


# ══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Master Pharma Agent API",
    description="""
🧬 **All-in-One FDA Pharmaceutical Intelligence API**

Combines 5 specialized agents:
- **Summarizer** — FDA label summary (indication, dosage, warnings)
- **Clinical Trial** — ClinicalTrials.gov search + label evidence
- **Adverse Event** — FAERS reports + label cross-reference
- **Dosage Recommender** — dosing guidelines + special populations
- **Compliance Checker** — boxed warnings, REMS, compliance score

All agents search live FDA DailyMed data.
    """,
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Singleton agent
_master: Optional[MasterPharmaAgent] = None

def get_master() -> MasterPharmaAgent:
    global _master
    if _master is None:
        _master = MasterPharmaAgent()
    return _master


# ── Request models ────────────────────────────────────────────────────────

class SingleAgentRequest(BaseModel):
    drug:       str       = Field(..., example="warfarin")
    agent_type: AgentType = Field(..., example="summarizer")

    class Config:
        json_schema_extra = {"example": {"drug": "warfarin", "agent_type": "summarizer"}}


class MasterRequest(BaseModel):
    drug: str = Field(..., example="warfarin",
                      description="Drug name to run all 5 agents on")


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "Master Pharma Agent API",
        "agents":  [a.value for a in AgentType if a != AgentType.ALL],
        "time":    datetime.utcnow().isoformat(),
    }


@app.post("/agent/run")
async def run_single_agent(req: SingleAgentRequest):
    """
    Run ONE specific agent for a drug.

    agent_type options:
    - **summarizer**     — FDA label summary
    - **clinical_trial** — Clinical trial search
    - **adverse_event**  — FAERS adverse events
    - **dosage**         — Dosage recommendations
    - **compliance**     — Regulatory compliance check
    """
    try:
        master = get_master()
        result = master.run_one(req.drug.strip().lower(), req.agent_type)
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/run-all")
async def run_all_agents(req: MasterRequest):
    """
    Run ALL 5 agents for a drug in sequence.

    Returns a combined report from:
    Summarizer + Clinical Trial + Adverse Event + Dosage + Compliance
    """
    try:
        master = get_master()
        result = master.run_all(req.drug.strip().lower())
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/types")
async def list_agents():
    """List all available agent types with descriptions."""
    return {
        "agents": [
            {"type": "summarizer",     "description": "FDA label summary — indication, dosage, warnings"},
            {"type": "clinical_trial", "description": "ClinicalTrials.gov search + label clinical evidence"},
            {"type": "adverse_event",  "description": "FAERS adverse event reports + label cross-reference"},
            {"type": "dosage",         "description": "Dosing guidelines + renal/hepatic adjustments"},
            {"type": "compliance",     "description": "Boxed warnings, REMS, contraindications, compliance score"},
        ]
    }