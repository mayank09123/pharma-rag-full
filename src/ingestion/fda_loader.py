"""
FDA DailyMed Drug Label Ingestion Pipeline
Downloads, parses, and chunks SPL (Structured Product Labeling) XML documents.
"""

import os
import re
import requests
import xml.etree.ElementTree as ET
from typing import Generator
from dataclasses import dataclass, field

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

SPL_SECTIONS = {
    "34066-1": "boxed_warning",
    "34067-9": "indications_and_usage",
    "34068-7": "dosage_and_administration",
    "34069-5": "how_supplied",
    "34070-3": "contraindications",
    "34071-1": "warnings",
    "34084-4": "adverse_reactions",
    "34073-7": "drug_interactions",
    "34074-5": "carcinogenesis",
    "42229-5": "special_populations",
    "34089-3": "description",
    "34090-1": "clinical_pharmacology",
    "34092-7": "clinical_studies",
    "34093-5": "references",
    "42230-3": "patient_information",
}


@dataclass
class DrugLabelChunk:
    """A single chunk of drug label content with rich metadata."""
    drug_name: str
    set_id: str
    section_code: str
    section_name: str
    text: str
    chunk_index: int
    source_url: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "drug_name": self.drug_name,
            "set_id": self.set_id,
            "section_code": self.section_code,
            "section_name": self.section_name,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "source_url": self.source_url,
            **self.metadata,
        }


class FDALabelLoader:
    """
    Fetches FDA drug labels from DailyMed API and converts them
    into structured chunks suitable for vector embedding.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ns = {
            "v3": "urn:hl7-org:v3",
        }

    def search_drugs(self, drug_name: str, limit: int = 5) -> list:
        """Search DailyMed for drug labels matching a name."""
        url = f"{DAILYMED_BASE}/spls.json"
        params = {"drug_name": drug_name, "pagesize": limit}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])

    def load_label_by_setid(self, set_id: str) -> list:
        """Download and parse a drug label by its SPL set ID."""
        xml_url = f"{DAILYMED_BASE}/spls/{set_id}.xml"
        resp = requests.get(xml_url, timeout=30)
        resp.raise_for_status()
        return list(self._parse_spl_xml(resp.text, set_id, xml_url))

    def load_multiple(self, drug_names: list) -> list:
        """Load labels for a list of drug names (first result each)."""
        all_chunks = []
        for name in drug_names:
            try:
                results = self.search_drugs(name, limit=1)
                if results:
                    set_id = results[0]["setid"]
                    chunks = self.load_label_by_setid(set_id)
                    all_chunks.extend(chunks)
                    print(f"  ✓ Loaded {len(chunks)} chunks for '{name}' (setid={set_id})")
                else:
                    print(f"  ✗ No results found for '{name}'")
            except Exception as e:
                print(f"  ✗ Error loading '{name}': {e}")
        return all_chunks

    def _parse_spl_xml(self, xml_text: str, set_id: str, source_url: str) -> Generator:
        root = ET.fromstring(xml_text)

        # Extract drug name from title
        title_el = root.find(".//v3:title", self.ns)
        drug_name = title_el.text.strip() if (title_el is not None and title_el.text) else "Unknown"
        drug_name = re.sub(r"\s+", " ", drug_name)

        # Walk all sections
        for section in root.findall(".//v3:section", self.ns):
            code_el = section.find("v3:code", self.ns)
            if code_el is None:
                continue
            section_code = code_el.get("code", "")
            section_name = SPL_SECTIONS.get(
                section_code,
                code_el.get("displayName", "unknown_section").lower().replace(" ", "_"),
            )

            raw_text = self._extract_text(section)
            if not raw_text.strip():
                continue

            for idx, chunk_text in enumerate(self._chunk_text(raw_text)):
                yield DrugLabelChunk(
                    drug_name=drug_name,
                    set_id=set_id,
                    section_code=section_code,
                    section_name=section_name,
                    text=chunk_text,
                    chunk_index=idx,
                    source_url=source_url,
                    metadata={"is_boxed_warning": section_code == "34066-1"},
                )

    def _extract_text(self, element: ET.Element) -> str:
        """Recursively extract clean text from an XML element."""
        parts = []
        if element.text:
            parts.append(element.text.strip())
        for child in element:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ("code", "subject", "excerpt"):
                continue
            parts.append(self._extract_text(child))
            if child.tail:
                parts.append(child.tail.strip())
        return " ".join(p for p in parts if p)

    def _chunk_text(self, text: str) -> Generator:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        if not words:
            return
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            yield " ".join(words[start:end])
            if end == len(words):
                break
            start += self.chunk_size - self.chunk_overlap
