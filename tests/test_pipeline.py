"""
Basic unit tests — run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
from src.ingestion.fda_loader import FDALabelLoader, DrugLabelChunk
from src.generation.rag_pipeline import OutputMode, PharmaRAGPipeline, RAGResponse


# ── FDALabelLoader tests ──────────────────────────────────────────────────

class TestFDALabelLoader:

    def test_chunk_text_basic(self):
        loader = FDALabelLoader(chunk_size=5, chunk_overlap=1)
        text = "one two three four five six seven eight nine ten"
        chunks = list(loader._chunk_text(text))
        assert len(chunks) > 1
        assert "one" in chunks[0]

    def test_chunk_text_empty(self):
        loader = FDALabelLoader()
        chunks = list(loader._chunk_text(""))
        assert chunks == []

    def test_chunk_text_short(self):
        loader = FDALabelLoader(chunk_size=100)
        text = "short text"
        chunks = list(loader._chunk_text(text))
        assert len(chunks) == 1
        assert chunks[0] == "short text"

    def test_drug_label_chunk_to_dict(self):
        chunk = DrugLabelChunk(
            drug_name="Warfarin",
            set_id="abc123",
            section_code="34070-3",
            section_name="contraindications",
            text="Do not use with active bleeding.",
            chunk_index=0,
            source_url="https://example.com",
            metadata={"is_boxed_warning": False},
        )
        d = chunk.to_dict()
        assert d["drug_name"] == "Warfarin"
        assert d["section_name"] == "contraindications"
        assert d["is_boxed_warning"] == False


# ── OutputMode tests ──────────────────────────────────────────────────────

class TestOutputMode:

    def test_all_modes_exist(self):
        assert OutputMode.CLINICAL == "clinical"
        assert OutputMode.MARKETING == "marketing"
        assert OutputMode.PATIENT == "patient"
        assert OutputMode.REGULATORY == "regulatory"

    def test_mode_from_string(self):
        mode = OutputMode("clinical")
        assert mode == OutputMode.CLINICAL


# ── Compliance check tests ────────────────────────────────────────────────

class TestComplianceCheck:

    def setup_method(self):
        """Create pipeline with mocked OpenAI client."""
        with patch("src.generation.rag_pipeline.OpenAI"):
            self.pipeline = PharmaRAGPipeline.__new__(PharmaRAGPipeline)
            self.pipeline._model = "gpt-4o"

    def test_no_flags_clean_answer(self):
        answer = "Warfarin is contraindicated in patients with active hemorrhage."
        flags = self.pipeline._compliance_check(answer, OutputMode.CLINICAL)
        assert flags == []

    def test_superlative_flag(self):
        answer = "This is the best anticoagulant available."
        flags = self.pipeline._compliance_check(answer, OutputMode.CLINICAL)
        assert any("SUPERLATIVE" in f for f in flags)

    def test_marketing_missing_fair_balance(self):
        answer = "Warfarin reduces stroke risk in AFib patients. Ask your doctor."
        flags = self.pipeline._compliance_check(answer, OutputMode.MARKETING)
        assert any("FAIR_BALANCE" in f for f in flags)

    def test_marketing_with_fair_balance(self):
        answer = "Warfarin reduces stroke risk. Warning: risk of bleeding. Contraindicated in active hemorrhage."
        flags = self.pipeline._compliance_check(answer, OutputMode.MARKETING)
        assert not any("FAIR_BALANCE" in f for f in flags)

    def test_off_label_flag(self):
        answer = "This drug is sometimes used off-label for migraines."
        flags = self.pipeline._compliance_check(answer, OutputMode.CLINICAL)
        assert any("OFF_LABEL" in f for f in flags)
