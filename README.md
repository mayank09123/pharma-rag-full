# FDA Drug Label RAG Pipeline

A production-grade Retrieval-Augmented Generation system built on FDA DailyMed drug label data.

## Quick Start (Mac)

```bash
# 1. One-command setup
bash setup.sh

# 2. Add your OpenAI API key to .env
# Edit .env → OPENAI_API_KEY=sk-your-key-here

# 3. Activate virtual environment
source venv/bin/activate

# 4. Ingest FDA drug labels
python scripts/ingest.py

# 5. Query the pipeline
python scripts/query.py

# 6. Run evaluation
python scripts/evaluate.py
```

## Project Structure

```
pharma-rag/
├── src/
│   ├── ingestion/
│   │   └── fda_loader.py        # FDA DailyMed API + XML parser
│   ├── retrieval/
│   │   └── vector_store.py      # ChromaDB + embeddings + hybrid search
│   ├── generation/
│   │   └── rag_pipeline.py      # RAG pipeline + compliance scan
│   ├── evaluation/
│   │   └── evaluator.py         # Faithfulness, precision, MRR metrics
│   └── agents/
│       └── pharma_agents.py     # LLM agents for automation
├── scripts/
│   ├── ingest.py                # Load FDA labels → vector store
│   ├── query.py                 # Interactive query
│   └── evaluate.py              # Run evaluation suite
├── mlops/
│   ├── docker/
│   │   └── docker-compose.yml   # Full stack deployment
│   └── ci_cd/
│       └── github-actions.yml   # CI/CD pipeline
├── .env.example                 # Copy to .env and add your API key
├── requirements.txt
└── setup.sh                     # One-command Mac setup
```

## Output Modes

Change `MODE` in `scripts/query.py` to switch between:

| Mode | Use Case |
|------|----------|
| `CLINICAL` | HCP communication, medical information |
| `MARKETING` | Promotional copy (requires fair balance) |
| `PATIENT` | Plain-language patient education |
| `REGULATORY` | Regulatory summaries and dossiers |

## Customizing Drug List

Edit `DRUGS` in `scripts/ingest.py`:
```python
DRUGS = ["warfarin", "metformin", "lisinopril"]
```

Any drug name searchable on [DailyMed](https://dailymed.nlm.nih.gov) will work.

## Requirements

- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- ~$0.10–0.50 in OpenAI credits per session
