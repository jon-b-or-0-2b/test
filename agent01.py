"""
Agent01 - Simple document processor
"""

from agent_toolkit import process_document_pages

# Configuration - set these paths manually
PROMPT_PATH = "land_and_layout.py"
SCHEMA_PATH = "prompt1_schema.json"
DOCUMENT_PATH = "solar.pdf"
MODEL = "gpt5-mini"
JUDGE_MODELS = ["gpt-4o"]
OUTPUT_CSV = "results.csv"

if __name__ == "__main__":
    process_document_pages(
        document_path=DOCUMENT_PATH,
        prompt_path=PROMPT_PATH,
        schema_path=SCHEMA_PATH,
        model=MODEL,
        judge_models=JUDGE_MODELS,
        output_csv=OUTPUT_CSV
    )
