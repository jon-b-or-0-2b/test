# Agent Toolkit - Multi-Agent Document Processing System

A comprehensive toolkit for processing documents (PDF, DOCX, TXT) using multiple LLM agents with extraction, validation, and evaluation capabilities.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Agent Toolkit Overview](#agent-toolkit-overview)
- [Agent01 - Single Agent Processing](#agent01---single-agent-processing)
- [Agent02 - Multi-Agent Round-Robin Processing](#agent02---multi-agent-round-robin-processing)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)

## Features

- **Document Processing**: Support for PDF, DOCX, and TXT files
- **Image Conversion**: Convert PDF pages to images for vision-based LLM processing
- **Multi-LLM Support**: OpenAI (GPT models), Anthropic (Claude), Google (Gemini)
- **Schema Validation**: Automatic validation against JSON schemas
- **LLM-as-Judge**: Multi-model evaluation of extraction quality
- **Round-Robin Processing**: Process multiple agents per page with early stopping
- **Confidence Scoring**: Automatic confidence assignment based on extraction quality
- **DataFrame Export**: Clean CSV output with structured extraction results

## Installation

### Required Packages

```bash
pip install openai PyPDF2 python-docx pdf2image pandas pillow anthropic google-generativeai PyMuPDF
```

### Optional: Poppler for pdf2image

If using `pdf2image`, install Poppler:
- **Windows**: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases) and add to PATH
- **Linux**: `sudo apt-get install poppler-utils`
- **macOS**: `brew install poppler`

**Note**: PyMuPDF is used as a fallback if Poppler is not available.

### API Keys

Create a `secrete.py` file with your API keys:

```python
OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"  # Optional
GOOGLE_API_KEY = "your-google-key"  # Optional
```

## Agent Toolkit Overview

The `agent_toolkit.py` provides core functionality for document processing:

### Key Functions

#### Document Loading
- `document_loader(file_path, is_image=False)`: Loads PDF/DOCX/TXT files
  - If `is_image=True`: Converts PDF pages to images (saves to `{filename}/{filename}_page_N.png`)
  - If `is_image=False`: Extracts text from documents

#### LLM Wrappers
- `llm_wrapper_text(model, prompt, system_prompt=None)`: Text-based LLM calls
- `llm_wrapper_image(model, image_path, prompt, system_prompt=None)`: Image-based LLM calls

**Supported Models**:
- OpenAI: `gpt5`, `gpt5-mini`, `gpt-4o`
- Anthropic: `claude-sonnet-4.1`, `claude-sonnet-4.5`
- Google: `gemini-flash`, `gemini-flash-lite-2.5`

#### Response Processing
- `llm_cleaner(response_text, json_schema)`: Cleans and validates LLM JSON responses
- `llm_as_judge(original_prompt, json_response, judge_models, json_schema)`: Evaluates extraction quality
- `_verify_all_extracted_items(json_response, json_schema)`: Programmatically verifies all schema items have valid values

#### Agent Building
- `agent_builder(is_image_input, model, data_input, prompt, schema, judge_models)`: Main agent orchestrator
- `round_robin_wrapper(document_pages, agents, is_image, parallel_process_flag, num_workers)`: Multi-agent page processing

#### Data Export
- `extract_to_dataframe(results, json_schema)`: Converts extraction results to DataFrame
- `schema_to_dataframe(results, json_schema)`: Converts full results to DataFrame

## Agent01 - Single Agent Processing

**File**: `agent01.py`

Simple single-agent document processor that processes pages one at a time and stops when all required information is found.

### Usage

```python
# Configuration at top of file
PROMPT_PATH = "land_and_layout.py"      # Path to prompt file
SCHEMA_PATH = "prompt1_schema.json"      # Path to schema file
DOCUMENT_PATH = "solar.pdf"              # Document to process
MODEL = "gpt5-mini"                       # LLM model to use
JUDGE_MODELS = ["gpt-4o"]                 # Models for judging
OUTPUT_CSV = "results.csv"                # Output filename
```

### How It Works

1. **Loads prompt** from a Python file (looks for variables: `prompt`, `prompt1`, `PROMPT`, `prompt_text`)
2. **Loads schema** from JSON file
3. **Converts PDF to images** (saves each page as PNG)
4. **Processes pages sequentially**:
   - Saves page as image
   - Calls LLM with page image and prompt
   - Validates response against schema
   - Checks if all required items found (`found=True`)
   - **Stops** when all items found
5. **Exports results** to CSV with columns: `domain`, `value_type`, `value`, `source`, `confidence`

### Example Prompt File (`land_and_layout.py`)

```python
prompt1 = """
###ROLE
You are the Land & Layout Extraction Agent.
Your job is to read the provided structured solar-plant document and extract only the information related to land area, installation footprint, and physical layout elements.

###RULES
Search the document only for information related to:
- land requirement

Extract precise factual values exactly as written in the document (no paraphrasing). Do not make up information.

###OUTPUT FORMAT
Output the extraction in the following JSON format:
{
  "domain": "domain name",
  "extracted": {
    "land_required": {value: "...", source: "page X"}
  },
  "confidence": "",
  "confidence_keyword": "",
  "summary": ""
}

If any requested item is not present, return "not found".
"""
```

### Example Schema File (`prompt1_schema.json`)

```json
{
  "domain": "domain name",
  "extracted": {
    "land_required": {"value": "...", "source": "page X"}
  },
  "confidence": "",
  "confidence_keyword": "",
  "summary": ""
}
```

### Running Agent01

```bash
python agent01.py
```

## Agent02 - Multi-Agent Round-Robin Processing

**File**: `agent02.py`

Multi-agent processor that runs multiple agents (with different prompts/schemas) on each page using round-robin logic.

### Usage

```python
# Configuration at top of file
PROMPT1_PATH = "land_and_layout.py"      # First agent prompt
SCHEMA1_PATH = "prompt1_schema copy.json" # First agent schema
PROMPT2_PATH = "cables.py"                # Second agent prompt
SCHEMA2_PATH = "prompt2_schema.json"      # Second agent schema
DOCUMENT_PATH = "solar.pdf"               # Document to process
MODEL = "gpt5-mini"                        # LLM model to use
JUDGE_MODELS = ["gpt-4o"]                 # Models for judging
OUTPUT_CSV = "results_multi_agent.csv"    # Output filename
PARALLEL_PROCESS = False                  # Use parallel processing
NUM_WORKERS = 10                          # Number of parallel workers
```

### How It Works

1. **Loads multiple prompts and schemas** for different agents
2. **Converts PDF to images** (saves each page as PNG)
3. **Processes pages with round-robin logic**:
   - Both agents process each page
   - Each agent uses its own prompt and schema
   - Agents that find all required information (`found=True`) are removed
   - Processing continues for remaining agents
   - **Stops** when all agents complete or all pages processed
4. **Combines results** from all agents
5. **Exports results** to CSV with columns: `domain`, `value_type`, `value`, `source`, `confidence`, `agent`

### Round-Robin Logic

- **Page 1**: Both agents process → Check if found → Remove completed agents
- **Page 2**: Remaining agents process → Check if found → Remove completed agents
- **Continue** until all agents complete or all pages processed
- **Only saves final results** when:
  - Agent found everything (`found=True`), OR
  - All pages processed (ran out of pages)

### Validation Rules

The system ensures **ALL schema items** must have valid values:
- All extracted items must be present
- All values must be non-empty (not "not found", not "nan", not empty)
- Both judge LLM and programmatic check must pass
- `found=True` only when ALL items are valid

### Running Agent02

```bash
python agent02.py
```

## Configuration

### Prompt Files

Prompt files are Python files containing a string variable with the prompt. Supported variable names:
- `prompt`
- `prompt1`
- `PROMPT`
- `prompt_text`

### Schema Files

Schema files are JSON files defining the expected extraction structure:

```json
{
  "domain": "domain name",
  "extracted": {
    "item1": {"value": "...", "source": "page X"},
    "item2": {"value": "...", "source": "page X"}
  },
  "confidence": "",
  "confidence_keyword": "",
  "summary": ""
}
```

### Page Number Handling

- **PDF Page Numbers**: Uses actual PDF page numbers (1, 2, 3...) from the PDF reader
- **Not Printed Page Numbers**: Ignores page numbers printed on the document itself
- **Source Field**: Automatically set to `"page X"` where X is the PDF page number

### Confidence Levels

Confidence is automatically set based on `confidence_keyword`:
- **"1"** (High): "clearly stated", "explicitly mentioned", "directly specified", "clearly indicates"
- **"0.6-0.99"** (Medium): "appears to be", "seems to indicate", "suggests", "indicates", "likely"
- **"0.1-0.59"** (Low): "possibly", "might be", "could be", "unclear", "ambiguous", "uncertain"

## Output Format

### CSV Columns

- **domain**: Domain name from extraction
- **value_type**: Key from the extracted object (e.g., "land_required", "DC_side")
- **value**: Extracted value
- **source**: Page number (e.g., "page 2")
- **confidence**: Confidence score ("1" for clearly stated, etc.)
- **agent**: Agent name (only in Agent02 multi-agent results)

### Filtering

The system automatically filters out:
- Rows with "not found" values
- Rows with empty/NaN values
- Rows with invalid data

Only final, valid results are saved to CSV.

## Troubleshooting

### PDF to Image Conversion Fails

**Error**: `PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?`

**Solution**: 
- Install PyMuPDF: `pip install PyMuPDF` (recommended, no external dependencies)
- OR install Poppler and add to PATH

### No Results Saved

**Possible Causes**:
- Agents didn't find all required information
- All extracted items have "not found" values
- Schema validation failed

**Check**:
- Review `agent_toolkit.log` for detailed logs
- Verify schema matches prompt requirements
- Check if document contains the required information

### Only One Agent's Results Saved (Agent02)

**Solution**: Fixed in latest version. The system now saves results from all agents when:
- All agents complete (all found everything), OR
- All pages processed

### "found" Always False

**Possible Causes**:
- Not all schema items have valid values
- Judge LLM is too strict
- Programmatic validation failing

**Check**:
- Review logs for `found_judge` and `found_programmatic` status
- Verify all extracted items have non-empty values
- Check schema structure matches response structure

## Logging

All operations are logged to:
- **Console**: Real-time progress and errors
- **agent_toolkit.log**: Detailed execution logs
- **agent01.log**: Agent01-specific logs (if used)

## Examples

### Example 1: Extract Land Requirements

```python
# agent01.py configuration
PROMPT_PATH = "land_and_layout.py"
SCHEMA_PATH = "prompt1_schema.json"
DOCUMENT_PATH = "solar.pdf"
MODEL = "gpt5-mini"
```

### Example 2: Extract Multiple Domains

```python
# agent02.py configuration
PROMPT1_PATH = "land_and_layout.py"
SCHEMA1_PATH = "prompt1_schema.json"
PROMPT2_PATH = "cables.py"
SCHEMA2_PATH = "prompt2_schema.json"
DOCUMENT_PATH = "solar.pdf"
MODEL = "gpt5-mini"
```

## License

This project is provided as-is for document processing and extraction tasks.


