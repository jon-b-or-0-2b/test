# TEVV Test Suite

**Testing, Evaluation, Validation, Verification** - A comprehensive, reusable testing framework for single and multi-agent systems.

## Overview

TEVV is a framework-agnostic evaluation toolkit designed to test and validate responses from any agent or chatbot system. Whether you're working with a single-agent system or a complex multi-agent framework, TEVV provides standardized evaluation metrics to ensure quality and accuracy.

### Key Features

- ✅ **Framework-Agnostic**: Works with any agent/chatbot system
- ✅ **Multiple Evaluation Metrics**: Cosine similarity, number matching, exact match, keyword checking, and more
- ✅ **Multi-Agent Support**: Evaluate multiple agents simultaneously
- ✅ **Flexible Input Handling**: Accepts both string and dictionary responses
- ✅ **Comprehensive Reporting**: Export results in JSON or text format
- ✅ **Robust Error Handling**: Graceful fallbacks and detailed error reporting

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced cosine similarity calculations (recommended):
```bash
pip install scikit-learn>=1.3.0
```

For advanced text processing (optional):
```bash
pip install nltk>=3.8.0
python -m nltk.downloader punkt stopwords
```

## Quick Start

### Basic Usage

```python
from tevv import evaluate_cosine_similarity, evaluate_numbers

# Evaluate cosine similarity
answer = "The solar panel system requires 500 square feet of land area."
gold_standard = "The solar installation needs 500 sq ft of land."

result = evaluate_cosine_similarity(answer, gold_standard)
print(f"Similarity Score: {result['similarity_score']:.3f}")
# Output: Similarity Score: 0.122

# Evaluate numbers
answer = "The system requires 500 square feet and costs $25,000."
gold_standard = "Land requirement: 500 sq ft. Total cost: $25000."

result = evaluate_numbers(answer, gold_standard)
print(f"Numbers Match: {result['numbers_match']}")
print(f"Match Score: {result['match_score']:.3f}")
# Output: Numbers Match: True
#         Match Score: 1.000
```

## Core Evaluation Functions

### 1. Cosine Similarity Evaluation

Evaluates semantic similarity between an answer and gold standard using TF-IDF vectorization.

```python
from tevv import evaluate_cosine_similarity

result = evaluate_cosine_similarity(
    answer="The solar panel system requires 500 square feet.",
    gold_standard="The installation needs 500 sq ft.",
    use_tfidf=True  # Use TF-IDF (requires sklearn) or simple word count
)

print(result)
# {
#     "similarity_score": 0.122,
#     "method": "tfidf",
#     "answer_text": "the solar panel system requires 500 square feet",
#     "gold_text": "the installation needs 500 sq ft",
#     "status": "success"
# }
```

**Parameters:**
- `answer`: The answer to evaluate (string or dict)
- `gold_standard`: The gold standard answer (string or dict)
- `use_tfidf`: If True, use TF-IDF vectorization (requires sklearn). Default: True

**Returns:**
- `similarity_score`: Float between 0 and 1
- `method`: Method used ("tfidf" or "simple_word_count")
- `answer_text`: Extracted text from answer
- `gold_text`: Extracted text from gold standard
- `status`: "success" or "error"

### 2. Numbers Evaluation

Extracts and compares all numeric values between answer and gold standard.

```python
from tevv import evaluate_numbers

result = evaluate_numbers(
    answer="Cost is $25,000 and area is 500 sq ft.",
    gold_standard="Total: $25000. Land: 500 square feet.",
    tolerance=0.01,  # Absolute or relative tolerance
    relative_tolerance=True  # Use percentage-based tolerance
)

print(result)
# {
#     "numbers_match": True,
#     "match_score": 1.0,
#     "answer_numbers": [500.0, 25000.0],
#     "gold_numbers": [500.0, 25000.0],
#     "matched_numbers": [(500.0, 500.0), (25000.0, 25000.0)],
#     "unmatched_answer": [],
#     "unmatched_gold": [],
#     "detailed_comparison": [...],
#     "status": "success"
# }
```

**Parameters:**
- `answer`: The answer to evaluate (string or dict)
- `gold_standard`: The gold standard answer (string or dict)
- `tolerance`: Absolute tolerance for number comparison (default: 0.01)
- `relative_tolerance`: If True, use relative tolerance (percentage-based). Default: True

**Returns:**
- `numbers_match`: Boolean indicating if all numbers match
- `match_score`: Float (0-1) representing percentage of numbers that match
- `answer_numbers`: List of extracted numbers from answer
- `gold_numbers`: List of extracted numbers from gold standard
- `matched_numbers`: List of matched number pairs
- `unmatched_answer`: Numbers in answer not found in gold standard
- `unmatched_gold`: Numbers in gold standard not found in answer
- `detailed_comparison`: Detailed comparison for each number

### 3. Exact Match Evaluation

Checks for exact string match between answer and gold standard.

```python
from tevv import evaluate_exact_match

result = evaluate_exact_match(
    answer="The cost is $25,000",
    gold_standard="The cost is $25,000",
    case_sensitive=False
)

print(result)
# {
#     "exact_match": True,
#     "answer_text": "the cost is $25,000",
#     "gold_text": "the cost is $25,000",
#     "status": "success"
# }
```

### 4. Keyword Evaluation

Checks if answer contains required keywords.

```python
from tevv import evaluate_contains_keywords

result = evaluate_contains_keywords(
    answer="The solar panel system requires 500 square feet.",
    required_keywords=["solar", "panel", "cost", "land"],
    case_sensitive=False
)

print(result)
# {
#     "found_keywords": ["solar", "panel", "land"],
#     "missing_keywords": ["cost"],
#     "coverage_score": 0.75,
#     "all_found": False,
#     "status": "success"
# }
```

### 5. JSON Structure Validation

Validates that answer matches expected JSON schema structure.

```python
from tevv import evaluate_json_structure

answer = {
    "domain": "solar",
    "extracted": {
        "land_required": {"value": "500 sq ft", "source": "page 1"}
    },
    "confidence": "high"
}

expected_schema = {
    "domain": "string",
    "extracted": {
        "land_required": {
            "value": "string",
            "source": "string"
        }
    },
    "confidence": "string"
}

result = evaluate_json_structure(answer, expected_schema)

print(result)
# {
#     "is_valid": True,
#     "missing_keys": [],
#     "extra_keys": [],
#     "type_mismatches": [],
#     "valid_keys": ["domain", "extracted", "extracted.land_required", ...],
#     "validity_score": 1.0,
#     "status": "success"
# }
```

### 6. Response Time Evaluation

Evaluates response time performance.

```python
from tevv import evaluate_response_time

result = evaluate_response_time(
    response_time_seconds=15.5,
    max_acceptable_time=30.0
)

print(result)
# {
#     "response_time_seconds": 15.5,
#     "max_acceptable_time": 30.0,
#     "is_acceptable": True,
#     "performance_score": 0.483,
#     "status": "success"
# }
```

## Comprehensive Evaluation

Run multiple evaluation metrics at once and get an overall score.

```python
from tevv import evaluate_comprehensive

result = evaluate_comprehensive(
    answer="The solar panel system requires 500 square feet and costs $25,000.",
    gold_standard="Land requirement: 500 sq ft. Total cost: $25000.",
    expected_schema={
        "domain": "string",
        "extracted": {
            "land_required": {"value": "string", "source": "string"},
            "cost": {"value": "string", "source": "string"}
        }
    },
    required_keywords=["solar", "cost", "land"],
    number_tolerance=0.01,
    response_time=15.5,
    max_response_time=30.0
)

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Cosine Similarity: {result['cosine_similarity']['similarity_score']:.3f}")
print(f"Numbers Match Score: {result['numbers_evaluation']['match_score']:.3f}")
print(f"Exact Match: {result['exact_match']['exact_match']}")
print(f"Keywords Coverage: {result['keyword_evaluation']['coverage_score']:.3f}")
```

**Score Weights:**
- Cosine Similarity: 30%
- Numbers Evaluation: 30%
- Exact Match: 10%
- Keyword Evaluation: 10%
- Schema Validation: 10%
- Response Time: 10%

## Multi-Agent Evaluation

Evaluate results from multiple agents simultaneously.

```python
from tevv import evaluate_multi_agent

# Agent results
agent_results = [
    {
        "agent_name": "agent_1",
        "response": "The system requires 500 sq ft.",
        "response_time": 12.5
    },
    {
        "agent_name": "agent_2",
        "response": "Land area needed: 500 square feet.",
        "response_time": 18.3
    }
]

# Gold standards (can be single dict or per-agent)
gold_standards = {
    "agent_1": "Land requirement: 500 sq ft.",
    "agent_2": "Land area needed: 500 square feet."
}

# Evaluation configuration
config = {
    "number_tolerance": 0.01,
    "max_response_time": 30.0,
    "required_keywords": {
        "agent_1": ["land", "500"],
        "agent_2": ["land", "500"]
    }
}

result = evaluate_multi_agent(agent_results, gold_standards, config)

print(f"Average Score: {result['average_score']:.3f}")
print(f"Best Agent: {result['best_agent']}")
print(f"Total Agents: {result['total_agents']}")

# Access individual agent evaluations
for agent_name, evaluation in result['agent_evaluations'].items():
    print(f"{agent_name}: {evaluation['overall_score']:.3f}")
```

## Batch Evaluation

Evaluate multiple test cases at once.

```python
from tevv import evaluate_batch

test_cases = [
    {
        "answer": "The cost is $25,000",
        "gold_standard": "Total cost: $25000",
        "required_keywords": ["cost"]
    },
    {
        "answer": "Land area: 500 sq ft",
        "gold_standard": "Land requirement: 500 square feet",
        "required_keywords": ["land"]
    }
]

result = evaluate_batch(test_cases)

print(f"Average Score: {result['statistics']['average_score']:.3f}")
print(f"Total Tests: {result['statistics']['total_tests']}")
print(f"Min Score: {result['statistics']['min_score']:.3f}")
print(f"Max Score: {result['statistics']['max_score']:.3f}")
print(f"Std Deviation: {result['statistics']['std_deviation']:.3f}")

# Access individual test results
for test_result in result['test_results']:
    print(f"Test {test_result['test_case_id']}: {test_result['overall_score']:.3f}")
```

## Exporting Results

Export evaluation results to a file.

```python
from tevv import export_evaluation_report, evaluate_comprehensive

result = evaluate_comprehensive(
    answer="The system requires 500 sq ft.",
    gold_standard="Land: 500 square feet."
)

# Export as JSON
export_evaluation_report(result, "evaluation_results.json", format="json")

# Export as text
export_evaluation_report(result, "evaluation_results.txt", format="txt")
```

## Working with Dictionary Responses

TEVV automatically handles dictionary responses by extracting text from common keys:

```python
# Dictionary response (e.g., from agent_toolkit)
answer_dict = {
    "domain": "solar",
    "extracted": {
        "land_required": {"value": "500 sq ft", "source": "page 1"},
        "cost": {"value": "$25,000", "source": "page 2"}
    },
    "confidence": "high"
}

# TEVV will automatically extract text from "extracted" field
result = evaluate_cosine_similarity(
    answer=answer_dict,
    gold_standard="Land: 500 sq ft. Cost: $25000."
)

# Or extract numbers from dictionary
result = evaluate_numbers(
    answer=answer_dict,
    gold_standard="Land: 500 sq ft. Cost: $25000."
)
```

## Best Practices

### 1. Choose Appropriate Metrics

- **Cosine Similarity**: Best for semantic similarity and paraphrasing
- **Numbers Evaluation**: Essential for quantitative data extraction
- **Exact Match**: Use for strict, verbatim requirements
- **Keyword Evaluation**: Good for ensuring specific terms are present
- **Schema Validation**: Critical for structured data validation

### 2. Set Appropriate Tolerances

```python
# For financial data (high precision)
result = evaluate_numbers(answer, gold, tolerance=0.001, relative_tolerance=True)

# For measurements (moderate precision)
result = evaluate_numbers(answer, gold, tolerance=0.01, relative_tolerance=True)

# For counts (exact match)
result = evaluate_numbers(answer, gold, tolerance=0.0, relative_tolerance=False)
```

### 3. Use Comprehensive Evaluation for Production

```python
# Production evaluation should include multiple metrics
result = evaluate_comprehensive(
    answer=agent_response,
    gold_standard=gold_standard,
    expected_schema=schema,
    required_keywords=keywords,
    number_tolerance=0.01,
    response_time=response_time
)

# Set threshold for passing
if result['overall_score'] >= 0.8:
    print("✅ Test passed")
else:
    print("❌ Test failed")
```

### 4. Handle Errors Gracefully

```python
result = evaluate_cosine_similarity(answer, gold_standard)

if result['status'] == 'error':
    print(f"Error: {result.get('error', 'Unknown error')}")
    # Handle error appropriately
else:
    score = result['similarity_score']
    # Use score
```

## Integration with agent_toolkit

TEVV works seamlessly with the agent_toolkit:

```python
from agent_toolkit import agent_builder
from tevv import evaluate_comprehensive

# Run agent
result = agent_builder(
    is_image_input=True,
    model="gpt5-mini",
    data_input="solar_page_1.png",
    prompt=prompt,
    schema=schema
)

# Evaluate with TEVV
evaluation = evaluate_comprehensive(
    answer=result,
    gold_standard=gold_standard,
    expected_schema=schema
)

print(f"Agent Score: {evaluation['overall_score']:.3f}")
```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `evaluate_cosine_similarity()` | Evaluate semantic similarity using cosine similarity |
| `evaluate_numbers()` | Extract and compare numeric values |
| `evaluate_exact_match()` | Check for exact string match |
| `evaluate_contains_keywords()` | Check for required keywords |
| `evaluate_json_structure()` | Validate JSON schema structure |
| `evaluate_response_time()` | Evaluate response time performance |
| `evaluate_comprehensive()` | Run multiple metrics and get overall score |
| `evaluate_multi_agent()` | Evaluate multiple agents simultaneously |
| `evaluate_batch()` | Evaluate batch of test cases |
| `export_evaluation_report()` | Export results to file |

### Helper Functions (Internal)

- `_extract_text()`: Extract text from string or dict
- `_normalize_text()`: Normalize text for comparison
- `_extract_numbers()`: Extract numbers from text
- `_simple_cosine_similarity()`: Simple cosine similarity calculation
- `_get_all_keys()`: Get all keys from nested dict

## Examples

See the `__main__` section in `tevv.py` for example usage:

```bash
python tevv.py
```

## Requirements

- Python 3.7+
- Standard library: `json`, `re`, `logging`, `pathlib`, `typing`, `math`
- Optional: `scikit-learn>=1.3.0` (for TF-IDF cosine similarity)
- Optional: `nltk>=3.8.0` (for advanced text processing)

## License

This is part of the agent toolkit project. Use as needed for your team's best practices.

## Contributing

When adding new evaluation metrics:

1. Follow the existing function signature pattern
2. Return a dictionary with `status` field ("success" or "error")
3. Include error handling with try/except
4. Add docstrings with parameter and return descriptions
5. Update this README with examples

## Support

For issues or questions, refer to the main project documentation or contact your team lead.

---

**TEVV Test Suite** - Making agent evaluation standardized and reusable across your team.

