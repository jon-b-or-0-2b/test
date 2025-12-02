"""
TEVV (Testing, Evaluation, Validation, Verification) Test Suite
A comprehensive, reusable testing framework for single and multi-agent systems.

This module provides evaluation metrics and testing utilities that work with
any agent/chatbot framework, whether single-agent or multi-agent.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from collections import Counter
import math

# Optional dependencies for advanced metrics
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    np = None

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    import nltk
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/stopwords')
    except LookupError:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def evaluate_cosine_similarity(
    answer: str,
    gold_standard: str,
    use_tfidf: bool = True
) -> Dict[str, Any]:
    """
    Evaluate cosine similarity between an answer and a gold standard answer.
    
    Args:
        answer: The answer to evaluate (can be string or dict)
        gold_standard: The gold standard answer (can be string or dict)
        use_tfidf: If True, use TF-IDF vectorization (requires sklearn).
                   If False, use simple word count vectors.
    
    Returns:
        Dictionary with:
        - similarity_score: float between 0 and 1
        - method: str indicating the method used
        - answer_text: str (extracted text from answer)
        - gold_text: str (extracted text from gold_standard)
        - status: str ("success" or "error")
        - error: str (if status is "error")
    """
    try:
        # Extract text from inputs (handle both string and dict)
        answer_text = _extract_text(answer)
        gold_text = _extract_text(gold_standard)
        
        if not answer_text or not gold_text:
            return {
                "similarity_score": 0.0,
                "method": "none",
                "answer_text": answer_text,
                "gold_text": gold_text,
                "status": "error",
                "error": "Empty answer or gold standard"
            }
        
        # Normalize text
        answer_text = _normalize_text(answer_text)
        gold_text = _normalize_text(gold_text)
        
        if use_tfidf and SKLEARN_AVAILABLE:
            # Use TF-IDF vectorization for better semantic similarity
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),  # Include unigrams and bigrams
                max_features=5000
            )
            
            try:
                vectors = vectorizer.fit_transform([answer_text, gold_text])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                method = "tfidf"
            except ValueError:
                # Fallback if vectorization fails
                similarity = _simple_cosine_similarity(answer_text, gold_text)
                method = "simple_word_count"
        else:
            # Simple word count cosine similarity
            similarity = _simple_cosine_similarity(answer_text, gold_text)
            method = "simple_word_count"
        
        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, float(similarity)))
        
        return {
            "similarity_score": similarity,
            "method": method,
            "answer_text": answer_text,
            "gold_text": gold_text,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_cosine_similarity: {str(e)}", exc_info=True)
        return {
            "similarity_score": 0.0,
            "method": "error",
            "answer_text": str(answer),
            "gold_text": str(gold_standard),
            "status": "error",
            "error": str(e)
        }


def evaluate_numbers(
    answer: Union[str, Dict[str, Any]],
    gold_standard: Union[str, Dict[str, Any]],
    tolerance: float = 0.01,
    relative_tolerance: bool = True
) -> Dict[str, Any]:
    """
    Evaluate all numbers returned in an answer and compare them to a gold standard.
    
    Extracts all numeric values from both answers and compares them.
    
    Args:
        answer: The answer to evaluate (string or dict)
        gold_standard: The gold standard answer (string or dict)
        tolerance: Absolute tolerance for number comparison
        relative_tolerance: If True, use relative tolerance (percentage-based)
    
    Returns:
        Dictionary with:
        - numbers_match: bool (True if all numbers match within tolerance)
        - match_score: float (0-1, percentage of numbers that match)
        - answer_numbers: List[float] (extracted numbers from answer)
        - gold_numbers: List[float] (extracted numbers from gold standard)
        - matched_numbers: List[Tuple[float, float]] (pairs of matched numbers)
        - unmatched_answer: List[float] (numbers in answer not in gold)
        - unmatched_gold: List[float] (numbers in gold not in answer)
        - detailed_comparison: List[Dict] (detailed comparison for each number)
        - status: str ("success" or "error")
    """
    try:
        # Extract numbers from both inputs
        answer_numbers = _extract_numbers(answer)
        gold_numbers = _extract_numbers(gold_standard)
        
        if not answer_numbers and not gold_numbers:
            return {
                "numbers_match": True,
                "match_score": 1.0,
                "answer_numbers": [],
                "gold_numbers": [],
                "matched_numbers": [],
                "unmatched_answer": [],
                "unmatched_gold": [],
                "detailed_comparison": [],
                "status": "success",
                "note": "No numbers found in either answer"
            }
        
        if not gold_numbers:
            return {
                "numbers_match": False,
                "match_score": 0.0,
                "answer_numbers": answer_numbers,
                "gold_numbers": [],
                "matched_numbers": [],
                "unmatched_answer": answer_numbers,
                "unmatched_gold": [],
                "detailed_comparison": [],
                "status": "success",
                "note": "No numbers in gold standard"
            }
        
        if not answer_numbers:
            return {
                "numbers_match": False,
                "match_score": 0.0,
                "answer_numbers": [],
                "gold_numbers": gold_numbers,
                "matched_numbers": [],
                "unmatched_answer": [],
                "unmatched_gold": gold_numbers,
                "detailed_comparison": [],
                "status": "success",
                "note": "No numbers in answer"
            }
        
        # Match numbers (find closest matches within tolerance)
        matched_pairs = []
        unmatched_answer = answer_numbers.copy()
        unmatched_gold = gold_numbers.copy()
        detailed_comparison = []
        
        # Sort by absolute value for better matching
        answer_sorted = sorted(enumerate(answer_numbers), key=lambda x: abs(x[1]))
        gold_sorted = sorted(enumerate(gold_numbers), key=lambda x: abs(x[1]))
        
        used_gold_indices = set()
        
        for ans_idx, ans_num in answer_sorted:
            best_match = None
            best_match_idx = None
            best_diff = float('inf')
            
            for gold_idx, gold_num in gold_sorted:
                if gold_idx in used_gold_indices:
                    continue
                
                diff = abs(ans_num - gold_num)
                
                # Check tolerance
                if relative_tolerance and gold_num != 0:
                    relative_diff = diff / abs(gold_num)
                    within_tolerance = relative_diff <= tolerance
                else:
                    within_tolerance = diff <= tolerance
                
                if within_tolerance and diff < best_diff:
                    best_match = gold_num
                    best_match_idx = gold_idx
                    best_diff = diff
            
            if best_match is not None:
                matched_pairs.append((ans_num, best_match))
                unmatched_answer.remove(ans_num)
                unmatched_gold.remove(best_match)
                used_gold_indices.add(best_match_idx)
                
                # Calculate relative difference
                if best_match != 0:
                    rel_diff = best_diff / abs(best_match) * 100
                else:
                    rel_diff = best_diff * 100 if best_diff != 0 else 0.0
                
                detailed_comparison.append({
                    "answer_number": ans_num,
                    "gold_number": best_match,
                    "absolute_difference": best_diff,
                    "relative_difference_percent": rel_diff,
                    "match": True
                })
            else:
                detailed_comparison.append({
                    "answer_number": ans_num,
                    "gold_number": None,
                    "absolute_difference": None,
                    "relative_difference_percent": None,
                    "match": False
                })
        
        # Add unmatched gold numbers
        for gold_num in unmatched_gold:
            detailed_comparison.append({
                "answer_number": None,
                "gold_number": gold_num,
                "absolute_difference": None,
                "relative_difference_percent": None,
                "match": False
            })
        
        # Calculate match score
        total_expected = len(gold_numbers)
        matched_count = len(matched_pairs)
        match_score = matched_count / total_expected if total_expected > 0 else 0.0
        
        # All numbers match if all gold numbers are matched
        numbers_match = len(unmatched_gold) == 0 and len(unmatched_answer) == 0
        
        return {
            "numbers_match": numbers_match,
            "match_score": match_score,
            "answer_numbers": answer_numbers,
            "gold_numbers": gold_numbers,
            "matched_numbers": matched_pairs,
            "unmatched_answer": unmatched_answer,
            "unmatched_gold": unmatched_gold,
            "detailed_comparison": detailed_comparison,
            "tolerance": tolerance,
            "relative_tolerance": relative_tolerance,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_numbers: {str(e)}", exc_info=True)
        return {
            "numbers_match": False,
            "match_score": 0.0,
            "answer_numbers": [],
            "gold_numbers": [],
            "matched_numbers": [],
            "unmatched_answer": [],
            "unmatched_gold": [],
            "detailed_comparison": [],
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Additional Evaluation Metrics
# ============================================================================

def evaluate_exact_match(
    answer: Union[str, Dict[str, Any]],
    gold_standard: Union[str, Dict[str, Any]],
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Evaluate exact match between answer and gold standard.
    
    Args:
        answer: The answer to evaluate
        gold_standard: The gold standard answer
        case_sensitive: Whether comparison should be case-sensitive
    
    Returns:
        Dictionary with exact_match (bool) and normalized texts
    """
    try:
        answer_text = _extract_text(answer)
        gold_text = _extract_text(gold_standard)
        
        if not case_sensitive:
            answer_text = answer_text.lower().strip()
            gold_text = gold_text.lower().strip()
        else:
            answer_text = answer_text.strip()
            gold_text = gold_text.strip()
        
        exact_match = answer_text == gold_text
        
        return {
            "exact_match": exact_match,
            "answer_text": answer_text,
            "gold_text": gold_text,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in evaluate_exact_match: {str(e)}")
        return {
            "exact_match": False,
            "status": "error",
            "error": str(e)
        }


def evaluate_contains_keywords(
    answer: Union[str, Dict[str, Any]],
    required_keywords: List[str],
    case_sensitive: bool = False
) -> Dict[str, Any]:
    """
    Check if answer contains required keywords.
    
    Args:
        answer: The answer to evaluate
        required_keywords: List of keywords that must be present
        case_sensitive: Whether keyword matching should be case-sensitive
    
    Returns:
        Dictionary with found_keywords, missing_keywords, and coverage score
    """
    try:
        answer_text = _extract_text(answer)
        
        if not case_sensitive:
            answer_text = answer_text.lower()
            required_keywords = [kw.lower() for kw in required_keywords]
        
        found_keywords = []
        missing_keywords = []
        
        for keyword in required_keywords:
            if keyword in answer_text:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        coverage = len(found_keywords) / len(required_keywords) if required_keywords else 1.0
        
        return {
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "coverage_score": coverage,
            "all_found": len(missing_keywords) == 0,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in evaluate_contains_keywords: {str(e)}")
        return {
            "found_keywords": [],
            "missing_keywords": required_keywords,
            "coverage_score": 0.0,
            "all_found": False,
            "status": "error",
            "error": str(e)
        }


def evaluate_json_structure(
    answer: Union[str, Dict[str, Any]],
    expected_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that answer matches expected JSON schema structure.
    
    Args:
        answer: The answer to evaluate (can be JSON string or dict)
        expected_schema: Expected schema structure (dict with keys and optional nested structures)
    
    Returns:
        Dictionary with validation results
    """
    try:
        # Parse JSON if string
        if isinstance(answer, str):
            try:
                answer_dict = json.loads(answer)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                pattern = r'```(?:json)?\s*(.*?)\s*```'
                match = re.search(pattern, answer, re.DOTALL)
                if match:
                    answer_dict = json.loads(match.group(1).strip())
                else:
                    raise ValueError("Could not parse JSON from answer")
        else:
            answer_dict = answer
        
        missing_keys = []
        extra_keys = []
        type_mismatches = []
        valid_keys = []
        
        def validate_recursive(data: Dict, schema: Dict, path: str = ""):
            """Recursively validate schema structure."""
            for key, expected_value in schema.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in data:
                    missing_keys.append(current_path)
                    continue
                
                valid_keys.append(current_path)
                
                # If schema value is a dict, recursively validate
                if isinstance(expected_value, dict) and isinstance(data[key], dict):
                    validate_recursive(data[key], expected_value, current_path)
                # If schema value is a type hint (like "string", "number"), check type
                elif isinstance(expected_value, str) and expected_value in ["string", "number", "boolean", "array", "object"]:
                    actual_type = type(data[key]).__name__
                    if expected_value == "string" and actual_type != "str":
                        type_mismatches.append(f"{current_path}: expected string, got {actual_type}")
                    elif expected_value == "number" and actual_type not in ["int", "float"]:
                        type_mismatches.append(f"{current_path}: expected number, got {actual_type}")
                    elif expected_value == "boolean" and actual_type != "bool":
                        type_mismatches.append(f"{current_path}: expected boolean, got {actual_type}")
                    elif expected_value == "array" and actual_type != "list":
                        type_mismatches.append(f"{current_path}: expected array, got {actual_type}")
                    elif expected_value == "object" and actual_type != "dict":
                        type_mismatches.append(f"{current_path}: expected object, got {actual_type}")
        
        validate_recursive(answer_dict, expected_schema)
        
        # Check for extra keys (optional - can be disabled)
        all_schema_keys = _get_all_keys(expected_schema)
        all_answer_keys = _get_all_keys(answer_dict)
        extra_keys = [key for key in all_answer_keys if key not in all_schema_keys]
        
        is_valid = len(missing_keys) == 0 and len(type_mismatches) == 0
        
        return {
            "is_valid": is_valid,
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
            "type_mismatches": type_mismatches,
            "valid_keys": valid_keys,
            "validity_score": len(valid_keys) / len(all_schema_keys) if all_schema_keys else 1.0,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in evaluate_json_structure: {str(e)}")
        return {
            "is_valid": False,
            "missing_keys": [],
            "extra_keys": [],
            "type_mismatches": [],
            "valid_keys": [],
            "validity_score": 0.0,
            "status": "error",
            "error": str(e)
        }


def evaluate_response_time(
    response_time_seconds: float,
    max_acceptable_time: float = 30.0
) -> Dict[str, Any]:
    """
    Evaluate response time performance.
    
    Args:
        response_time_seconds: Time taken for response in seconds
        max_acceptable_time: Maximum acceptable time in seconds
    
    Returns:
        Dictionary with performance metrics
    """
    try:
        is_acceptable = response_time_seconds <= max_acceptable_time
        performance_score = max(0.0, 1.0 - (response_time_seconds / max_acceptable_time))
        
        return {
            "response_time_seconds": response_time_seconds,
            "max_acceptable_time": max_acceptable_time,
            "is_acceptable": is_acceptable,
            "performance_score": performance_score,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in evaluate_response_time: {str(e)}")
        return {
            "response_time_seconds": response_time_seconds,
            "is_acceptable": False,
            "performance_score": 0.0,
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Comprehensive Evaluation Suite
# ============================================================================

def evaluate_comprehensive(
    answer: Union[str, Dict[str, Any]],
    gold_standard: Union[str, Dict[str, Any]],
    expected_schema: Optional[Dict[str, Any]] = None,
    required_keywords: Optional[List[str]] = None,
    number_tolerance: float = 0.01,
    response_time: Optional[float] = None,
    max_response_time: float = 30.0
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation with multiple metrics.
    
    Args:
        answer: The answer to evaluate
        gold_standard: The gold standard answer
        expected_schema: Optional JSON schema to validate against
        required_keywords: Optional list of keywords that must be present
        number_tolerance: Tolerance for number comparison
        response_time: Optional response time in seconds
        max_response_time: Maximum acceptable response time
    
    Returns:
        Dictionary with all evaluation results and overall score
    """
    results = {
        "cosine_similarity": None,
        "numbers_evaluation": None,
        "exact_match": None,
        "keyword_evaluation": None,
        "schema_validation": None,
        "response_time_evaluation": None,
        "overall_score": 0.0,
        "status": "success"
    }
    
    try:
        # Cosine similarity
        results["cosine_similarity"] = evaluate_cosine_similarity(answer, gold_standard)
        
        # Numbers evaluation
        results["numbers_evaluation"] = evaluate_numbers(
            answer, gold_standard, tolerance=number_tolerance
        )
        
        # Exact match
        results["exact_match"] = evaluate_exact_match(answer, gold_standard)
        
        # Keyword evaluation
        if required_keywords:
            results["keyword_evaluation"] = evaluate_contains_keywords(
                answer, required_keywords
            )
        
        # Schema validation
        if expected_schema:
            results["schema_validation"] = evaluate_json_structure(
                answer, expected_schema
            )
        
        # Response time
        if response_time is not None:
            results["response_time_evaluation"] = evaluate_response_time(
                response_time, max_response_time
            )
        
        # Calculate overall score (weighted average)
        scores = []
        weights = []
        
        if results["cosine_similarity"] and results["cosine_similarity"]["status"] == "success":
            scores.append(results["cosine_similarity"]["similarity_score"])
            weights.append(0.3)
        
        if results["numbers_evaluation"] and results["numbers_evaluation"]["status"] == "success":
            scores.append(results["numbers_evaluation"]["match_score"])
            weights.append(0.3)
        
        if results["exact_match"] and results["exact_match"]["status"] == "success":
            scores.append(1.0 if results["exact_match"]["exact_match"] else 0.0)
            weights.append(0.1)
        
        if results.get("keyword_evaluation") and results["keyword_evaluation"]["status"] == "success":
            scores.append(results["keyword_evaluation"]["coverage_score"])
            weights.append(0.1)
        
        if results.get("schema_validation") and results["schema_validation"]["status"] == "success":
            scores.append(results["schema_validation"]["validity_score"])
            weights.append(0.1)
        
        if results.get("response_time_evaluation") and results["response_time_evaluation"]["status"] == "success":
            scores.append(results["response_time_evaluation"]["performance_score"])
            weights.append(0.1)
        
        if weights:
            total_weight = sum(weights)
            if total_weight > 0:
                results["overall_score"] = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                results["overall_score"] = 0.0
        else:
            results["overall_score"] = 0.0
        
        return results
        
    except Exception as e:
        logger.error(f"Error in evaluate_comprehensive: {str(e)}", exc_info=True)
        results["status"] = "error"
        results["error"] = str(e)
        return results


# ============================================================================
# Multi-Agent Evaluation
# ============================================================================

def evaluate_multi_agent(
    agent_results: List[Dict[str, Any]],
    gold_standards: Union[Dict[str, Any], List[Dict[str, Any]]],
    evaluation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate results from multiple agents.
    
    Args:
        agent_results: List of results from different agents
                      Each result should have at least: {"agent_name": str, "response": ...}
        gold_standards: Either a single gold standard dict (applied to all) or
                       a list/dict mapping agent names to gold standards
        evaluation_config: Optional configuration dict with:
                          - expected_schemas: Dict mapping agent names to schemas
                          - required_keywords: Dict mapping agent names to keyword lists
                          - number_tolerance: float
                          - max_response_time: float
    
    Returns:
        Dictionary with per-agent evaluations and aggregate metrics
    """
    try:
        if evaluation_config is None:
            evaluation_config = {}
        
        # Normalize gold_standards
        if isinstance(gold_standards, dict) and "agent_name" not in gold_standards:
            # Assume it's a mapping of agent_name -> gold_standard
            gold_standards_dict = gold_standards
        elif isinstance(gold_standards, list):
            # Assume it's a list matching agent_results order
            gold_standards_dict = {
                result.get("agent_name", f"agent_{i}"): gold_std
                for i, (result, gold_std) in enumerate(zip(agent_results, gold_standards))
            }
        else:
            # Single gold standard for all agents
            gold_standards_dict = {
                result.get("agent_name", f"agent_{i}"): gold_standards
                for i, result in enumerate(agent_results)
            }
        
        agent_evaluations = {}
        all_scores = []
        
        for result in agent_results:
            agent_name = result.get("agent_name", "unknown")
            answer = result.get("response", result)  # Use 'response' key or entire result
            
            gold_standard = gold_standards_dict.get(agent_name, gold_standards_dict.get(list(gold_standards_dict.keys())[0]))
            
            # Get agent-specific config
            expected_schema = None
            if "expected_schemas" in evaluation_config:
                expected_schema = evaluation_config["expected_schemas"].get(agent_name)
            
            required_keywords = None
            if "required_keywords" in evaluation_config:
                required_keywords = evaluation_config["required_keywords"].get(agent_name)
            
            number_tolerance = evaluation_config.get("number_tolerance", 0.01)
            response_time = result.get("response_time")
            max_response_time = evaluation_config.get("max_response_time", 30.0)
            
            # Run comprehensive evaluation
            evaluation = evaluate_comprehensive(
                answer=answer,
                gold_standard=gold_standard,
                expected_schema=expected_schema,
                required_keywords=required_keywords,
                number_tolerance=number_tolerance,
                response_time=response_time,
                max_response_time=max_response_time
            )
            
            agent_evaluations[agent_name] = evaluation
            all_scores.append(evaluation.get("overall_score", 0.0))
        
        # Calculate aggregate metrics
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        best_agent = max(agent_evaluations.items(), key=lambda x: x[1].get("overall_score", 0.0))[0] if agent_evaluations else None
        
        return {
            "agent_evaluations": agent_evaluations,
            "average_score": avg_score,
            "best_agent": best_agent,
            "total_agents": len(agent_results),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_multi_agent: {str(e)}", exc_info=True)
        return {
            "agent_evaluations": {},
            "average_score": 0.0,
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Batch Evaluation
# ============================================================================

def evaluate_batch(
    test_cases: List[Dict[str, Any]],
    evaluation_function: callable = evaluate_comprehensive,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluate a batch of test cases.
    
    Args:
        test_cases: List of test case dicts, each with:
                   - "answer": the answer to evaluate
                   - "gold_standard": the gold standard
                   - Optional: "expected_schema", "required_keywords", etc.
        evaluation_function: Function to use for evaluation (default: evaluate_comprehensive)
        **kwargs: Additional arguments to pass to evaluation_function
    
    Returns:
        Dictionary with per-test results and aggregate statistics
    """
    try:
        results = []
        all_scores = []
        
        for i, test_case in enumerate(test_cases):
            test_kwargs = {**kwargs, **test_case}
            answer = test_kwargs.pop("answer")
            gold_standard = test_kwargs.pop("gold_standard")
            
            evaluation = evaluation_function(answer, gold_standard, **test_kwargs)
            
            result = {
                "test_case_id": i + 1,
                "evaluation": evaluation,
                "overall_score": evaluation.get("overall_score", 0.0)
            }
            results.append(result)
            all_scores.append(result["overall_score"])
        
        # Calculate statistics
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        min_score = min(all_scores) if all_scores else 0.0
        max_score = max(all_scores) if all_scores else 0.0
        
        # Calculate standard deviation
        if len(all_scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in all_scores) / len(all_scores)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0
        
        return {
            "test_results": results,
            "statistics": {
                "total_tests": len(test_cases),
                "average_score": avg_score,
                "min_score": min_score,
                "max_score": max_score,
                "std_deviation": std_dev
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error in evaluate_batch: {str(e)}", exc_info=True)
        return {
            "test_results": [],
            "statistics": {},
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_text(input_data: Union[str, Dict[str, Any]]) -> str:
    """Extract text from string or dict input."""
    if isinstance(input_data, str):
        return input_data
    elif isinstance(input_data, dict):
        # Try common keys
        for key in ["response", "answer", "text", "content", "output"]:
            if key in input_data:
                return str(input_data[key])
        # If it's a structured response, try to extract from "extracted" field
        if "extracted" in input_data:
            extracted = input_data["extracted"]
            if isinstance(extracted, dict):
                # Combine all values
                texts = []
                for key, value in extracted.items():
                    if isinstance(value, dict) and "value" in value:
                        texts.append(str(value["value"]))
                    else:
                        texts.append(str(value))
                return " ".join(texts)
        # Fallback: convert entire dict to JSON string
        return json.dumps(input_data, indent=2)
    else:
        return str(input_data)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (optional - can be customized)
    # text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def _extract_numbers(input_data: Union[str, Dict[str, Any]]) -> List[float]:
    """Extract all numbers from input."""
    text = _extract_text(input_data)
    
    # Pattern to match numbers (integers, decimals, scientific notation)
    # Also handles negative numbers and numbers with commas
    # Strategy: Match numbers with commas first (longest match), then simple numbers
    # This prevents partial matches like "250" and "00" from "25000"
    
    numbers = []
    
    # Pattern 1: Numbers with commas (e.g., 25,000 or 1,234,567.89)
    # This must come first to match longer numbers before shorter ones
    comma_pattern = r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?'
    for match in re.finditer(comma_pattern, text):
        try:
            num_str = match.group(0).replace(',', '')
            num = float(num_str)
            numbers.append((match.start(), match.end(), num))  # Store position to avoid overlaps
        except ValueError:
            continue
    
    # Pattern 2: Simple decimals (e.g., 123.45 or .5)
    decimal_pattern = r'-?\d+\.\d+|\.\d+'
    for match in re.finditer(decimal_pattern, text):
        # Check if this match overlaps with any comma-number match
        start, end = match.span()
        overlaps = any(s <= start < e or s < end <= e for s, e, _ in numbers)
        if not overlaps:
            try:
                num = float(match.group(0))
                numbers.append((start, end, num))
            except ValueError:
                continue
    
    # Pattern 3: Simple integers (e.g., 123 or -456)
    # Only match if not part of a comma-number or decimal
    integer_pattern = r'-?\d+'
    for match in re.finditer(integer_pattern, text):
        start, end = match.span()
        overlaps = any(s <= start < e or s < end <= e for s, e, _ in numbers)
        if not overlaps:
            try:
                num = float(match.group(0))
                numbers.append((start, end, num))
            except ValueError:
                continue
    
    # Extract just the numbers, remove duplicates while preserving order
    unique_numbers = []
    seen = set()
    for _, _, num in sorted(numbers, key=lambda x: x[0]):  # Sort by position
        # Round to avoid floating point precision issues
        rounded = round(num, 6)
        if rounded not in seen:
            seen.add(rounded)
            unique_numbers.append(num)
    
    return unique_numbers


def _simple_cosine_similarity(text1: str, text2: str) -> float:
    """Calculate simple cosine similarity using word counts."""
    # Tokenize
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Get all unique words
    all_words = words1.union(words2)
    
    if not all_words:
        return 1.0 if text1 == text2 else 0.0
    
    # Create vectors
    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]
    
    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(a * a for a in vector2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def _get_all_keys(d: Dict, prefix: str = "") -> List[str]:
    """Get all keys from nested dictionary."""
    keys = []
    for key, value in d.items():
        current_key = f"{prefix}.{key}" if prefix else key
        keys.append(current_key)
        if isinstance(value, dict):
            keys.extend(_get_all_keys(value, current_key))
    return keys


# ============================================================================
# Export Report Functions
# ============================================================================

def export_evaluation_report(
    evaluation_results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "json"
) -> None:
    """
    Export evaluation results to a file.
    
    Args:
        evaluation_results: Results from any evaluation function
        output_path: Path to save the report
        format: "json" or "txt"
    """
    try:
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("EVALUATION REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(json.dumps(evaluation_results, indent=2, default=str))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Evaluation report exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting report: {str(e)}")
        raise


# ============================================================================
# Example Usage and Tests
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("TEVV Test Suite - Example Usage")
    print("=" * 80)
    
    # Example 1: Cosine similarity
    print("\n1. Cosine Similarity Evaluation:")
    answer1 = "The solar panel system requires 500 square feet of land area."
    gold1 = "The solar installation needs 500 sq ft of land."
    result1 = evaluate_cosine_similarity(answer1, gold1)
    print(f"   Similarity Score: {result1['similarity_score']:.3f}")
    print(f"   Method: {result1['method']}")
    
    # Example 2: Numbers evaluation
    print("\n2. Numbers Evaluation:")
    answer2 = "The system requires 500 square feet and costs $25,000."
    gold2 = "Land requirement: 500 sq ft. Total cost: $25000."
    result2 = evaluate_numbers(answer2, gold2)
    print(f"   Numbers Match: {result2['numbers_match']}")
    print(f"   Match Score: {result2['match_score']:.3f}")
    print(f"   Answer Numbers: {result2['answer_numbers']}")
    print(f"   Gold Numbers: {result2['gold_numbers']}")
    
    # Example 3: Comprehensive evaluation
    print("\n3. Comprehensive Evaluation:")
    result3 = evaluate_comprehensive(
        answer=answer2,
        gold_standard=gold2,
        required_keywords=["solar", "cost", "land"]
    )
    print(f"   Overall Score: {result3['overall_score']:.3f}")
    print(f"   Cosine Similarity: {result3['cosine_similarity']['similarity_score']:.3f}")
    print(f"   Numbers Match Score: {result3['numbers_evaluation']['match_score']:.3f}")
    
    print("\n" + "=" * 80)
    print("TEVV Test Suite is ready to use!")

