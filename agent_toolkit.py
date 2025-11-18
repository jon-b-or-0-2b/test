"""
Agent Toolkit - Comprehensive document processing and LLM evaluation toolkit.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Document processing
from PyPDF2 import PdfReader
import docx
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# LLM clients
from openai import OpenAI
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from secrete import OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_toolkit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log availability of optional dependencies
if not PDF2IMAGE_AVAILABLE:
    logger.warning("pdf2image not available, will use PyMuPDF as fallback")
if not PYMUPDF_AVAILABLE:
    logger.warning("PyMuPDF not available - PDF to image conversion may fail without poppler")

# API Keys (add to secrete.py if needed)
ANTHROPIC_API_KEY = getattr(__import__('secrete', fromlist=['ANTHROPIC_API_KEY']), 'ANTHROPIC_API_KEY', None)
GOOGLE_API_KEY = getattr(__import__('secrete', fromlist=['GOOGLE_API_KEY']), 'GOOGLE_API_KEY', None)


# ============================================================================
# Document Loader
# ============================================================================

def document_loader(file_path: Union[str, Path], is_image: bool = False) -> Union[List[tuple], List[str]]:
    """
    Load PDF/DOCX/TXT documents.
    
    Args:
        file_path: Path to the document
        is_image: If True, convert pages to images and save locally
        
    Returns:
        If is_image: List of image paths
        If not is_image: List of (page_number, text) tuples
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading document: {file_path}, is_image={is_image}")
        
        suffix = file_path.suffix.lower()
        
        if is_image:
            if suffix == '.pdf':
                return _pdf_to_images(file_path)
            else:
                logger.warning(f"Image conversion only supported for PDF files. Got: {suffix}")
                return []
        else:
            if suffix == '.pdf':
                return _load_pdf_text(file_path)
            elif suffix == '.docx':
                return _load_docx_text(file_path)
            elif suffix == '.txt':
                return _load_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
                
    except Exception as e:
        logger.error(f"Error in document_loader: {str(e)}", exc_info=True)
        print(f"ERROR: document_loader failed: {str(e)}")
        raise


def _pdf_to_images(pdf_path: Path) -> List[str]:
    """Convert PDF pages to images and save them locally."""
    try:
        base_name = pdf_path.stem
        output_dir = Path(base_name)
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Try pdf2image first, fallback to PyMuPDF
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_path(str(pdf_path))
                image_paths = []
                for idx, image in enumerate(images, start=1):
                    image_path = output_dir / f"{base_name}_page_{idx}.png"
                    image.save(image_path, 'PNG')
                    image_paths.append(str(image_path))
                    logger.info(f"Saved image: {image_path}")
                
                logger.info(f"SUCCESS: Converted {len(images)} pages to images using pdf2image")
                return image_paths
            except Exception as e:
                logger.warning(f"pdf2image failed: {str(e)}, trying PyMuPDF fallback")
                if not PYMUPDF_AVAILABLE:
                    raise
        
        # Fallback to PyMuPDF
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(str(pdf_path))
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image (300 DPI for good quality)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                image_path = output_dir / f"{base_name}_page_{page_num + 1}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
                logger.info(f"Saved image: {image_path}")
            
            doc.close()
            logger.info(f"SUCCESS: Converted {len(image_paths)} pages to images using PyMuPDF")
            return image_paths
        else:
            error_msg = (
                "PDF to image conversion requires either:\n"
                "1. pdf2image with poppler installed (see: https://github.com/Belval/pdf2image)\n"
                "2. PyMuPDF installed (pip install PyMuPDF)\n\n"
                "For Windows poppler installation:\n"
                "  - Download from: https://github.com/oschwartz10612/poppler-windows/releases\n"
                "  - Extract and add bin folder to PATH"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}", exc_info=True)
        print(f"ERROR: PDF to image conversion failed: {str(e)}")
        if "poppler" in str(e).lower() or "PDFInfoNotInstalled" in str(e):
            print("\n" + "="*60)
            print("SOLUTION: Install PyMuPDF for PDF to image conversion:")
            print("  pip install PyMuPDF")
            print("="*60 + "\n")
        raise


def _load_pdf_text(pdf_path: Path) -> List[tuple]:
    """Extract text from PDF pages."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((idx, text))
        logger.info(f"SUCCESS: Extracted text from {len(pages)} PDF pages")
        return pages
    except Exception as e:
        logger.error(f"Error loading PDF text: {str(e)}", exc_info=True)
        print(f"ERROR: PDF text extraction failed: {str(e)}")
        raise


def _load_docx_text(docx_path: Path) -> List[tuple]:
    """Extract text from DOCX file."""
    try:
        doc = docx.Document(str(docx_path))
        full_text = "\n".join([para.text for para in doc.paragraphs])
        logger.info(f"SUCCESS: Extracted text from DOCX file")
        return [(1, full_text)]
    except Exception as e:
        logger.error(f"Error loading DOCX text: {str(e)}", exc_info=True)
        print(f"ERROR: DOCX text extraction failed: {str(e)}")
        raise


def _load_txt_text(txt_path: Path) -> List[tuple]:
    """Extract text from TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"SUCCESS: Extracted text from TXT file")
        return [(1, text)]
    except Exception as e:
        logger.error(f"Error loading TXT text: {str(e)}", exc_info=True)
        print(f"ERROR: TXT text extraction failed: {str(e)}")
        raise


# ============================================================================
# LLM Cleaner
# ============================================================================

def llm_cleaner(response_text: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and validate LLM response against JSON schema.
    
    Args:
        response_text: Raw LLM response
        json_schema: Expected JSON schema structure
        
    Returns:
        Cleaned and validated JSON data
    """
    try:
        logger.info("Cleaning LLM response")
        
        # Step 1: Extract JSON from markdown code blocks
        pattern = r'```(?:json)?\s*(.*?)\s*```'
        match = re.search(pattern, str(response_text), re.DOTALL)
        json_str = match.group(1).strip() if match else response_text.strip()
        
        # Remove leading/trailing non-JSON characters
        json_str = re.sub(r'^[^{]*', '', json_str)
        json_str = re.sub(r'[^}]*$', '', json_str)
        
        # Step 2: Load JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error, attempting to fix: {str(e)}")
            # Try to find JSON object boundaries
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = json_str[start:end]
                data = json.loads(json_str)
            else:
                raise
        
        # Step 3: Verify schema keys
        _validate_schema(data, json_schema)
        
        logger.info("SUCCESS: LLM response cleaned and validated")
        return data
        
    except Exception as e:
        logger.error(f"Error in llm_cleaner: {str(e)}", exc_info=True)
        print(f"ERROR: llm_cleaner failed: {str(e)}")
        return {"error": str(e), "raw_response": response_text}


def _validate_schema(data: Dict, schema: Dict, path: str = "") -> None:
    """Recursively validate that all schema keys exist in data."""
    for key, expected_value in schema.items():
        current_path = f"{path}.{key}" if path else key
        if key not in data:
            logger.warning(f"Missing key in response: {current_path}")
            continue
        
        if isinstance(expected_value, dict) and isinstance(data[key], dict):
            _validate_schema(data[key], expected_value, current_path)


# ============================================================================
# LLM Wrappers
# ============================================================================

def llm_wrapper_text(model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call text-based LLM models.
    
    Supported models:
    - gpt5, gpt5-mini, gpt-4o
    - claude-sonnet-4.1, claude-sonnet-4.5
    - gemini-flash, gemini-flash-lite-2.5
    """
    try:
        logger.info(f"Calling LLM (text): {model}")
        
        if model.startswith('gpt'):
            return _openai_text_call(model, prompt, system_prompt)
        elif model.startswith('claude'):
            return _anthropic_text_call(model, prompt, system_prompt)
        elif model.startswith('gemini'):
            return _gemini_text_call(model, prompt, system_prompt)
        else:
            raise ValueError(f"Unknown model: {model}")
            
    except Exception as e:
        logger.error(f"Error in llm_wrapper_text for {model}: {str(e)}", exc_info=True)
        print(f"ERROR: llm_wrapper_text failed for {model}: {str(e)}")
        raise


def llm_wrapper_image(model: str, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call image-based LLM models.
    
    Supported models: same as llm_wrapper_text
    """
    try:
        logger.info(f"Calling LLM (image): {model}, image: {image_path}")
        
        if model.startswith('gpt'):
            return _openai_image_call(model, image_path, prompt, system_prompt)
        elif model.startswith('claude'):
            return _anthropic_image_call(model, image_path, prompt, system_prompt)
        elif model.startswith('gemini'):
            return _gemini_image_call(model, image_path, prompt, system_prompt)
        else:
            raise ValueError(f"Unknown model: {model}")
            
    except Exception as e:
        logger.error(f"Error in llm_wrapper_image for {model}: {str(e)}", exc_info=True)
        print(f"ERROR: llm_wrapper_image failed for {model}: {str(e)}")
        raise


def _openai_text_call(model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """OpenAI text API call."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Map model names (update these when actual model names are available)
        # Note: GPT-5 models may not exist yet - adjust model names as needed
        model_map = {
            "gpt5": "gpt-4o",  # Placeholder - update when GPT-5 is available
            "gpt5-mini": "gpt-4o-mini",  # Placeholder - update when GPT-5-mini is available
            "gpt-4o": "gpt-4o"
        }
        api_model = model_map.get(model, model)
        
        response = client.chat.completions.create(
            model=api_model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI text call error: {str(e)}")
        raise


def _openai_image_call(model: str, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """OpenAI image API call."""
    try:
        import base64
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        })
        
        # Map model names (update these when actual model names are available)
        model_map = {
            "gpt5": "gpt-4o",  # Placeholder - update when GPT-5 is available
            "gpt5-mini": "gpt-4o-mini",  # Placeholder - update when GPT-5-mini is available
            "gpt-4o": "gpt-4o"
        }
        api_model = model_map.get(model, model)
        
        response = client.chat.completions.create(
            model=api_model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI image call error: {str(e)}")
        raise


def _anthropic_text_call(model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Anthropic text API call."""
    try:
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        model_map = {
            "claude-sonnet-4.1": "claude-sonnet-4-20250514",
            "claude-sonnet-4.5": "claude-sonnet-4-20250514"
        }
        api_model = model_map.get(model, model)
        
        messages = [{"role": "user", "content": prompt}]
        system = system_prompt if system_prompt else None
        
        response = client.messages.create(
            model=api_model,
            max_tokens=4096,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Anthropic text call error: {str(e)}")
        raise


def _anthropic_image_call(model: str, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Anthropic image API call."""
    try:
        if anthropic is None:
            raise ImportError("anthropic package not installed")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        import base64
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        model_map = {
            "claude-sonnet-4.1": "claude-sonnet-4-20250514",
            "claude-sonnet-4.5": "claude-sonnet-4-20250514"
        }
        api_model = model_map.get(model, model)
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }]
        system = system_prompt if system_prompt else None
        
        response = client.messages.create(
            model=api_model,
            max_tokens=4096,
            system=system,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Anthropic image call error: {str(e)}")
        raise


def _gemini_text_call(model: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Google Gemini text API call."""
    try:
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        model_map = {
            "gemini-flash": "gemini-1.5-flash",
            "gemini-flash-lite-2.5": "gemini-1.5-flash"
        }
        api_model = model_map.get(model, model)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        model_instance = genai.GenerativeModel(api_model)
        response = model_instance.generate_content(
            full_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini text call error: {str(e)}")
        raise


def _gemini_image_call(model: str, image_path: str, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Google Gemini image API call."""
    try:
        if genai is None:
            raise ImportError("google-generativeai package not installed")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set")
        
        genai.configure(api_key=GOOGLE_API_KEY)
        
        model_map = {
            "gemini-flash": "gemini-1.5-flash",
            "gemini-flash-lite-2.5": "gemini-1.5-flash"
        }
        api_model = model_map.get(model, model)
        
        import PIL.Image
        img = PIL.Image.open(image_path)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        model_instance = genai.GenerativeModel(api_model)
        response = model_instance.generate_content(
            [full_prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini image call error: {str(e)}")
        raise


# ============================================================================
# LLM as Judge
# ============================================================================

def llm_as_judge(
    original_prompt: str,
    json_response: Dict[str, Any],
    judge_models: List[str],
    json_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use multiple LLMs to judge a model's response.
    
    Args:
        original_prompt: The prompt given to the model being evaluated
        json_response: The JSON response from the model
        judge_models: List of model names to use as judges
        json_schema: Expected schema structure
        
    Returns:
        Dictionary with scores from each judge, overall score, and found status
    """
    try:
        logger.info(f"Starting LLM judgment with {len(judge_models)} judges")
        
        judge_prompt = _build_judge_prompt(original_prompt, json_response, json_schema)
        
        scores = {}
        found_flags = []
        
        for judge_model in judge_models:
            try:
                logger.info(f"Judge {judge_model} evaluating response")
                response = llm_wrapper_text(judge_model, judge_prompt)
                judge_result = json.loads(response)
                
                score = float(judge_result.get("score", 0.0))
                found = bool(judge_result.get("found", False))
                
                scores[judge_model] = score
                found_flags.append(found)
                
                logger.info(f"Judge {judge_model}: score={score}, found={found}")
                
            except Exception as e:
                logger.error(f"Judge {judge_model} failed: {str(e)}")
                scores[judge_model] = 0.0
                found_flags.append(False)
        
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        found_from_judges = any(found_flags)
        
        # Programmatically verify that ALL extracted items have valid values
        found_programmatic = _verify_all_extracted_items(json_response, json_schema)
        
        # Both judge and programmatic check must pass
        found = found_from_judges and found_programmatic
        
        result = {
            **{f"judge_{model}": score for model, score in scores.items()},
            "overall_score": overall_score,
            "found": found,
            "found_judge": found_from_judges,
            "found_programmatic": found_programmatic
        }
        
        logger.info(f"SUCCESS: Judgment complete. Overall score: {overall_score}, Found (judge): {found_from_judges}, Found (programmatic): {found_programmatic}, Found (final): {found}")
        return result
        
    except Exception as e:
        logger.error(f"Error in llm_as_judge: {str(e)}", exc_info=True)
        print(f"ERROR: llm_as_judge failed: {str(e)}")
        return {
            "overall_score": 0.0,
            "found": False,
            "error": str(e)
        }


def _build_judge_prompt(original_prompt: str, json_response: Dict, json_schema: Dict) -> str:
    """Build the prompt for the judge LLM."""
    schema_keys = _extract_schema_keys(json_schema)
    extracted_keys = _extract_extracted_keys(json_schema)
    
    prompt = f"""You are an expert evaluator. Your job is to judge a model's response against the prompt it was given.

Model was instructed as follows:
{original_prompt}

Model responded as follows:
{json.dumps(json_response, indent=2)}

Expected schema structure:
{json.dumps(json_schema, indent=2)}

CRITICAL REQUIREMENTS FOR "found" = true:
1. ALL required extracted items must be present in the "extracted" object
2. ALL extracted items must have VALID values (not "not found", not empty, not "nan", not "none")
3. Each extracted item must have both "value" and "source" fields with valid content

Expected extracted items (ALL must have valid values):
{json.dumps(extracted_keys, indent=2)}

Evaluate:
1. How well does the response answer the prompt? (0.0 to 1.0)
2. Are ALL required extracted items present with VALID values (not "not found")? (True/False)
3. Is the information accurate and complete?

IMPORTANT: Set "found" = true ONLY if ALL extracted items have valid, non-empty values. If even ONE item is missing or has "not found", set "found" = false.

Respond with JSON format:
{{
  "score": 0.0-1.0,
  "found": true/false,
  "reasoning": "brief explanation"
}}
"""
    return prompt


def _extract_schema_keys(schema: Dict, path: str = "") -> List[str]:
    """Extract all keys from schema recursively."""
    keys = []
    for key, value in schema.items():
        current_path = f"{path}.{key}" if path else key
        keys.append(current_path)
        if isinstance(value, dict):
            keys.extend(_extract_schema_keys(value, current_path))
    return keys


def _extract_extracted_keys(schema: Dict) -> List[str]:
    """Extract keys from the 'extracted' section of the schema."""
    extracted = schema.get("extracted", {})
    if isinstance(extracted, dict):
        return list(extracted.keys())
    return []


def _verify_all_extracted_items(json_response: Dict, json_schema: Dict) -> bool:
    """
    Programmatically verify that ALL extracted items have valid values.
    Returns True only if ALL items are present and have valid (non-empty, not "not found") values.
    """
    try:
        # Get expected extracted keys from schema
        expected_keys = _extract_extracted_keys(json_schema)
        if not expected_keys:
            logger.warning("No extracted keys found in schema")
            return False
        
        # Get actual extracted items from response
        extracted = json_response.get("extracted", {})
        if not isinstance(extracted, dict):
            logger.warning("Extracted is not a dictionary")
            return False
        
        # Check that all expected keys are present
        missing_keys = []
        invalid_values = []
        
        for key in expected_keys:
            if key not in extracted:
                missing_keys.append(key)
                continue
            
            item = extracted[key]
            if not isinstance(item, dict):
                invalid_values.append(f"{key}: not a dict")
                continue
            
            value = item.get("value", "")
            value_str = str(value).strip().lower()
            
            # Check if value is valid (not empty, not "not found", not "nan", not "none")
            if not value or value_str in ["not found", "nan", "none", ""]:
                invalid_values.append(f"{key}: invalid value '{value}'")
        
        if missing_keys:
            logger.warning(f"Missing extracted keys: {missing_keys}")
            return False
        
        if invalid_values:
            logger.warning(f"Invalid extracted values: {invalid_values}")
            return False
        
        logger.info(f"SUCCESS: All {len(expected_keys)} extracted items have valid values")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying extracted items: {str(e)}")
        return False


# ============================================================================
# Parallel Processing
# ============================================================================

def parallel_process(
    func,
    items: List[Any],
    max_workers: int = 10,
    **kwargs
) -> List[Any]:
    """
    Parallel processing wrapper using ThreadPoolExecutor.
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Number of worker threads
        **kwargs: Additional arguments to pass to func
        
    Returns:
        List of results in the same order as items
    """
    try:
        logger.info(f"Starting parallel processing: {len(items)} items, {max_workers} workers")
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(func, item, **kwargs): idx
                for idx, item in enumerate(items)
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    logger.info(f"SUCCESS: Completed item {idx+1}/{len(items)}")
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {str(e)}")
                    print(f"ERROR: Parallel processing failed for item {idx}: {str(e)}")
                    results[idx] = {"error": str(e)}
        
        logger.info("SUCCESS: Parallel processing complete")
        return results
        
    except Exception as e:
        logger.error(f"Error in parallel_process: {str(e)}", exc_info=True)
        print(f"ERROR: parallel_process failed: {str(e)}")
        raise


# ============================================================================
# Schema to DataFrame
# ============================================================================

def extract_to_dataframe(results: List[Dict[str, Any]], json_schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert extraction results to DataFrame with columns: domain, value_type, value, source, confidence.
    Each extracted item becomes a row.
    """
    try:
        logger.info("Converting extraction results to DataFrame")
        
        rows = []
        for result in results:
            domain = result.get("domain", "")
            confidence = result.get("confidence", "")
            confidence_keyword = result.get("confidence_keyword", "").lower()
            
            # Convert confidence to string if it's a number
            if isinstance(confidence, (int, float)):
                confidence = str(confidence)
            
            # Set confidence to "1" if "clearly stated" or similar high confidence indicators
            if any(keyword in confidence_keyword for keyword in ['clearly stated', 'explicitly mentioned', 'directly specified', 'clearly indicates']):
                confidence = "1"
            # Also check if confidence is already 1 or "1"
            elif str(confidence).strip() in ["1", "1.0"]:
                confidence = "1"
            
            extracted = result.get("extracted", {})
            page = result.get("page", "")
            
            # Create a row for each extracted item
            for key, item_data in extracted.items():
                if isinstance(item_data, dict):
                    value = item_data.get("value", "")
                    source = item_data.get("source", "")
                    
                    # Use PDF page number if source contains page reference
                    if source and "page" in str(source).lower():
                        # Extract page number from source, but use the actual PDF page number
                        source = f"page {page}"
                    elif not source:
                        source = f"page {page}"
                    
                    rows.append({
                        "domain": domain,
                        "value_type": key,
                        "value": value,
                        "source": source,
                        "confidence": confidence
                    })
        
        df = pd.DataFrame(rows)
        logger.info(f"SUCCESS: Created DataFrame with {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error in extract_to_dataframe: {str(e)}", exc_info=True)
        print(f"ERROR: extract_to_dataframe failed: {str(e)}")
        raise


def schema_to_dataframe(results: List[Dict[str, Any]], json_schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert results to DataFrame with all schema columns + judge outputs.
    
    Args:
        results: List of result dictionaries
        json_schema: Schema structure
        
    Returns:
        pandas DataFrame
    """
    try:
        logger.info("Converting results to DataFrame")
        
        # Flatten schema structure
        columns = _flatten_schema(json_schema)
        
        # Add judge columns
        judge_columns = ["overall_score", "found"]
        
        # Extract all unique judge columns from results
        for result in results:
            for key in result.keys():
                if key.startswith("judge_") and key not in judge_columns:
                    judge_columns.append(key)
        
        all_columns = columns + judge_columns
        
        # Build rows
        rows = []
        for result in results:
            row = {}
            for col in all_columns:
                row[col] = _get_nested_value(result, col)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"SUCCESS: Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error in schema_to_dataframe: {str(e)}", exc_info=True)
        print(f"ERROR: schema_to_dataframe failed: {str(e)}")
        raise


def _flatten_schema(schema: Dict, prefix: str = "") -> List[str]:
    """Flatten nested schema into column names."""
    columns = []
    for key, value in schema.items():
        current_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            columns.extend(_flatten_schema(value, current_key))
        else:
            columns.append(current_key)
    return columns


def _get_nested_value(data: Dict, key: str) -> Any:
    """Get value from nested dictionary using dot notation."""
    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, None)
        else:
            return None
    return value


# ============================================================================
# Agent Builder
# ============================================================================

def agent_builder(
    is_image_input: bool,
    model: str,
    data_input: Union[str, str],  # text or image path
    prompt: str,
    schema: Dict[str, Any],
    judge_models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main agent builder that orchestrates the extraction process.
    
    Args:
        is_image_input: Whether input is image or text
        model: Model name to use
        data_input: Text content or image path
        prompt: Extraction prompt
        schema: Expected JSON schema
        judge_models: Optional list of models to use as judges
        
    Returns:
        Complete result dictionary with extraction and judgment
    """
    try:
        logger.info(f"Building agent: model={model}, is_image={is_image_input}")
        
        # Call appropriate LLM wrapper
        if is_image_input:
            logger.info(f"Calling image LLM: {model}")
            raw_response = llm_wrapper_image(model, data_input, prompt)
        else:
            logger.info(f"Calling text LLM: {model}")
            raw_response = llm_wrapper_text(model, prompt)
        
        # Clean response
        logger.info("Cleaning LLM response")
        cleaned_response = llm_cleaner(raw_response, schema)
        
        if "error" in cleaned_response:
            logger.warning("LLM cleaner returned error, skipping judgment")
            return cleaned_response
        
        # Add judgment if requested
        if judge_models:
            logger.info("Running LLM judgment")
            judgment = llm_as_judge(prompt, cleaned_response, judge_models, schema)
            cleaned_response.update(judgment)
        
        logger.info("SUCCESS: Agent builder completed")
        return cleaned_response
        
    except Exception as e:
        logger.error(f"Error in agent_builder: {str(e)}", exc_info=True)
        print(f"ERROR: agent_builder failed: {str(e)}")
        return {
            "error": str(e),
            "model": model,
            "is_image": is_image_input
        }


# ============================================================================
# Round Robin Wrapper
# ============================================================================

def round_robin_wrapper(
    document_pages: Union[List[tuple], List[str]],
    agents: List[Dict[str, Any]],  # List of agent configs: {model, prompt, schema, judge_models}
    is_image: bool = False,
    parallel_process_flag: bool = False,
    num_workers: int = 10
) -> List[Dict[str, Any]]:
    """
    Round-robin document processing with multiple agents.
    
    Each agent processes pages until it finds all required information.
    Agents that find everything are removed, but processing continues for others.
    
    Args:
        document_pages: List of (page_num, text) tuples or image paths
        agents: List of agent configurations
        is_image: Whether pages are images
        parallel_process_flag: Whether to process agents in parallel
        num_workers: Number of parallel workers
        
    Returns:
        List of results from all agents
    """
    try:
        logger.info(f"Starting round-robin: {len(document_pages)} pages, {len(agents)} agents")
        
        active_agents = agents.copy()
        all_results = []
        
        for page_idx, page_data in enumerate(document_pages, start=1):
            if not active_agents:
                logger.info("All agents completed, stopping")
                break
            
            logger.info(f"Processing page {page_idx}/{len(document_pages)} with {len(active_agents)} active agents")
            
            # Prepare page data
            if is_image:
                data_input = page_data  # image path
            else:
                page_num, text = page_data
                data_input = text
            
            # Process agents
            if parallel_process_flag:
                def process_agent_with_idx(agent_data):
                    agent_idx, agent_config = agent_data
                    try:
                        result = agent_builder(
                            is_image_input=is_image,
                            model=agent_config["model"],
                            data_input=data_input,
                            prompt=agent_config["prompt"],
                            schema=agent_config["schema"],
                            judge_models=agent_config.get("judge_models")
                        )
                        result["page"] = page_idx
                        result["agent_model"] = agent_config["model"]
                        result["agent_index"] = agent_idx
                        return result
                    except Exception as e:
                        logger.error(f"Agent {agent_config['model']} failed: {str(e)}")
                        return {"error": str(e), "page": page_idx, "agent_model": agent_config["model"], "agent_index": agent_idx}
                
                # Create list with indices for proper mapping
                agents_with_idx = list(enumerate(active_agents))
                page_results = parallel_process(
                    process_agent_with_idx,
                    agents_with_idx,
                    max_workers=num_workers
                )
                # Sort by agent_index to maintain order
                page_results.sort(key=lambda x: x.get("agent_index", 0))
            else:
                page_results = []
                for agent_idx, agent_config in enumerate(active_agents):
                    try:
                        result = agent_builder(
                            is_image_input=is_image,
                            model=agent_config["model"],
                            data_input=data_input,
                            prompt=agent_config["prompt"],
                            schema=agent_config["schema"],
                            judge_models=agent_config.get("judge_models")
                        )
                        result["page"] = page_idx
                        result["agent_model"] = agent_config["model"]
                        result["agent_index"] = agent_idx
                        page_results.append(result)
                    except Exception as e:
                        logger.error(f"Agent {agent_config['model']} failed: {str(e)}")
                        page_results.append({
                            "error": str(e),
                            "page": page_idx,
                            "agent_model": agent_config["model"],
                            "agent_index": agent_idx
                        })
            
            all_results.extend(page_results)
            
            # Remove agents that found everything
            agents_to_remove = []
            for idx, (agent_config, result) in enumerate(zip(active_agents, page_results)):
                if result.get("found", False) and "error" not in result:
                    logger.info(f"Agent {agent_config['model']} found all required information on page {page_idx}")
                    agents_to_remove.append(idx)
            
            # Remove in reverse order to maintain indices
            for idx in reversed(agents_to_remove):
                active_agents.pop(idx)
        
        logger.info(f"SUCCESS: Round-robin complete. {len(all_results)} total results")
        return all_results
        
    except Exception as e:
        logger.error(f"Error in round_robin_wrapper: {str(e)}", exc_info=True)
        print(f"ERROR: round_robin_wrapper failed: {str(e)}")
        raise


# ============================================================================
# Simple Page-by-Page Processor
# ============================================================================

def process_document_pages(
    document_path: str,
    prompt_path: str,
    schema_path: str,
    model: str = "gpt5-mini",
    judge_models: Optional[List[str]] = None,
    output_csv: str = "results.csv"
) -> None:
    """
    Simple document processor that reads one page at a time.
    Stops when found=True, saves each page as image.
    
    Args:
        document_path: Path to PDF document
        prompt_path: Path to Python file containing prompt variable
        schema_path: Path to JSON file containing schema
        model: Model name to use
        judge_models: Optional list of judge models
        output_csv: Output CSV filename
    """
    try:
        logger.info(f"Processing document: {document_path}")
        logger.info(f"Method: Image extraction (saves each page as image)")
        
        # Load base prompt from file
        logger.info(f"Loading prompt from: {prompt_path}")
        base_prompt = _load_prompt_from_file(prompt_path)
        
        # Load schema from file
        logger.info(f"Loading schema from: {schema_path}")
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        # Load document and convert to images (one page at a time)
        pdf_path = Path(document_path)
        base_name = pdf_path.stem
        output_dir = Path(base_name)
        output_dir.mkdir(exist_ok=True)
        
        # Open PDF
        if PYMUPDF_AVAILABLE:
            doc = fitz.open(str(pdf_path))
        else:
            raise ImportError("PyMuPDF required for image processing")
        
        results = []
        
        # Process one page at a time
        for page_num in range(len(doc)):
            page = doc[page_num]
            pdf_page_number = page_num + 1  # 1-indexed PDF page number
            
            # Save page as image
            mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
            pix = page.get_pixmap(matrix=mat)
            image_path = output_dir / f"{base_name}_page_{pdf_page_number}.png"
            pix.save(str(image_path))
            logger.info(f"Saved image: {image_path}")
            
            # Add page number instruction to prompt
            page_prompt = f"{base_prompt}\n\nIMPORTANT: Use page number {pdf_page_number} as the source page number. This is the PDF page number (page {pdf_page_number} of {len(doc)}), NOT the page number that may be printed on the document itself."
            
            # Process page with agent
            logger.info(f"Processing page {pdf_page_number}/{len(doc)}")
            result = agent_builder(
                is_image_input=True,
                model=model,
                data_input=str(image_path),
                prompt=page_prompt,
                schema=schema,
                judge_models=judge_models
            )
            result["page"] = pdf_page_number
            results.append(result)
            
            # Check if found, stop if True
            if result.get("found", False):
                logger.info(f"SUCCESS: Found all required information on page {pdf_page_number}")
                break
        
        doc.close()
        
        # Convert to DataFrame and save
        if results:
            logger.info("Converting results to DataFrame")
            df = extract_to_dataframe(results, schema)
            df.to_csv(output_csv, index=False)
            logger.info(f"SUCCESS: Saved results to {output_csv}")
            print(f"\nProcessing Complete! Results saved to {output_csv}\n")
        else:
            logger.warning("No results generated")
        
    except Exception as e:
        logger.error(f"Error in process_document_pages: {str(e)}", exc_info=True)
        print(f"ERROR: Document processing failed: {str(e)}")
        raise


def _load_prompt_from_file(prompt_path: str) -> str:
    """Load prompt text from a Python file."""
    try:
        prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        # Try to load as Python module
        import importlib.util
        spec = importlib.util.spec_from_file_location("prompt_module", prompt_file)
        prompt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_module)
        
        # Look for common prompt variable names
        for var_name in ['prompt', 'prompt', 'PROMPT', 'prompt_text']:
            if hasattr(prompt_module, var_name):
                prompt_text = getattr(prompt_module, var_name)
                if isinstance(prompt_text, str):
                    logger.info(f"Loaded prompt from variable: {var_name}")
                    return prompt_text
        
        raise ValueError(f"No prompt variable found in {prompt_path}. Expected: prompt, prompt1, PROMPT, or prompt_text")
        
    except Exception as e:
        logger.error(f"Error loading prompt: {str(e)}")
        raise

