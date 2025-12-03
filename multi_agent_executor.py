"""
Multi-Agent Executor - Dynamic multi-agent document processor
Processes documents with dynamically loaded agents using parallel processing.
"""

import json
import logging
from pathlib import Path
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from agent_toolkit import (
    document_loader,
    agent_builder,
    _load_prompt_from_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_agent_executor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_agents(agents_dir: str = "agents") -> List[Dict[str, Any]]:
    """Dynamically load all agents from agents folder."""
    agents_dir = Path(agents_dir)
    logger.info(f"Loading agents from: {agents_dir}")
    
    agents = []
    agent_files = sorted(agents_dir.glob("*.py"))
    
    for prompt_file in agent_files:
        agent_name = prompt_file.stem
        schema_file = agents_dir / f"{agent_name}_schema.json"
        
        if not schema_file.exists():
            logger.warning(f"Schema not found for {agent_name}, skipping")
            continue
        
        try:
            prompt = _load_prompt_from_file(str(prompt_file))
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            agents.append({
                "agent_name": agent_name,
                "prompt": prompt,
                "schema": schema
            })
            logger.info(f"Loaded agent: {agent_name}")
        except Exception as e:
            logger.error(f"Error loading agent {agent_name}: {str(e)}")
    
    logger.info(f"Total agents loaded: {len(agents)}")
    return agents


def process_agent_page(args: tuple) -> Dict:
    """Process a single agent on current page (for parallel execution)."""
    agent_config, page_data, page_num, total_pages = args
    
    agent_name = agent_config["agent_name"]
    prompt = agent_config["prompt"]
    schema = agent_config["schema"]
    model = agent_config.get("model", "gpt5-mini")
    judge_models = agent_config.get("judge_models", ["gpt-4o"])
    is_image = agent_config.get("is_image", True)
    
    try:
        # Add page number instruction to prompt
        page_prompt = f"{prompt}\n\nIMPORTANT: Use page number {page_num} as the source page number. This is the PDF page number (page {page_num} of {total_pages}), NOT the page number that may be printed on the document itself."
        
        logger.info(f"Agent {agent_name} processing page {page_num}")
        
        result = agent_builder(
            is_image_input=is_image,
            model=model,
            data_input=page_data,
            prompt=page_prompt,
            schema=schema,
            judge_models=judge_models
        )
        
        result["page"] = page_num
        result["agent_name"] = agent_name
        
        found = result.get("found", False)
        logger.info(f"Agent {agent_name} page {page_num}: found={found}")
        
        return result
        
    except Exception as e:
        logger.error(f"Agent {agent_name} failed on page {page_num}: {str(e)}")
        return {
            "error": str(e),
            "page": page_num,
            "agent_name": agent_name,
            "found": False
        }


def process_document_multi_agent(
    document_path: str,
    agents_dir: str = "agents",
    model: str = "gpt5-mini",
    judge_models: Optional[List[str]] = None,
    is_image: bool = True,
    output_csv: str = "results_multi_agent.csv"
) -> None:
    """
    Process document with multiple agents in parallel.
    
    Args:
        document_path: Path to document (PDF/DOCX/TXT)
        agents_dir: Directory containing agent files
        model: LLM model to use
        judge_models: Optional list of judge models
        is_image: Whether to process as images (True) or text (False)
        output_csv: Output CSV filename
    """
    logger.info(f"Starting multi-agent processing: document={document_path}, is_image={is_image}")
    
    # Load agents dynamically
    agents = load_agents(agents_dir)
    if not agents:
        logger.error("No agents loaded, exiting")
        return
    
    # Add model and judge_models to each agent config
    judge_models = judge_models or ["gpt-4o"]
    for agent in agents:
        agent["model"] = model
        agent["judge_models"] = judge_models
        agent["is_image"] = is_image
    
    # Load document and convert to images/text
    logger.info(f"Loading document: {document_path}")
    if is_image:
        pages = document_loader(document_path, is_image=True)
        logger.info(f"Converted {len(pages)} pages to images")
    else:
        pages = document_loader(document_path, is_image=False)
        logger.info(f"Loaded {len(pages)} pages as text")
    
    total_pages = len(pages)
    active_agents = agents.copy()
    all_results = []
    
    # Process pages one at a time
    for page_idx, page_data in enumerate(pages, start=1):
        if not active_agents:
            logger.info("All agents completed, stopping")
            break
        
        logger.info(f"Processing page {page_idx}/{total_pages} with {len(active_agents)} active agents")
        
        # Prepare arguments for parallel processing
        pool_args = [(agent, page_data, page_idx, total_pages) for agent in active_agents]
        
        # Process all active agents in parallel
        with Pool(processes=len(active_agents)) as pool:
            page_results = pool.map(process_agent_page, pool_args)
        
        # Update results and remove completed agents
        agents_to_remove = []
        for idx, (agent, result) in enumerate(zip(active_agents, page_results)):
            all_results.append(result)
            
            if result.get("found", False) and "error" not in result:
                logger.info(f"Agent {agent['agent_name']} completed on page {page_idx}")
                agents_to_remove.append(idx)
        
        # Remove completed agents (reverse order to maintain indices)
        for idx in reversed(agents_to_remove):
            active_agents.pop(idx)
    
    # Filter and combine results
    logger.info("Combining results from all agents")
    valid_results = [r for r in all_results if "error" not in r and r.get("found", False)]
    
    if not valid_results:
        logger.warning("No valid results found")
        return
    
    # Convert to DataFrame
    logger.info("Converting results to DataFrame")
    all_extracted = []
    
    for result in valid_results:
        agent_name = result.get("agent_name", "unknown")
        domain = result.get("domain", "")
        confidence = result.get("confidence", "")
        confidence_keyword = result.get("confidence_keyword", "")
        summary = result.get("summary", "")
        extracted = result.get("extracted", {})
        page = result.get("page", "")
        
        # Normalize confidence
        if isinstance(confidence, (int, float)):
            confidence = str(confidence)
        confidence_keyword_lower = confidence_keyword.lower() if confidence_keyword else ""
        if any(kw in confidence_keyword_lower for kw in ['clearly stated', 'explicitly mentioned', 'directly specified', 'clearly indicates']):
            confidence = "1"
        elif str(confidence).strip() in ["1", "1.0"]:
            confidence = "1"
        
        # Extract all items
        for key, item_data in extracted.items():
            if isinstance(item_data, dict):
                value = item_data.get("value", "")
                source = item_data.get("source", "")
                
                # Skip invalid values
                if value and str(value).lower().strip() not in ["not found", "nan", "none", ""]:
                    all_extracted.append({
                        "domain": domain,
                        "value_type": key,
                        "value": value,
                        "source": f"page {page}" if page else source,
                        "confidence": confidence,
                        "confidence_keyword": confidence_keyword,
                        "summary": summary,
                        "agent": agent_name
                    })
    
    # Create and save DataFrame
    df = pd.DataFrame(all_extracted)
    
    if not df.empty:
        df = df.dropna(subset=['value'])
        df = df[df['value'].astype(str).str.strip().str.lower() != 'not found']
        df = df[df['value'].astype(str).str.strip() != '']
        df = df[df['value'].astype(str).str.strip().str.lower() != 'nan']
        df = df[df['value'].astype(str).str.strip().str.lower() != 'none']
    
    df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}: {len(df)} rows")
    print(f"\n✅ Processing complete! Results saved to {output_csv}")
    print(f"   Total rows: {len(df)}")
    print(f"   Agents: {df['agent'].unique().tolist() if 'agent' in df.columns else 'N/A'}\n")


if __name__ == "__main__":
    # Configuration
    DOCUMENT_PATH = "solar.pdf"
    AGENTS_DIR = "agents"
    MODEL = "gpt5-mini"
    JUDGE_MODELS = ["gpt-4o"]
    IS_IMAGE = True
    OUTPUT_CSV = "results_multi_agent.csv"
    
    process_document_multi_agent(
        document_path=DOCUMENT_PATH,
        agents_dir=AGENTS_DIR,
        model=MODEL,
        judge_models=JUDGE_MODELS,
        is_image=IS_IMAGE,
        output_csv=OUTPUT_CSV
    )

