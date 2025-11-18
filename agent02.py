"""
Agent02 - Multi-agent document processor using round-robin
"""

import json
from pathlib import Path
from agent_toolkit import (
    document_loader,
    round_robin_wrapper,
    extract_to_dataframe,
    _load_prompt_from_file
)

# Configuration - set these paths manually
PROMPT1_PATH = "land_and_layout.py"
SCHEMA1_PATH = "prompt1_schema.json"
PROMPT2_PATH = "cables.py"
SCHEMA2_PATH = "prompt2_schema.json"
DOCUMENT_PATH = "solar.pdf"
MODEL = "gpt5-mini"
JUDGE_MODELS = ["gpt-4o"]
OUTPUT_CSV = "results_multi_agent.csv"
PARALLEL_PROCESS = False
NUM_WORKERS = 10

if __name__ == "__main__":
    # Load prompts
    print(f"Loading prompt 1 from: {PROMPT1_PATH}")
    prompt1 = _load_prompt_from_file(PROMPT1_PATH)
    
    print(f"Loading prompt 2 from: {PROMPT2_PATH}")
    prompt2 = _load_prompt_from_file(PROMPT2_PATH)
    
    # Load schemas
    print(f"Loading schema 1 from: {SCHEMA1_PATH}")
    with open(SCHEMA1_PATH, 'r', encoding='utf-8') as f:
        schema1 = json.load(f)
    
    print(f"Loading schema 2 from: {SCHEMA2_PATH}")
    with open(SCHEMA2_PATH, 'r', encoding='utf-8') as f:
        schema2 = json.load(f)
    
    # Create agent configurations
    agents = [
        {
            "model": MODEL,
            "prompt": prompt1,
            "schema": schema1,
            "judge_models": JUDGE_MODELS,
            "agent_name": "land_and_layout"
        },
        {
            "model": MODEL,
            "prompt": prompt2,
            "schema": schema2,
            "judge_models": JUDGE_MODELS,
            "agent_name": "cables"
        }
    ]
    
    print(f"\nProcessing document: {DOCUMENT_PATH}")
    print(f"Using {len(agents)} agents:")
    for idx, agent in enumerate(agents, 1):
        print(f"  Agent {idx}: {agent['agent_name']}")
    
    # Load document and convert to images
    print("\nLoading document and converting to images...")
    image_paths = document_loader(DOCUMENT_PATH, is_image=True)
    print(f"Converted {len(image_paths)} pages to images\n")
    
    # Add page number instructions to prompts for each page
    # We'll modify agents to include page-aware prompts
    def create_page_aware_agents(base_agents, page_num, total_pages):
        """Create agents with page number instructions added to prompts."""
        page_aware_agents = []
        for agent in base_agents:
            page_instruction = f"\n\nIMPORTANT: Use page number {page_num} as the source page number. This is the PDF page number (page {page_num} of {total_pages}), NOT the page number that may be printed on the document itself."
            page_aware_agent = agent.copy()
            page_aware_agent["prompt"] = agent["prompt"] + page_instruction
            page_aware_agents.append(page_aware_agent)
        return page_aware_agents
    
    # Process with round-robin wrapper
    # We need to modify round_robin_wrapper to accept page-aware prompts
    # For now, we'll process pages manually with round-robin logic
    print("Processing pages with round-robin wrapper...")
    
    # Open PDF to get total page count
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(DOCUMENT_PATH)
        total_pages = len(doc)
        doc.close()
    except ImportError:
        # Fallback: use number of images
        total_pages = len(image_paths)
    
    # Process pages one at a time with both agents
    # Track final results for each agent (only save when found=True or pages run out)
    final_results = {}  # agent_name -> final result
    active_agents = agents.copy()
    last_processed_page = 0
    
    for page_idx, image_path in enumerate(image_paths, start=1):
        last_processed_page = page_idx
        if not active_agents:
            print("All agents completed, stopping")
            break
        
        print(f"\nProcessing page {page_idx}/{len(image_paths)} with {len(active_agents)} active agents")
        
        # Create page-aware agents for this page
        page_aware_agents = create_page_aware_agents(active_agents, page_idx, total_pages)
        
        # Process all active agents on this page
        page_results = []
        for agent_config in page_aware_agents:
            try:
                from agent_toolkit import agent_builder
                result = agent_builder(
                    is_image_input=True,
                    model=agent_config["model"],
                    data_input=image_path,
                    prompt=agent_config["prompt"],
                    schema=agent_config["schema"],
                    judge_models=agent_config.get("judge_models")
                )
                result["page"] = page_idx
                result["agent_model"] = agent_config["model"]
                result["agent_name"] = agent_config["agent_name"]
                page_results.append(result)
                
                # Update final result for this agent (always keep latest)
                agent_name = agent_config["agent_name"]
                final_results[agent_name] = result
                
            except Exception as e:
                print(f"ERROR: Agent {agent_config['agent_name']} failed: {str(e)}")
                error_result = {
                    "error": str(e),
                    "page": page_idx,
                    "agent_name": agent_config["agent_name"]
                }
                page_results.append(error_result)
        
        # Remove agents that found everything
        agents_to_remove = []
        for idx, (agent_config, result) in enumerate(zip(active_agents, page_results)):
            if result.get("found", False) and "error" not in result:
                print(f"Agent {agent_config['agent_name']} found all required information on page {page_idx}")
                agents_to_remove.append(idx)
        
        # Remove in reverse order to maintain indices
        for idx in reversed(agents_to_remove):
            active_agents.pop(idx)
    
    # Get final results (only from agents that found everything OR ran out of pages)
    # Track if we processed all pages or all agents completed
    processed_all_pages = last_processed_page >= len(image_paths)
    all_agents_completed = len(active_agents) == 0
    
    print(f"\nFinal results tracking:")
    print(f"  Processed all pages: {processed_all_pages}")
    print(f"  All agents completed: {all_agents_completed}")
    print(f"  Final results keys: {list(final_results.keys())}")
    
    results = []
    for agent_name, result in final_results.items():
        # Include result if:
        # 1. Agent found everything (found=True), OR
        # 2. We processed all pages (ran out of pages), OR
        # 3. All agents completed (stopped early - means all found everything)
        found_status = result.get("found", False)
        
        # If all agents completed, include all results (they all found everything)
        # Otherwise, only include if agent found everything or we ran out of pages
        if all_agents_completed:
            should_include = True  # All agents completed, so all found everything
        else:
            should_include = found_status or processed_all_pages
        
        print(f"  Agent {agent_name}: found={found_status}, should_include={should_include}, has_error={'error' in result}")
        
        if should_include:
            # Check if result has valid data (not just errors)
            if "error" not in result:
                results.append(result)
                print(f"    -> INCLUDED")
            else:
                print(f"    -> EXCLUDED (has error)")
        else:
            print(f"    -> EXCLUDED (doesn't meet criteria)")
    
    print(f"\nProcessed {len(results)} results")
    
    # Combine results from both agents and convert to DataFrame
    if results:
        print("Converting results to DataFrame...")
        
        # Combine all results into a single list for DataFrame conversion
        # We'll use the first schema for structure, but extract all items
        all_extracted_results = []
        
        for result in results:
            # Add agent name to result for tracking
            agent_name = result.get("agent_name", "unknown")
            domain = result.get("domain", "")
            confidence = result.get("confidence", "")
            confidence_keyword = result.get("confidence_keyword", "").lower()
            extracted = result.get("extracted", {})
            page = result.get("page", "")
            
            # Convert confidence to string if needed
            if isinstance(confidence, (int, float)):
                confidence = str(confidence)
            
            # Set confidence to "1" if "clearly stated" or similar
            if any(keyword in confidence_keyword for keyword in ['clearly stated', 'explicitly mentioned', 'directly specified', 'clearly indicates']):
                confidence = "1"
            elif str(confidence).strip() in ["1", "1.0"]:
                confidence = "1"
            
            # Create a row for each extracted item
            for key, item_data in extracted.items():
                if isinstance(item_data, dict):
                    value = item_data.get("value", "")
                    source = item_data.get("source", "")
                    
                    # Skip "not found" values and empty/NaN values
                    if value and str(value).lower().strip() not in ["not found", "nan", "none", ""]:
                        # Use PDF page number
                        if source and "page" in str(source).lower():
                            source = f"page {page}"
                        elif not source:
                            source = f"page {page}"
                        
                        all_extracted_results.append({
                            "domain": domain,
                            "value_type": key,
                            "value": value,
                            "source": source,
                            "confidence": confidence,
                            "agent": agent_name
                        })
        
        # Create DataFrame
        import pandas as pd
        import numpy as np
        df = pd.DataFrame(all_extracted_results)
        
        # Remove rows with NaN values in critical columns
        if not df.empty:
            # Drop rows where value is NaN or empty
            df = df.dropna(subset=['value'])
            # Remove rows where value is still empty string or "not found" (in case it got through)
            df = df[df['value'].astype(str).str.strip().str.lower() != 'not found']
            df = df[df['value'].astype(str).str.strip() != '']
            df = df[df['value'].astype(str).str.strip().str.lower() != 'nan']
            df = df[df['value'].astype(str).str.strip().str.lower() != 'none']
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSUCCESS: Results saved to {OUTPUT_CSV}")
        print(f"Total rows: {len(df)}")
        print(f"Agents: {df['agent'].unique().tolist() if 'agent' in df.columns else 'N/A'}")
    else:
        print("WARNING: No results generated")

