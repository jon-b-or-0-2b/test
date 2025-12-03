"""
Flask backend for multi-agent document processor
"""

import os
import json
import logging
import threading
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import pandas as pd
from multi_agent_executor import process_document_multi_agent, load_agents

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create folders
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global progress tracking
processing_status = {
    'status': 'idle',  # idle, processing, completed, error
    'progress': {},  # {agent_name: percentage}
    'total_pages': 0,
    'current_page': 0,
    'filename': None,
    'image_paths': []  # Store image paths for page viewing
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get list of all available agents"""
    try:
        agents = load_agents('agents')
        agent_names = [agent['agent_name'] for agent in agents]
        return jsonify({'agents': agent_names})
    except Exception as e:
        logger.error(f"Error loading agents: {str(e)}")
        return jsonify({'error': str(e)}), 500


def process_file_background(filepath, output_path, output_filename):
    """Process file in background thread with progress tracking"""
    global processing_status
    
    try:
        # Load agents to get names
        agents = load_agents('agents')
        agent_names = [agent['agent_name'] for agent in agents]
        
        # Initialize progress
        processing_status['progress'] = {name: 0 for name in agent_names}
        processing_status['status'] = 'processing'
        
        # Get image paths first
        from agent_toolkit import document_loader
        image_paths = document_loader(filepath, is_image=True)
        processing_status['image_paths'] = image_paths
        processing_status['total_pages'] = len(image_paths)
        total_pages = len(image_paths)
        
        # Process using multi_agent_executor logic with progress updates
        from multi_agent_executor import load_agents as load_exec_agents, process_agent_page
        from agent_toolkit import agent_builder
        from multiprocessing import Pool
        import pandas as pd
        
        agents_config = load_exec_agents('agents')
        for agent in agents_config:
            agent['model'] = 'gpt5-mini'
            agent['judge_models'] = ['gpt-4o']
            agent['is_image'] = True
        
        active_agents = agents_config.copy()
        all_results = []
        
        # Process pages one at a time with progress updates
        for page_idx, page_data in enumerate(image_paths, start=1):
            if not active_agents:
                break
            
            # Update progress
            progress_pct = int((page_idx / total_pages) * 100)
            for agent in active_agents:
                agent_name = agent['agent_name']
                processing_status['progress'][agent_name] = progress_pct
            processing_status['current_page'] = page_idx
            
            # Process agents
            pool_args = [(agent, page_data, page_idx, total_pages) for agent in active_agents]
            with Pool(processes=len(active_agents)) as pool:
                page_results = pool.map(process_agent_page, pool_args)
            
            # Update results and remove completed agents
            agents_to_remove = []
            for idx, (agent, result) in enumerate(zip(active_agents, page_results)):
                all_results.append(result)
                if result.get("found", False) and "error" not in result:
                    processing_status['progress'][agent['agent_name']] = 100
                    agents_to_remove.append(idx)
            
            for idx in reversed(agents_to_remove):
                active_agents.pop(idx)
        
        # Process results and save CSV (same logic as multi_agent_executor)
        valid_results = [r for r in all_results if "error" not in r and r.get("found", False)]
        if not valid_results:
            processing_status['status'] = 'error'
            processing_status['error'] = 'No valid results found'
            return
        
        all_extracted = []
        for result in valid_results:
            agent_name = result.get("agent_name", "unknown")
            domain = result.get("domain", "")
            confidence = result.get("confidence", "")
            confidence_keyword = result.get("confidence_keyword", "")
            summary = result.get("summary", "")
            extracted = result.get("extracted", {})
            page = result.get("page", "")
            
            if isinstance(confidence, (int, float)):
                confidence = str(confidence)
            confidence_keyword_lower = confidence_keyword.lower() if confidence_keyword else ""
            if any(kw in confidence_keyword_lower for kw in ['clearly stated', 'explicitly mentioned', 'directly specified', 'clearly indicates']):
                confidence = "1"
            elif str(confidence).strip() in ["1", "1.0"]:
                confidence = "1"
            
            for key, item_data in extracted.items():
                if isinstance(item_data, dict):
                    value = item_data.get("value", "")
                    source = item_data.get("source", "")
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
        
        df = pd.DataFrame(all_extracted)
        if not df.empty:
            df = df.dropna(subset=['value'])
            df = df[df['value'].astype(str).str.strip().str.lower() != 'not found']
            df = df[df['value'].astype(str).str.strip() != '']
            df = df[df['value'].astype(str).str.strip().str.lower() != 'nan']
            df = df[df['value'].astype(str).str.strip().str.lower() != 'none']
        
        df.to_csv(output_path, index=False)
        
        # Mark all agents as complete
        processing_status['progress'] = {name: 100 for name in agent_names}
        processing_status['status'] = 'completed'
        processing_status['filename'] = output_filename
        
    except Exception as e:
        logger.error(f"Background processing error: {str(e)}")
        processing_status['status'] = 'error'
        processing_status['error'] = str(e)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start processing"""
    global processing_status
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        # Reset processing status
        processing_status = {
            'status': 'processing',
            'progress': {},
            'total_pages': 0,
            'current_page': 0,
            'filename': None,
            'image_paths': []
        }
        
        output_filename = f"{Path(filename).stem}_results.csv"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Start processing in background thread
        thread = threading.Thread(
            target=process_file_background,
            args=(filepath, output_path, output_filename)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Processing started'
        })
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<filename>', methods=['GET'])
def get_results(filename):
    """Get CSV results as JSON"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(filepath)
        return jsonify({
            'data': df.to_dict('records'),
            'columns': list(df.columns)
        })
    except Exception as e:
        logger.error(f"Error reading results: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download CSV file"""
    try:
        return send_from_directory(
            app.config['RESULTS_FOLDER'],
            filename,
            as_attachment=True,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for a row"""
    try:
        data = request.json
        feedback = {
            'row_id': data.get('row_id'),
            'text': data.get('text', ''),
            'score': data.get('score'),
            'page': data.get('page'),
            'extracted_value': data.get('extracted_value'),
            'summary': data.get('summary')
        }
        
        # Save feedback to file (simple implementation)
        feedback_file = os.path.join(app.config['RESULTS_FOLDER'], 'feedback.json')
        feedbacks = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedbacks = json.load(f)
        
        feedbacks.append(feedback)
        
        with open(feedback_file, 'w') as f:
            json.dump(feedbacks, f, indent=2)
        
        logger.info(f"Feedback submitted: {feedback}")
        return jsonify({'success': True, 'message': 'Feedback saved'})
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get processing progress"""
    global processing_status
    return jsonify(processing_status)


@app.route('/api/image/<int:page_num>', methods=['GET'])
def get_page_image(page_num):
    """Get image for a specific page"""
    global processing_status
    try:
        image_paths = processing_status.get('image_paths', [])
        if 1 <= page_num <= len(image_paths):
            image_path = image_paths[page_num - 1]
            if os.path.exists(image_path):
                return send_file(image_path, mimetype='image/png')
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    # Disable reloader to avoid conflicts with multiprocessing on Windows
    app.run(debug=True, port=5000, use_reloader=False)

