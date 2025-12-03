# Multi-Agent Document Processor Web Application

A simple Flask backend with AngularJS frontend for processing documents with multiple agents.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create required folders (they will be created automatically, but you can create them manually):
```bash
mkdir uploads results
```

3. Make sure you have agents in the `agents` folder with the naming convention:
   - `{agent_name}.py` - contains the prompt
   - `{agent_name}_schema.json` - contains the schema

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload a file**: Click "Choose File" and select a PDF, DOCX, or TXT file
2. **Process**: Click "Upload & Process" to start processing
3. **View Results**: Once processing completes, results will appear in the table
4. **Provide Feedback**: Click "Feedback" button on any row to submit feedback
5. **Download**: Click "Download CSV" to download the results
6. **Generate Report**: Placeholder button (not yet implemented)

## Features

- **Side A (30% width)**: Shows all agents with progress bars
- **Side B (70% width)**:
  - **B1 (80% height)**: Displays CSV results in a table with feedback buttons
  - **B2 (20% height)**: Download CSV and Generate Report buttons
- **Top**: File upload area

## API Endpoints

- `GET /api/agents` - Get list of all agents
- `POST /api/upload` - Upload and process a file
- `GET /api/results/<filename>` - Get results as JSON
- `GET /api/download/<filename>` - Download CSV file
- `POST /api/feedback` - Submit feedback for a row

## File Structure

```
.
├── app.py                 # Flask backend
├── static/
│   ├── index.html        # Frontend HTML
│   └── app.js            # AngularJS controller
├── uploads/              # Uploaded files
├── results/              # Generated CSV files and feedback
└── agents/               # Agent definitions
```

