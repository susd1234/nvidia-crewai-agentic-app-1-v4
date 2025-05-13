# NVIDIA NIM & Agentic AI Powered GitHub Repo Documentation Generator App

Tech Stack

LIM Model (NVIDIA NIM) - meta/1lama-3.1-70b-instruct (LLM Model)
Embedding (NVIDIA NIM) - nvidia/nemo-retriever-e5-large
Agentic AI Framework - Crew AI
Vector Database - ChromaDB
I/P - GitHub Repo Clone Link
Generated O/P File Types - PDE / MDX / DOCX

An AI-powered documentation generator that uses NVIDIA's AI endpoints and CrewAI to create comprehensive documentation for any GitHub repository automatically.

# UI
<img width="1732" alt="image" src="https://github.com/user-attachments/assets/53a60e6d-b847-4223-aa44-67f8ecf19ace" />


# Tech Stack
<img width="1149" alt="image" src="https://github.com/user-attachments/assets/4ceb5f02-feeb-4d9e-a6b0-e20d4ed32166" />



## Features

- Analyzes GitHub repositories to understand their structure and purpose
- Creates a documentation plan based on the repository analysis
- Generates documentation in multiple formats (MDX, PDF, DOCX)
- Includes Mermaid diagrams to visualize complex concepts
- Provides a FastAPI interface for integrating with other applications
- Features a user-friendly Gradio UI for easy interaction

## Installation

1. Clone this repository:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables by creating a `.env` file:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
```

## Usage

### Gradio UI (Recommended)

The easiest way to use the application is through the Gradio UI:

```bash
python run_app.py
```

This will:
1. Start the API server in the background
2. Launch the Gradio UI
3. Open your default web browser to the UI

The UI provides:
- A simple form to enter a GitHub repository URL
- Real-time progress tracking with visual indicators
- Interactive display of the documentation plan
- Easy viewing of generated documentation files
- Support for various file formats (Markdown, HTML, JSON, etc.)

### Command-line Interface

Run the application to generate documentation for a GitHub repository:

```bash
python app.py
```

You'll be prompted to enter a GitHub repository URL and select an output format (MDX, PDF, or DOCX). The application will clone the repository, analyze it, and generate documentation in the `workdir` directory.

### API Interface

Start the API server:

```bash
python run_api.py
```

The API server will run at http://localhost:8000 and provides the following endpoints:

#### Generate Documentation

```
POST /generate
```

Request body:
```json
{
  "github_url": "https://github.com/username/repository",
  "email": "optional-email@example.com",
  "output_format": "mdx" // Options: "mdx", "pdf", "docx"
}
```

Response:
```json
{
  "job_id": "unique-job-id",
  "status": "queued",
  "message": "Documentation generation job queued successfully. Output format: mdx"
}
```

#### Check Job Status

```
GET /status/{job_id}
```

Response:
```json
{
  "job_id": "unique-job-id",
  "status": "completed|running|failed|queued",
  "message": "Status message",
  "docs": ["list", "of", "generated", "files"],
  "plan": "Documentation plan in JSON format",
  "output_format": "mdx" // or "pdf", "docx"
}
```

#### List All Jobs

```
GET /jobs
```

Response:
```json
[
  {
    "job_id": "unique-job-id-1",
    "status": "completed",
    "message": "Documentation generation completed successfully",
    "docs": ["list", "of", "generated", "files"],
    "plan": "Documentation plan in JSON format"
  },
  {
    "job_id": "unique-job-id-2",
    "status": "running",
    "message": "Documentation generation in progress",
    "docs": [],
    "plan": null
  }
]
```

## API Usage Example

```python
import requests

# Start a documentation generation job with PDF output
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "github_url": "https://github.com/username/repository",
        "output_format": "pdf"  // Options: "mdx", "pdf", "docx"
    }
)
job_id = response.json()["job_id"]

# Check job status
status_response = requests.get(f"http://localhost:8000/status/{job_id}")
print(status_response.json())
```

## Running Individual Components

If you need to run the components separately:

### Run just the API server:
```bash
python run_api.py
```
### Run just the Gradio UI:
```bash
python app_ui.py
```

## License

[Insert license information here]

