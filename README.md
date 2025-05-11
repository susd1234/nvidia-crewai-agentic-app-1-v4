# Documentation Generator

An AI-powered documentation generator that uses NVIDIA's AI endpoints and CrewAI to automatically create comprehensive documentation for any GitHub repository.

## Features

- Analyzes GitHub repositories to understand their structure and purpose
- Creates a documentation plan based on the repository analysis
- Generates MDX documentation files with rich content and examples
- Includes Mermaid diagrams to visualize complex concepts
- Provides a FastAPI interface for integrating with other applications

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

### Command-line Interface

Run the application to generate documentation for a GitHub repository:

```bash
python app.py
```

You'll be prompted to enter a GitHub repository URL. The application will clone the repository, analyze it, and generate documentation in the `workdir` directory.

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
  "email": "optional-email@example.com"
}
```

Response:
```json
{
  "job_id": "unique-job-id",
  "status": "queued",
  "message": "Documentation generation job queued successfully"
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
  "plan": "Documentation plan in JSON format"
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

# Start a documentation generation job
response = requests.post(
    "http://localhost:8000/generate",
    json={"github_url": "https://github.com/username/repository"}
)
job_id = response.json()["job_id"]

# Check job status
status_response = requests.get(f"http://localhost:8000/status/{job_id}")
print(status_response.json())
```

## License

[Insert license information here]
