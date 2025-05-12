import os
import uuid
import datetime
from pathlib import Path
import logging
import traceback
import subprocess
from typing import Dict, List, Optional, Literal
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import sys


# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set higher logging levels for LiteLLM related loggers
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm.cost_calculator").setLevel(logging.CRITICAL)
logging.getLogger("litellm.llms").setLevel(logging.CRITICAL)
logging.getLogger("litellm.litellm").setLevel(logging.CRITICAL)
logging.getLogger("litellm.utils").setLevel(logging.CRITICAL)


# Configure a custom filter for logs
class NoLiteLLMFilter(logging.Filter):
    def filter(self, record):
        if record.name.startswith("litellm") or "LiteLLM" in record.getMessage():
            return False
        if "HTTP Request" in record.getMessage():
            return False
        if "cost_calculator" in record.getMessage():
            return False
        if "selected model name" in record.getMessage():
            return False
        if "llama-3" in record.getMessage():
            return False
        if "Wrapper: Completed Call" in record.getMessage():
            return False
        return True


# Add filter to root logger to prevent duplicate messages
root_logger = logging.getLogger()
root_logger.addFilter(NoLiteLLMFilter())

# Create API logger
api_logger = logging.getLogger("api")
api_logger.setLevel(logging.INFO)

# Import from app.py
planning_crew = None
documentation_crew = None
DocumentationState = None

try:
    # Import only what we need - avoid importing logger from app.py
    from app import planning_crew, documentation_crew, DocumentationState

    api_logger.info("Successfully imported from app.py")
except Exception as e:
    api_logger.error(f"Error importing from app.py: {e}")
    api_logger.error(traceback.format_exc())
    raise

# FastAPI app setup
app = FastAPI(
    title="Documentation Generator API",
    description="API for generating documentation from GitHub repositories",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure logging on startup
@app.on_event("startup")
async def configure_logging():
    """Configure logging when the API starts up"""
    api_logger.info("Configuring logging for API server")

    # Set all LiteLLM loggers to CRITICAL level
    for logger_name in [
        "litellm",
        "LiteLLM",
        "litellm.cost_calculator",
        "litellm.llms",
        "litellm.litellm",
        "litellm.utils",
        "openai",
        "httpx",
        "httpcore",
    ]:
        log = logging.getLogger(logger_name)
        log.setLevel(logging.CRITICAL)
        log.propagate = False
        # Remove all handlers and add a null handler
        for handler in log.handlers[:]:
            log.removeHandler(handler)
        log.addHandler(logging.NullHandler())

    # Try to monkeypatch the LiteLLM cost calculator
    try:
        import litellm.cost_calculator
        import functools

        # Override the info method of the cost calculator logger
        original_info = litellm.cost_calculator.logger.info

        @functools.wraps(original_info)
        def filtered_info(msg, *args, **kwargs):
            if "selected model name for cost calculation" in msg:
                return None  # Skip these messages
            return original_info(msg, *args, **kwargs)

        litellm.cost_calculator.logger.info = filtered_info
        api_logger.info("Successfully patched LiteLLM cost calculator logging")
    except Exception as e:
        api_logger.warning(f"Could not patch LiteLLM cost calculator logging: {e}")

    api_logger.info("Logging configuration complete")


# Store for active jobs
job_store: Dict[str, Dict] = {}


class DocumentationRequest(BaseModel):
    github_url: str = Field(
        ..., description="GitHub repository URL to generate documentation for"
    )
    email: Optional[str] = Field(
        None, description="Email to notify when documentation is complete"
    )
    output_format: Literal["mdx", "pdf", "docx"] = Field(
        "mdx",
        description="Format of the generated documentation files (mdx, pdf, docx)",
    )


class DocumentationResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    docs: Optional[List[str]] = None
    plan: Optional[str] = None
    output_format: Optional[str] = None


def generate_documentation_background(
    job_id: str, github_url: str, output_format: str = "mdx"
):
    """Background task to generate documentation"""
    api_logger.info(
        f"Starting documentation generation for job {job_id} - Repository: {github_url}, Format: {output_format}"
    )
    try:
        job_store[job_id]["status"] = "running"
        api_logger.info(f"Job {job_id} status set to 'running'")

        # Set up paths for this job
        timestamp = datetime.datetime.now().strftime("%d%b%y_%H%M%S")
        repo_name = github_url.split("/")[-1]
        execution_folder = f"{repo_name}_{timestamp}"
        api_logger.info(f"Creating execution folder: {execution_folder}")

        workdir_path = Path("workdir")
        workdir_path.mkdir(exist_ok=True)

        execution_path = workdir_path / execution_folder
        execution_path.mkdir(exist_ok=True)

        input_path = execution_path / "input"
        input_path.mkdir(exist_ok=True)

        output_path = execution_path / "output"
        output_path.mkdir(exist_ok=True)
        api_logger.info(f"Directory structure created at {execution_path}")

        # Instead of using CrewAI Flow, implement the documentation generation steps directly
        api_logger.info(f"Starting documentation generation process directly")

        input_path_str = str(input_path)
        output_path_str = str(output_path)

        # 1. Clone the repository
        api_logger.info(f"Cloning repository: {github_url} to {input_path_str}")
        try:
            subprocess.run(
                ["git", "clone", github_url, input_path_str],
                check=True,
            )
            api_logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            api_logger.error(f"Failed to clone repository: {e}")
            raise Exception(f"Failed to clone repository: {e}")

        # 2. Import the necessary modules from app.py for documentation generation
        api_logger.info("Importing modules for documentation generation")
        from app import (
            planning_crew,
            documentation_crew,
            logger,
            convert_markdown_to_pdf,
            convert_markdown_to_docx,
        )

        # 3. Generate documentation plan
        api_logger.info(
            f"Generating documentation plan for repository at {input_path_str}"
        )
        try:
            plan_result = planning_crew.kickoff(inputs={"repo_path": input_path_str})
            api_logger.info("Documentation planning completed")

            # Save the plan
            docs_dir = Path(output_path_str)
            docs_dir.mkdir(exist_ok=True)
            plan_file = docs_dir / "plan.json"
            api_logger.info(f"Saving documentation plan to {plan_file}")
            with open(plan_file, "w") as f:
                f.write(plan_result.raw)
            api_logger.info("Documentation plan saved successfully")

            job_store[job_id]["plan"] = plan_result.raw

            # 4. Generate documentation for each planned document
            api_logger.info("Starting documentation creation process")
            generated_docs = []

            for doc in plan_result.pydantic.docs:
                api_logger.info(f"Creating documentation for: {doc.title}")
                try:
                    result = documentation_crew.kickoff(
                        inputs={
                            "repo_path": input_path_str,
                            "title": doc.title,
                            "overview": plan_result.pydantic.overview,
                            "description": doc.description,
                            "prerequisites": doc.prerequisites,
                            "examples": "\n".join(doc.examples),
                            "goal": doc.goal,
                        }
                    )

                    base_title = doc.title.lower().replace(" ", "_")

                    if output_format == "mdx":
                        file_ext = ".mdx"
                        doc_file = docs_dir / f"{base_title}{file_ext}"
                        with open(doc_file, "w") as f:
                            f.write(result.raw)
                        api_logger.info(f"Documentation saved to {doc_file}")
                    elif output_format == "pdf":
                        file_ext = ".pdf"
                        doc_file = docs_dir / f"{base_title}{file_ext}"
                        convert_markdown_to_pdf(result.raw, doc_file)
                    elif output_format == "docx":
                        file_ext = ".docx"
                        doc_file = docs_dir / f"{base_title}{file_ext}"
                        convert_markdown_to_docx(result.raw, doc_file)

                    generated_docs.append(str(doc_file))
                    api_logger.info(f"Documentation saved to {doc_file}")
                except Exception as e:
                    api_logger.error(
                        f"Error creating documentation for {doc.title}: {e}"
                    )
                    continue

            api_logger.info(f"Documentation creation completed for: {github_url}")

            # Update job with the results
            job_store[job_id]["docs"] = generated_docs
            job_store[job_id]["output_format"] = output_format

            job_store[job_id]["status"] = "completed"
            job_store[job_id][
                "message"
            ] = f"Documentation generation completed successfully. Generated {len(generated_docs)} {output_format.upper()} files."
            api_logger.info(f"Job {job_id} marked as completed")

        except Exception as e:
            api_logger.error(f"Error in documentation planning: {e}")
            job_store[job_id]["status"] = "failed"
            job_store[job_id]["message"] = f"Error in documentation planning: {str(e)}"
            api_logger.info(f"Job {job_id} marked as failed")
            raise

    except Exception as e:
        error_msg = f"Error in background task: {str(e)}"
        api_logger.error(error_msg)
        api_logger.error(traceback.format_exc())

        job_store[job_id]["status"] = "failed"
        job_store[job_id]["message"] = f"Error: {str(e)}"
        api_logger.info(f"Job {job_id} marked as failed")


@app.post("/generate", response_model=DocumentationResponse)
async def generate_documentation(
    request: DocumentationRequest, background_tasks: BackgroundTasks
):
    """
    Generate documentation for a GitHub repository.
    Returns a job ID that can be used to check the status of the job.
    """
    # SusanD
    api_logger.info(" ")
    api_logger.info(
        "**********************************************************************************"
    )
    api_logger.info(
        "*       This is an NVIDIA NIM & Agentic AI Powered App Development Work          *"
    )
    api_logger.info(
        "*                                                                                *"
    )
    api_logger.info(
        "* --->  Processing Started - Agentic AI Based Code Documentation Generator  <--- *"
    )
    api_logger.info(
        "*                                                                                *"
    )
    api_logger.info(
        "**********************************************************************************"
    )
    api_logger.info(" ")
    # SusanD

    job_id = str(uuid.uuid4())

    job_store[job_id] = {
        "status": "queued",
        "message": "Job queued for processing",
        "github_url": request.github_url,
        "email": request.email,
        "output_format": request.output_format,
        "docs": [],
        "created_at": datetime.datetime.now().isoformat(),
    }

    background_tasks.add_task(
        generate_documentation_background,
        job_id,
        request.github_url,
        request.output_format,
    )

    return DocumentationResponse(
        job_id=job_id,
        status="queued",
        message=f"Documentation generation job queued successfully. Output format: {request.output_format}",
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a documentation generation job.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    job = job_store[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        docs=job.get("docs"),
        plan=job.get("plan"),
        output_format=job.get("output_format"),
    )


@app.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs():
    """
    List all documentation generation jobs.
    """
    return [
        JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            message=job["message"],
            docs=job.get("docs"),
            plan=job.get("plan"),
            output_format=job.get("output_format"),
        )
        for job_id, job in job_store.items()
    ]


@app.get("/")
async def root():
    """
    Root endpoint - provides basic API information
    """
    return {
        "name": "Documentation Generator API",
        "version": "1.0.0",
        "description": "API for automatically generating documentation from GitHub repositories",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "This information"},
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint",
            },
            {
                "path": "/generate",
                "method": "POST",
                "description": "Generate documentation for a GitHub repository",
            },
            {
                "path": "/status/{job_id}",
                "method": "GET",
                "description": "Get the status of a documentation generation job",
            },
            {
                "path": "/jobs",
                "method": "GET",
                "description": "List all documentation generation jobs",
            },
        ],
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check that we can import necessary classes from app.py
        from app import planning_crew, documentation_crew, logger

        api_logger.info("Successfully imported planning and documentation crews")

        # Check that we can create a DocumentationState object
        from app import DocumentationState

        test_state = DocumentationState.create(
            project_url="https://github.com/test/test",
            repo_path="test_path",
            output_path="test_output",
        )
        api_logger.info("Successfully created test state")

        # Check for workdir
        workdir = Path("workdir")
        workdir_exists = workdir.exists()
        if not workdir_exists:
            workdir.mkdir(exist_ok=True)
            api_logger.info("Created workdir directory")
            workdir_exists = True

        # Check for git command
        try:
            git_version = subprocess.run(
                ["git", "--version"], capture_output=True, text=True, check=True
            ).stdout.strip()
            api_logger.info(f"Git is available: {git_version}")
            git_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            api_logger.warning("Git is not available")
            git_available = False

        # Check for NVIDIA API key
        nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "")
        has_api_key = nvidia_api_key.startswith("nvapi-")

        return {
            "status": "healthy",
            "checks": {
                "imports": True,
                "state_init": True,
                "workdir_exists": workdir_exists,
                "git_available": git_available,
                "has_api_key": has_api_key,
            },
        }
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        api_logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "API server is not functioning correctly",
            },
        )


# Run the API when executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
