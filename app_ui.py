import gradio as gr
import requests
import json
import time
from typing import Dict, List, Optional
import os
import threading
import html
import markdown
import re

# API endpoints
API_BASE_URL = "http://localhost:8000"  # Adjust if needed
GENERATE_ENDPOINT = f"{API_BASE_URL}/generate"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"

# Progress stages with emojis for visual indication
PROGRESS_STAGES = [
    ("submitted", "🔄 Job submitted and waiting to start..."),
    ("cloning", "📥 Cloning repository..."),
    ("planning", "🧠 Planning documentation structure..."),
    ("generating", "✍️ Generating documentation..."),
    ("completed", "✅ Documentation generation completed!"),
    ("failed", "❌ Documentation generation failed."),
]

# Tutorial content
TUTORIAL_CONTENT = """
## How to Use This Tool

1. **Enter a GitHub Repository URL**
   - Paste the URL of the GitHub repository you want to document
   - Example: `https://github.com/username/repository`

2. **Click "Generate Documentation"**
   - The system will clone the repository
   - Analyze the codebase
   - Generate comprehensive documentation

3. **Monitor Progress**
   - Watch the progress indicator as documentation is generated
   - The system will show each step of the process

4. **View Results**
   - Once complete, view the documentation plan
   - Explore the generated documentation files
   - Click on any document to view its contents

## Tips for Best Results

- Use public repositories for best results
- Repositories with clear structure and comments work best
- Larger repositories may take longer to process
- Documentation is saved in the `workdir` folder
"""


class JobTracker:
    def __init__(self):
        self.current_job_id = None
        self.status = "idle"
        self.message = ""
        self.plan = None
        self.docs = []
        self.stop_tracking = False
        self.progress_step = 0
        self.repo_url = ""
        self.output_format = "pdf"  # Default to PDF
        self.last_poll_time = 0
        self.poll_interval = 2  # seconds
        self.consecutive_failures = 0
        self.max_failures = 3
        self.last_status_update = None
        self.file_type_icons = {"pdf": "📄", "mdx": "📝", "docx": "📘"}

    def reset(self):
        self.current_job_id = None
        self.status = "idle"
        self.message = ""
        self.plan = None
        self.docs = []
        self.stop_tracking = False
        self.progress_step = 0
        self.repo_url = ""
        self.output_format = "pdf"
        self.last_poll_time = 0
        self.consecutive_failures = 0
        self.last_status_update = None

    def track_job(self, job_id, repo_url, output_format="pdf"):
        self.current_job_id = job_id
        self.status = "submitted"
        self.message = "Job submitted successfully"
        self.stop_tracking = False
        self.progress_step = 0
        self.repo_url = repo_url
        self.output_format = output_format.lower()  # Normalize format
        self.last_poll_time = time.time()
        self.consecutive_failures = 0
        self.last_status_update = time.time()

        # Start tracking in a separate thread
        tracking_thread = threading.Thread(target=self._track_progress)
        tracking_thread.daemon = True
        tracking_thread.start()

    def _track_progress(self):
        while not self.stop_tracking and self.current_job_id:
            try:
                current_time = time.time()
                if current_time - self.last_poll_time >= self.poll_interval:
                    self.last_poll_time = current_time

                    response = requests.get(f"{STATUS_ENDPOINT}/{self.current_job_id}")
                    if response.status_code == 200:
                        data = response.json()
                        old_status = self.status
                        self.status = data.get("status", "unknown")
                        self.message = data.get("message", "")

                        # Log status change for debugging
                        if old_status != self.status:
                            print(f"Job status changed: {old_status} -> {self.status}")
                            self.last_status_update = current_time

                        # Update progress step based on status
                        for i, (stage, _) in enumerate(PROGRESS_STAGES):
                            if stage == self.status:
                                self.progress_step = i
                                break

                        if data.get("plan"):
                            self.plan = data.get("plan")

                        if data.get("docs"):
                            self.docs = data.get("docs")

                        # Reset failure counter on successful request
                        self.consecutive_failures = 0

                        # Only stop tracking if we've reached a terminal state
                        if self.status in ["completed", "failed"]:
                            print(
                                f"Job {self.current_job_id} reached terminal state: {self.status}"
                            )
                            time.sleep(1)  # Wait a moment to ensure UI updates
                            self.stop_tracking = True
                    else:
                        self.consecutive_failures += 1
                        if self.consecutive_failures >= self.max_failures:
                            print(
                                f"Too many consecutive failures ({self.consecutive_failures}), stopping tracking"
                            )
                            self.stop_tracking = True
                        time.sleep(self.poll_interval)  # Wait before retrying
                else:
                    time.sleep(0.1)  # Small sleep to prevent CPU spinning
            except Exception as e:
                print(f"Error in tracking thread: {str(e)}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    self.stop_tracking = True
                time.sleep(self.poll_interval)

    def get_file_type_icon(self):
        return self.file_type_icons.get(self.output_format, "📄")

    def get_status_age(self):
        if not self.last_status_update:
            return "N/A"
        age = time.time() - self.last_status_update
        if age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age/60)}m ago"
        else:
            return f"{int(age/3600)}h ago"


job_tracker = JobTracker()


def render_markdown(md_text):
    """Render markdown to HTML with syntax highlighting"""
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_text,
        extensions=[
            "markdown.extensions.fenced_code",
            "markdown.extensions.tables",
            "markdown.extensions.codehilite",
        ],
    )

    # Add styling for code blocks
    html_content = f"""
    <div class="markdown-content">
        {html_content}
    </div>
    <style>
        .markdown-content code {{
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            margin: 0;
            border-radius: 3px;
            font-family: monospace;
        }}
        .markdown-content pre {{
            background-color: #f6f8fa;
            padding: 16px;
            overflow: auto;
            border-radius: 6px;
            line-height: 1.45;
        }}
        .markdown-content pre code {{
            background-color: transparent;
            padding: 0;
        }}
        .markdown-content h1, .markdown-content h2 {{
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }}
        .markdown-content table {{
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .markdown-content table th, .markdown-content table td {{
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }}
        .markdown-content table tr:nth-child(2n) {{
            background-color: #f6f8fa;
        }}
    </style>
    """

    return html_content


def submit_job(repo_url: str, output_format: str):
    """Submit a documentation generation job to the API"""
    if not repo_url or not repo_url.strip():
        return (
            "Please enter a valid repository URL",
            generate_progress_html(),
            None,
            None,
        )

    try:
        # Reset the tracker
        job_tracker.reset()

        # Submit the job
        response = requests.post(
            GENERATE_ENDPOINT,
            json={"github_url": repo_url, "output_format": output_format},
        )

        if response.status_code == 200:
            data = response.json()
            job_id = data.get("job_id")

            # Start tracking the job
            job_tracker.track_job(job_id, repo_url, output_format)

            # Create a more detailed response message with the job ID
            message = f"""
            <div style="padding: 10px; border-left: 4px solid #4CAF50; background-color: #f9f9f9;">
                <p><strong>Job submitted successfully!</strong></p>
                <p>Job ID: <code>{job_id}</code></p>
                <p>Output Format: <strong>{output_format.upper()}</strong></p>
            </div>
            """

            return (
                message,
                generate_progress_html(),
                None,
                None,
            )
        else:
            return f"Error: {response.text}", generate_progress_html(), None, None
    except Exception as e:
        return f"Error submitting job: {str(e)}", generate_progress_html(), None, None


def generate_progress_html():
    """Generate HTML for the progress animation with enhanced file type support"""
    if job_tracker.status == "idle":
        return None

    file_icon = job_tracker.get_file_type_icon()
    status_age = job_tracker.get_status_age()

    html_content = f"""
    <div class="progress-container">
        <h3 style="margin-bottom: 1rem; color: #1f2937;">
            {file_icon} Processing Repository: {html.escape(job_tracker.repo_url)}
        </h3>
        <div class="job-info" style="background: #f0f7ff; padding: 10px; border-radius: 6px; margin-bottom: 15px;">
            <p style="margin-bottom: 5px;"><strong>Job ID:</strong> <code>{job_tracker.current_job_id}</code></p>
            <p style="margin-bottom: 5px;"><strong>Output Format:</strong> <span style="font-weight: bold;">{job_tracker.output_format.upper()}</span></p>
            <p style="margin-bottom: 1rem;"><strong>Last Update:</strong> <span style="color: #666;">{status_age}</span></p>
        </div>
        <div style="margin: 20px 0;">
    """

    # Add progress steps with enhanced styling
    for i, (stage, description) in enumerate(PROGRESS_STAGES):
        status_class = (
            "completed"
            if i < job_tracker.progress_step
            else "active" if i == job_tracker.progress_step else ""
        )
        icon_class = (
            "completed"
            if i < job_tracker.progress_step
            else "active" if i == job_tracker.progress_step else ""
        )
        icon = (
            "✓"
            if i < job_tracker.progress_step
            else "⟳" if i == job_tracker.progress_step else "○"
        )

        # Add file type specific styling for the generating stage
        stage_style = ""
        if stage == "generating" and i == job_tracker.progress_step:
            stage_style = f"""
                <div style="background: #f8f9fa; padding: 8px; border-radius: 4px; margin-top: 5px;">
                    <span style="color: #666;">Generating {job_tracker.output_format.upper()} documentation...</span>
                </div>
            """

        html_content += f"""
        <div class="progress-step {status_class}" style="margin-bottom: 15px; padding: 10px; border-radius: 6px; background: {'#f0f7ff' if i == job_tracker.progress_step else 'transparent'};">
            <div class="progress-icon {icon_class}" style="display: inline-block; width: 24px; height: 24px; line-height: 24px; text-align: center; border-radius: 50%; background: {'#4CAF50' if i < job_tracker.progress_step else '#2196F3' if i == job_tracker.progress_step else '#e0e0e0'}; color: white; margin-right: 10px;">{icon}</div>
            <div style="flex-grow: 1; display: inline-block;">
                <div style="font-weight: bold; color: #1f2937;">{description}</div>
                {f'<div style="color: #666; font-size: 0.9em; margin-top: 5px;">{job_tracker.message}</div>' if i == job_tracker.progress_step else ''}
                {stage_style if i == job_tracker.progress_step else ''}
            </div>
        </div>
        """

    html_content += """
        </div>
    </div>
    """

    return html_content


def detect_file_type(file_path):
    """Detect the file type based on extension or content"""
    extension = os.path.splitext(file_path)[1].lower()

    if extension in [".md", ".markdown"]:
        return "markdown"
    elif extension in [".html", ".htm"]:
        return "html"
    elif extension in [".txt"]:
        return "text"
    elif extension in [".json"]:
        return "json"
    elif extension in [".py", ".js", ".java", ".c", ".cpp", ".cs", ".rb", ".go", ".ts"]:
        return "code"
    else:
        # Try to detect by reading a bit of the file
        try:
            with open(file_path, "r") as f:
                content = f.read(1000)  # Read first 1000 chars

                if content.startswith("<!DOCTYPE html>") or "<html" in content:
                    return "html"
                elif "```" in content or "#" in content.split("\n")[0]:
                    return "markdown"
                elif "{" in content and "}" in content and ":" in content:
                    return "json"
        except:
            pass

    return "text"  # Default to text


def get_status_updates():
    """Get updates about the running job"""
    status_msg = f"Status: {job_tracker.status.upper()}"
    if job_tracker.message:
        status_msg += f"\nMessage: {job_tracker.message}"

    # Generate progress animation
    progress_html = generate_progress_html()

    # Format the plan if available
    plan_html = None
    if job_tracker.plan:
        try:
            plan_data = json.loads(job_tracker.plan)
            plan_html = f"<h3>Documentation Plan</h3>"
            plan_html += (
                f"<p><strong>Overview:</strong> {plan_data.get('overview', 'N/A')}</p>"
            )

            docs = plan_data.get("docs", [])
            if docs:
                plan_html += "<h4>Documentation Items:</h4>"
                for i, doc in enumerate(docs):
                    plan_html += f"<div style='margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
                    plan_html += (
                        f"<p><strong>Title:</strong> {doc.get('title', 'N/A')}</p>"
                    )
                    plan_html += f"<p><strong>Description:</strong> {doc.get('description', 'N/A')}</p>"
                    plan_html += (
                        f"<p><strong>Goal:</strong> {doc.get('goal', 'N/A')}</p>"
                    )
                    plan_html += "</div>"
        except:
            plan_html = f"<p>Raw plan data: {job_tracker.plan}</p>"

    # Format the docs if available
    docs_html = None
    if job_tracker.docs:
        docs_html = f"""
        <h3>Generated Documentation</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
        """

        for i, doc_path in enumerate(job_tracker.docs):
            doc_name = os.path.basename(doc_path)
            file_type = detect_file_type(doc_path)

            # Create a card for each document
            docs_html += f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <div style="padding: 15px; background-color: #f8f9fa; border-bottom: 1px solid #ddd;">
                    <strong>{doc_name}</strong>
                </div>
                <div style="padding: 15px;">
            """

            # Try to read the document content
            try:
                with open(doc_path, "r") as f:
                    content = f.read()

                if file_type == "markdown":
                    # Render markdown content
                    rendered_content = render_markdown(content)
                    docs_html += f"""
                    <details>
                        <summary style="cursor: pointer; padding: 5px; background-color: #f3f4f6; border-radius: 4px;">View Content</summary>
                        <div style="margin-top: 10px; max-height: 400px; overflow-y: auto;">
                            {rendered_content}
                        </div>
                    </details>
                    """
                elif file_type == "html":
                    docs_html += f"""
                    <details>
                        <summary style="cursor: pointer; padding: 5px; background-color: #f3f4f6; border-radius: 4px;">View Content</summary>
                        <iframe srcdoc="{html.escape(content)}" style="width: 100%; height: 400px; border: none; margin-top: 10px;"></iframe>
                    </details>
                    """
                elif file_type == "json":
                    try:
                        # Pretty print JSON
                        json_content = json.loads(content)
                        formatted_json = json.dumps(json_content, indent=2)
                        docs_html += f"""
                        <details>
                            <summary style="cursor: pointer; padding: 5px; background-color: #f3f4f6; border-radius: 4px;">View Content</summary>
                            <pre style="margin-top: 10px; max-height: 400px; overflow-y: auto; background-color: #f6f8fa; padding: 10px; border-radius: 5px;">{html.escape(formatted_json)}</pre>
                        </details>
                        """
                    except:
                        docs_html += f"""
                        <details>
                            <summary style="cursor: pointer; padding: 5px; background-color: #f3f4f6; border-radius: 4px;">View Content</summary>
                            <pre style="margin-top: 10px; max-height: 400px; overflow-y: auto; background-color: #f6f8fa; padding: 10px; border-radius: 5px;">{html.escape(content)}</pre>
                        </details>
                        """
                else:
                    docs_html += f"""
                    <details>
                        <summary style="cursor: pointer; padding: 5px; background-color: #f3f4f6; border-radius: 4px;">View Content</summary>
                        <pre style="margin-top: 10px; max-height: 400px; overflow-y: auto; background-color: #f6f8fa; padding: 10px; border-radius: 5px;">{html.escape(content)}</pre>
                    </details>
                    """
            except Exception as e:
                docs_html += f"<p>Error loading file: {str(e)}</p>"

            docs_html += """
                </div>
            </div>
            """

        docs_html += "</div>"

    return status_msg, progress_html, plan_html, docs_html


def poll_status():
    # Return the latest status, progress, plan, docs
    return (
        f"Job status: {job_tracker.status} - {job_tracker.message}",
        generate_progress_html(),
        job_tracker.plan,
        job_tracker.docs,
    )


def refresh_status():
    """Manual refresh of job status with enhanced file type handling and correct UI output types"""
    job_id = job_tracker.current_job_id

    if not job_id:
        return (
            "No active job to refresh. Please submit a new job.",
            None,
            None,
            None,
        )

    try:
        print(f"Manually refreshing status for job: {job_id}")
        # Always make a direct API call to get fresh status
        response = requests.get(f"{STATUS_ENDPOINT}/{job_id}")

        if response.status_code == 200:
            # Get fresh data from API
            data = response.json()
            current_status = data.get("status", "unknown")
            current_message = data.get("message", "")
            current_plan = data.get("plan")
            current_docs = data.get("docs", [])

            # Update the job tracker with latest data
            job_tracker.status = current_status
            job_tracker.message = current_message
            job_tracker.last_status_update = time.time()

            # Important: Update progress step based on freshly retrieved status
            for i, (stage, _) in enumerate(PROGRESS_STAGES):
                if stage == current_status:
                    job_tracker.progress_step = i
                    break

            # Parse plan as JSON for Gradio JSON output
            if current_plan:
                try:
                    plan_json = json.loads(current_plan)
                except Exception:
                    plan_json = {}
                job_tracker.plan = plan_json
            else:
                plan_json = {}
                job_tracker.plan = plan_json

            # Only keep files with the right extension for docs (case-insensitive, mdx accepts .mdx and .md)
            if current_docs:
                if job_tracker.output_format == "mdx":
                    docs_files = [
                        f
                        for f in current_docs
                        if f.lower().endswith(".mdx") or f.lower().endswith(".md")
                    ]
                else:
                    docs_files = [
                        f
                        for f in current_docs
                        if f.lower().endswith(f".{job_tracker.output_format}")
                    ]
            else:
                docs_files = []
            print("Filtered docs for UI:", docs_files)
            job_tracker.docs = docs_files

            print(f"Manual refresh complete. Current status: {current_status}")

            # Reset the stop_tracking flag if it was set incorrectly
            if current_status not in ["completed", "failed"]:
                job_tracker.stop_tracking = False
            elif (
                current_status in ["completed", "failed"]
                and not job_tracker.stop_tracking
            ):
                # If job is completed/failed but we're still tracking, stop tracking
                job_tracker.stop_tracking = True
                print(f"Job {job_id} has {current_status}. Stopping tracking.")

            # Build response for UI with file type specific message
            status_message = f"""
            <div style="padding: 10px; border-left: 4px solid {'#4CAF50' if current_status == 'completed' else '#f44336' if current_status == 'failed' else '#2196F3'}; background-color: #f9f9f9;">
                <p><strong>Job Status:</strong> {current_status.upper()}</p>
                <p><strong>Output Format:</strong> {job_tracker.output_format.upper()}</p>
                <p><strong>Last Update:</strong> {job_tracker.get_status_age()}</p>
                <p>{current_message}</p>
            </div>
            """

            return (
                status_message,
                generate_progress_html(),
                plan_json,
                docs_files,
            )
        elif response.status_code == 404:
            # Job not found - reset the tracker and show a helpful message
            job_tracker.reset()
            return (
                "The previous job is no longer available. Please submit a new job.",
                None,
                None,
                None,
            )
        else:
            error_msg = (
                f"Error refreshing job status: {response.status_code} - {response.text}"
            )
            print(error_msg)
            return (error_msg, generate_progress_html(), None, None)
    except Exception as e:
        error_msg = f"Error refreshing status: {str(e)}"
        print(f"Error in manual refresh: {str(e)}")
        return (
            error_msg,
            generate_progress_html(),
            None,
            None,
        )


def load_specific_job():
    """Load a specific job (d868ddf8-485f-42ef-a7f3-d7acdcf4a5c1) for display"""
    specific_job_id = "d868ddf8-485f-42ef-a7f3-d7acdcf4a5c1"
    try:
        # Check if API is available first
        try:
            health_check = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if health_check.status_code != 200:
                return (
                    "API server appears to be unavailable. Please ensure the API is running.",
                    None,
                    None,
                    None,
                )
        except requests.exceptions.RequestException:
            return (
                "Unable to connect to API server. Please ensure the API is running at "
                + API_BASE_URL,
                None,
                None,
                None,
            )

        response = requests.get(f"{STATUS_ENDPOINT}/{specific_job_id}")
        if response.status_code == 200:
            data = response.json()
            job_tracker.current_job_id = specific_job_id
            job_tracker.status = data.get("status", "unknown")
            job_tracker.message = data.get("message", "")
            job_tracker.repo_url = data.get("github_url", "Unknown repository")
            job_tracker.output_format = data.get("output_format", "mdx")

            # Update progress step based on status
            for i, (stage, _) in enumerate(PROGRESS_STAGES):
                if stage == job_tracker.status:
                    job_tracker.progress_step = i
                    break

            if data.get("plan"):
                job_tracker.plan = data.get("plan")

            if data.get("docs"):
                job_tracker.docs = data.get("docs")

            return (
                f"Loaded job: {specific_job_id} - Status: {job_tracker.status}",
                generate_progress_html(),
                job_tracker.plan,
                job_tracker.docs,
            )
        else:
            # Job not found - provide a helpful message instead of an error
            return (
                "Welcome to the NVIDIA NIM Documentation Generator. Enter a GitHub repository URL and select an output format to generate documentation.",
                None,
                None,
                None,
            )
    except Exception as e:
        return (
            f"Error connecting to the API: {str(e)}. Please ensure the API server is running.",
            None,
            None,
            None,
        )


# Create a function to check if we should load the job on startup
def should_auto_load_job():
    """Check if we should auto-load the job on startup"""
    welcome_msg = "Welcome to the NVIDIA NIM Documentation Generator. Enter a GitHub repository URL and select an output format to generate documentation."

    # Check if API is available
    try:
        health_check = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_check.status_code != 200:
            return welcome_msg, None, None, None

        # Try to check if job exists before auto-loading
        specific_job_id = "d868ddf8-485f-42ef-a7f3-d7acdcf4a5c1"
        response = requests.get(f"{STATUS_ENDPOINT}/{specific_job_id}")
        if response.status_code == 200:
            # Job exists, load it
            return load_specific_job()
        else:
            # Job doesn't exist, just show welcome message
            return welcome_msg, None, None, None
    except requests.exceptions.RequestException:
        # API not available, show welcome message
        return welcome_msg, None, None, None


def auto_refresh_status():
    """Automatically refresh status if a job is running"""
    # Only refresh if we have an active job
    if job_tracker.current_job_id and not job_tracker.stop_tracking:
        print(f"Auto-refreshing status for job: {job_tracker.current_job_id}")
        return refresh_status()
    return None, None, None, None


def create_ui():
    with gr.Blocks(
        title="NVIDIA NIM Powered Agentic AI Based Code Documentation Generator App",
        theme=gr.themes.Soft(),
    ) as app:
        # Add custom CSS for enhanced styling
        gr.HTML(
            """
        <style>
            .gradient-header {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                position: relative;
                overflow: hidden;
            }
            .gradient-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
                opacity: 0.5;
            }
            .gradient-header h1 {
                color: white;
                font-size: 2.5rem;
                margin: 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            .gradient-header h1::before {
                content: '📚';
                font-size: 3rem;
                animation: float 3s ease-in-out infinite;
            }
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
                100% { transform: translateY(0px); }
            }
            .progress-container {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                margin: 2rem 0;
            }
            .progress-step {
                display: flex;
                align-items: center;
                margin-bottom: 1.5rem;
                padding: 1rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            .progress-step:hover {
                background: #f8f9fa;
                transform: translateX(5px);
            }
            .progress-step.active {
                background: #f0f7ff;
                border-left: 4px solid #2196F3;
            }
            .progress-step.completed {
                background: #f0fff4;
                border-left: 4px solid #4CAF50;
            }
            .progress-icon {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 1rem;
                font-size: 1.2rem;
                transition: all 0.3s ease;
            }
            .progress-icon.completed {
                background: #4CAF50;
                color: white;
            }
            .progress-icon.active {
                background: #2196F3;
                color: white;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            .input-container {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                margin-bottom: 2rem;
            }
            .button-primary {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            .button-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
        </style>
        """
        )

        # Modern header with gradient background
        gr.HTML(
            """
        <div class="gradient-header">
            <h1>NVIDIA NIM & Agentic AI Powered Code Documentation Generator App</h1>
        </div>
        """
        )

        # Tutorial content with enhanced styling
        gr.Markdown(TUTORIAL_CONTENT)

        # Input section
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="input-container"):
                    repo_url = gr.Textbox(
                        label="GitHub Repository URL",
                        placeholder="https://github.com/username/repository",
                    )
                    output_format = gr.Radio(
                        choices=["pdf", "mdx", "docx"],
                        value="pdf",
                        label="Output Format",
                        info="Select the format for generated documentation files",
                    )
                    generate_btn = gr.Button(
                        "Generate Documentation",
                        variant="primary",
                        elem_classes="button-primary",
                    )
                    refresh_btn = gr.Button(
                        "🔄 Refresh Status (Click periodically to update)",
                        elem_classes="button-primary",
                        variant="primary",
                    )

        # Output section
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes="progress-container"):
                    status_output = gr.HTML(label="Status")
                    progress_output = gr.HTML(label="Progress")
        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes="progress-container"):
                    plan_output = gr.JSON(label="Documentation Plan")
                    docs_output = gr.File(label="Generated Documentation")

        # Set up event handler for the generate button
        generate_btn.click(
            fn=submit_job,
            inputs=[repo_url, output_format],
            outputs=[status_output, progress_output, plan_output, docs_output],
        )

        # Manual refresh button with more emphasis
        refresh_btn.click(
            fn=refresh_status,
            inputs=[],
            outputs=[status_output, progress_output, plan_output, docs_output],
        )

        # Simple welcome message
        app.load(
            fn=lambda: (
                "Welcome to the NVIDIA NIM Documentation Generator. Enter a GitHub repository URL to start.",
                None,
                None,
                None,
            ),
            inputs=[],
            outputs=[status_output, progress_output, plan_output, docs_output],
        )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
