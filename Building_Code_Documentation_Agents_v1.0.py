import getpass
import os
from dotenv import load_dotenv, find_dotenv
import datetime
import logging
import sys
from pathlib import Path

# Disable all LiteLLM logging before any imports
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

# # Configure logging
# def setup_logging(execution_folder=None):
#     """Setup logging configuration for the application"""
#     # logger = logging.getLogger("documentation_agent")
#     # logger.setLevel(logging.INFO)

#     # # Create formatters
#     # console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     # # file_formatter = logging.Formatter(
#     # #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s",

#     # # )

#     logging.basicConfig(
#         filename=log_file,
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )

#     # Create console handler
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(console_formatter)

#     # Add handlers to logger
#     logger.addHandler(console_handler)

#     # Add file handler if execution folder is provided
#     if execution_folder:
#         log_dir = Path(execution_folder) / "logs"
#         log_dir.mkdir(exist_ok=True)
#         log_file = log_dir / "documentation_agent.log"

#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(logging.DEBUG)
#         file_handler.setFormatter(file_formatter)
#         logger.addHandler(file_handler)

#     return logger


# Load environment variables from .env file
_ = load_dotenv(find_dotenv(), override=True)

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_NIM_API_KEY"] = nvapi_key
    os.environ["NVIDIA_API_KEY"] = nvapi_key

### --> Install Dependencies
# Create reusable loading animation class
import time
import threading


class LoadingAnimation:
    def __init__(self):
        self.stop_event = threading.Event()
        self.animation_thread = None
        self.message = ""

    def _animate(self, message="Loading"):
        self.message = message
        chars = "/—\\|"
        logger.debug(f"Starting animation: {message}")
        while not self.stop_event.is_set():
            for char in chars:
                sys.stdout.write("\r" + message + "... " + char)
                sys.stdout.flush()
                time.sleep(0.1)
                if self.stop_event.is_set():
                    sys.stdout.write("\n")
                    break

    def start(self, message="Loading"):
        self.stop_event.clear()
        logger.info(f"Operation started: {message}")
        self.animation_thread = threading.Thread(target=self._animate, args=(message,))
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop(self, completion_message="Complete"):
        self.stop_event.set()
        if self.animation_thread:
            self.animation_thread.join()
        logger.info(f"Operation completed: {self.message} → {completion_message}")
        print(f"\r{completion_message} ✓")


# # Use the animation for pip install
# loader = LoadingAnimation()
# loader.start("Installing")
# # %pip install -r requirements.txt -q
# loader.stop("Installation complete")

import dotenv
from dotenv import dotenv_values

### --> Helper Functions


# Define a fake `load_dotenv` function
def _load_dotenv(*args, **kwargs):
    env_path = kwargs.get("dotenv_path", ".env")  # Default to '.env'
    parsed_env = dotenv_values(env_path)

    # Manually set valid key-value pairs
    for key, value in parsed_env.items():
        if key and value:  # Check for valid key-value pairs
            os.environ[key] = value


dotenv.load_dotenv = _load_dotenv

### --> Initialization and Setup
# Importing necessary libraries
import yaml
import subprocess
from pydantic import BaseModel

# Importing Crew related components
from crewai import Agent, Task, Crew

# Importing CrewAI Flow related components
from crewai.flow.flow import Flow, listen, start

# Apply a patch to allow nested asyncio loops in Jupyter
import nest_asyncio

nest_asyncio.apply()

# Set up a custom wrapper to track LLM calls
import functools
import inspect
import litellm

# Store the original completion method
original_completion = litellm.completion


# Patch the litellm.completion method to log when it's called
@functools.wraps(original_completion)
def logged_completion(*args, **kwargs):
    model = kwargs.get("model", "unknown")
    # Get custom app logger
    custom_logger = logging.getLogger("documentation_agent")
    custom_logger.info(f"🤖 LLM Call: {model}")

    # Call the original method
    return original_completion(*args, **kwargs)


# Replace the litellm.completion method with our wrapped version
litellm.completion = logged_completion


# Create a custom callback handler for crewAI Agent activities
class AgentLoggerCallback:
    def __init__(self, logger):
        self.logger = logger

    def on_agent_start(self, agent):
        self.logger.info(f"🧠 Agent starting: {agent.name}")

    def on_agent_end(self, agent, result):
        self.logger.info(f"✅ Agent completed: {agent.name}")


# Will use this callback when creating agents later

# Silence litellm loggers again after all imports
logging.getLogger("litellm").setLevel(logging.CRITICAL)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm.litellm").setLevel(logging.CRITICAL)
logging.getLogger("litellm.utils").setLevel(logging.CRITICAL)
logging.getLogger("litellm.llms.custom").setLevel(logging.CRITICAL)
logging.getLogger("litellm.llms").setLevel(logging.CRITICAL)
logging.getLogger("litellm.cost_calculator").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

### --> Define the Project URL
# project_url = "https://github.com/crewAIInc/nvidia-demo"
# project_url = "https://github.com/Stability-AI/stablediffusion"

# Get GitHub repository URL from user input
project_url = input(
    "Please enter the GitHub repository URL (e.g., https://github.com/username/repository): "
).strip()

# Get repository name from URL for folder naming
repo_name = project_url.split("/")[-1]

# Create a timestamp for the execution
timestamp = datetime.datetime.now().strftime("%d%b%y_%H%M%S")
execution_folder = f"{repo_name}_{timestamp}"

# Create workdir if it doesn't exist
workdir_path = Path("workdir")
workdir_path.mkdir(exist_ok=True)

# Create execution folder
execution_path = workdir_path / execution_folder
execution_path.mkdir(exist_ok=True)

# Create input and output folders
input_path = execution_path / "input"
input_path.mkdir(exist_ok=True)

output_path = execution_path / "output"
output_path.mkdir(exist_ok=True)


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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Apply filter to root logger
for handler in logging.root.handlers:
    handler.addFilter(NoLiteLLMFilter())

# Create and configure our application logger
logger = logging.getLogger("documentation_agent")
logger.setLevel(logging.INFO)

# Add a console handler with the filter
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.addFilter(NoLiteLLMFilter())
logger.addHandler(console_handler)

# Override default litellm logger behavior by adding a null handler
litellm_logger = logging.getLogger("litellm")
litellm_logger.setLevel(logging.CRITICAL)
litellm_logger.addHandler(logging.NullHandler())
litellm_logger.propagate = False

# Create an agent logger callback instance
agent_logger = AgentLoggerCallback(logger)

# logger.info(f"Initialized documentation agent for repository: {project_url}")
# logger.info(f"Created execution folder: {execution_folder}")
# logger.info(f"Input path: {input_path}")
# logger.info(f"Output path: {output_path}")

# # Validate the URL format
# if not project_url.startswith("https://github.com/") or len(project_url.split("/")) < 5:
#     raise ValueError(
#         "Invalid GitHub repository URL format. Please use the format: https://github.com/username/repository"
#     )


### --> Create Planning Crew
# Define data structures to capture documentation planning output
class DocItem(BaseModel):
    """Represents a documentation item"""

    title: str
    description: str
    prerequisites: str
    examples: list[str]
    goal: str


class DocPlan(BaseModel):
    """Documentation plan"""

    overview: str
    docs: list[DocItem]


### --> Optimizing for Llama 3.3 Prompting Template
# Agents Prompting Template for Llama 3.3
system_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>"""
prompt_template = """<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>"""
response_template = (
    """<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>"""
)

### --> Create the Crew
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
)

# Load agent and task configurations from YAML files
with open("config/planner_agents.yaml", "r") as f:
    agents_config = yaml.safe_load(f)

with open("config/planner_tasks.yaml", "r") as f:
    tasks_config = yaml.safe_load(f)

code_explorer = Agent(
    config=agents_config["code_explorer"],
    system_template=system_template,
    prompt_template=prompt_template,
    response_template=response_template,
    tools=[DirectoryReadTool(), FileReadTool()],
)
documentation_planner = Agent(
    config=agents_config["documentation_planner"],
    system_template=system_template,
    prompt_template=prompt_template,
    response_template=response_template,
    tools=[DirectoryReadTool(), FileReadTool()],
)

analyze_codebase = Task(config=tasks_config["analyze_codebase"], agent=code_explorer)
create_documentation_plan = Task(
    config=tasks_config["create_documentation_plan"],
    agent=documentation_planner,
    output_pydantic=DocPlan,
)

planning_crew = Crew(
    agents=[code_explorer, documentation_planner],
    tasks=[analyze_codebase, create_documentation_plan],
    verbose=False,
)

### --> ~ Create Documentation Crew
from crewai.tasks import TaskOutput
import re


def check_mermaid_syntax(task_output: TaskOutput):
    text = task_output.raw
    logger.info("Checking Mermaid syntax in documentation output")

    # Find all mermaid code blocks in the text
    mermaid_blocks = re.findall(r"```mermaid\n(.*?)\n```", text, re.DOTALL)

    if not mermaid_blocks:
        logger.debug("No Mermaid blocks found in the output")
        return (True, task_output)

    logger.info(f"Found {len(mermaid_blocks)} Mermaid blocks for syntax check")

    for i, block in enumerate(mermaid_blocks):
        logger.debug(f"Processing Mermaid block {i+1}")
        diagram_text = block.strip()
        lines = diagram_text.split("\n")
        corrected_lines = []

        for line in lines:
            corrected_line = re.sub(
                r"\|.*?\|>", lambda match: match.group(0).replace("|>", "|"), line
            )
            if corrected_line != line:
                logger.debug(f"Corrected syntax in line: {line} -> {corrected_line}")
            corrected_lines.append(corrected_line)

        text = text.replace(block, "\n".join(corrected_lines))

    task_output.raw = text
    logger.info("Mermaid syntax check completed")
    return (True, task_output)


from crewai_tools import DirectoryReadTool, FileReadTool, WebsiteSearchTool

# Load agent and task configurations from YAML files
with open("config/documentation_agents.yaml", "r") as f:
    agents_config = yaml.safe_load(f)

with open("config/documentation_tasks.yaml", "r") as f:
    tasks_config = yaml.safe_load(f)

overview_writer = Agent(
    config=agents_config["overview_writer"],
    tools=[
        DirectoryReadTool(),
        FileReadTool(),
        WebsiteSearchTool(
            website="https://mermaid.js.org/intro/",
            config=dict(
                embedder=dict(
                    provider="nvidia",
                    config=dict(model="nvidia/nv-embedqa-e5-v5"),
                )
            ),
        ),
    ],
)

documentation_reviewer = Agent(
    config=agents_config["documentation_reviewer"],
    tools=[
        DirectoryReadTool(
            directory="docs/", name="Check existing documentation folder"
        ),
        FileReadTool(),
    ],
)

draft_documentation = Task(
    config=tasks_config["draft_documentation"], agent=overview_writer
)

qa_review_documentation = Task(
    config=tasks_config["qa_review_documentation"],
    agent=documentation_reviewer,
    guardrail=check_mermaid_syntax,
    max_retries=5,
)

documentation_crew = Crew(
    agents=[overview_writer, documentation_reviewer],
    tasks=[draft_documentation, qa_review_documentation],
    verbose=False,
)

### --> Create Documentation Flow
from typing import List


class DocumentationState(BaseModel):
    """
    State for the documentation flow
    """

    project_url: str = project_url
    repo_path: str = str(input_path)
    docs: List[str] = []
    output_path: str = str(output_path)


class CreateDocumentationFlow(Flow[DocumentationState]):
    # Clone the repository, initial step
    # No need for AI Agents on this step, so we just use regular Python code
    @start()
    def clone_repo(self):
        logger.info(f"Cloning repository: {self.state.project_url}")

        # Check if directory exists
        if Path(self.state.repo_path).exists() and any(
            Path(self.state.repo_path).iterdir()
        ):
            logger.info(
                f"Repository directory already exists at {self.state.repo_path}"
            )
            subprocess.run(["rm", "-rf", self.state.repo_path])
            logger.info("Removed existing directory")
            # Recreate the directory
            Path(self.state.repo_path).mkdir(exist_ok=True)

        # Clone the repository
        logger.info("Starting git clone operation")
        try:
            subprocess.run(
                ["git", "clone", self.state.project_url, self.state.repo_path],
                check=True,
            )
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
        return self.state

    @listen(clone_repo)
    def plan_docs(self):
        logger.info(f"Planning documentation for: {self.state.repo_path}")
        # Convert Path to string for CrewAI
        repo_path_str = str(self.state.repo_path)

        try:
            result = planning_crew.kickoff(inputs={"repo_path": repo_path_str})
            logger.info(f"Documentation planning completed")
            for doc in result.pydantic.docs:
                logger.info(f"Planned document: {doc.title}")
            return result
        except Exception as e:
            logger.error(f"Error in documentation planning: {e}")
            raise

    @listen(plan_docs)
    def save_plan(self, plan):
        # Create docs directory if it doesn't exist
        docs_dir = Path(self.state.output_path)
        docs_dir.mkdir(exist_ok=True)

        plan_file = docs_dir / "plan.json"
        logger.info(f"Saving documentation plan to {plan_file}")
        try:
            with open(plan_file, "w") as f:
                f.write(plan.raw)
            logger.info("Documentation plan saved successfully")
        except Exception as e:
            logger.error(f"Failed to save documentation plan: {e}")
            raise

    @listen(plan_docs)
    def create_docs(self, plan):
        logger.info("Starting documentation creation process")
        for doc in plan.pydantic.docs:
            logger.info(f"Creating documentation for: {doc.title}")
            try:
                result = documentation_crew.kickoff(
                    inputs={
                        "repo_path": str(self.state.repo_path),
                        "title": doc.title,
                        "overview": plan.pydantic.overview,
                        "description": doc.description,
                        "prerequisites": doc.prerequisites,
                        "examples": "\n".join(doc.examples),
                        "goal": doc.goal,
                    }
                )

                # Save documentation to file in output folder
                docs_dir = Path(self.state.output_path)
                docs_dir.mkdir(exist_ok=True)
                title = doc.title.lower().replace(" ", "_") + ".mdx"
                doc_file = docs_dir / title
                self.state.docs.append(str(doc_file))

                with open(doc_file, "w") as f:
                    f.write(result.raw)
                logger.info(f"Documentation saved to {doc_file}")
            except Exception as e:
                logger.error(f"Error creating documentation for {doc.title}: {e}")
                continue

        logger.info(f"Documentation creation completed for: {self.state.repo_path}")


### --> Implementing helper methods to plot and execute the flow in a Jupyter notebook
# Plot the flow
flow = CreateDocumentationFlow()
flow.plot()

# Display the flow visualization using IFrame
from IPython.display import IFrame

# Display the flow visualization
IFrame(src="./crewai_flow.html", width="100%", height=400)

### --> ' Run Documentation Flow
logger.info("Starting documentation flow")
try:
    flow = CreateDocumentationFlow()
    flow.kickoff()
    logger.info("Documentation flow completed successfully")
except Exception as e:
    logger.error(f"Error during documentation flow execution: {e}", exc_info=True)
    raise

### --> Plot One of the Documents
# List all files in docs folder and display the first doc using IPython.display
from IPython.display import Markdown, display
import pathlib

docs_dir = output_path
logger.info("Documentation files generated:")
doc_files = list(docs_dir.glob("*.mdx"))
for doc_file in doc_files:
    logger.info(f"- {doc_file}")

if doc_files:
    logger.info("Displaying contents of first doc")
    first_doc = Path(doc_files[0]).read_text()
    try:
        display(Markdown(first_doc))
    except NameError:
        # If not running in Jupyter, just log the path
        logger.info(f"First document available at: {doc_files[0]}")
else:
    logger.warning("No documentation files were generated.")
