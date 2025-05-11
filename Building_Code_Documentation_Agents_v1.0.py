import getpass
import os
from dotenv import load_dotenv, find_dotenv
import datetime


# Load environment variables from .env file
_ = load_dotenv(find_dotenv(), override=True)

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_NIM_API_KEY"] = nvapi_key
    os.environ["NVIDIA_API_KEY"] = nvapi_key

### --> Install Dependencies
# Create reusable loading animation class
import sys
import time
import threading


class LoadingAnimation:
    def __init__(self):
        self.stop_event = threading.Event()
        self.animation_thread = None

    def _animate(self, message="Loading"):
        chars = "/—\\|"
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
        self.animation_thread = threading.Thread(target=self._animate, args=(message,))
        self.animation_thread.daemon = True
        self.animation_thread.start()

    def stop(self, completion_message="Complete"):
        self.stop_event.set()
        if self.animation_thread:
            self.animation_thread.join()
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
from pathlib import Path
from pydantic import BaseModel

# Importing Crew related components
from crewai import Agent, Task, Crew

# Importing CrewAI Flow related components
from crewai.flow.flow import Flow, listen, start

# Apply a patch to allow nested asyncio loops in Jupyter
import nest_asyncio

nest_asyncio.apply()

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

    # Find all mermaid code blocks in the text
    mermaid_blocks = re.findall(r"```mermaid\n(.*?)\n```", text, re.DOTALL)

    for block in mermaid_blocks:
        diagram_text = block.strip()
        lines = diagram_text.split("\n")
        corrected_lines = []

        for line in lines:
            corrected_line = re.sub(
                r"\|.*?\|>", lambda match: match.group(0).replace("|>", "|"), line
            )
            corrected_lines.append(corrected_line)

        text = text.replace(block, "\n".join(corrected_lines))

    task_output.raw = text
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
        print(f"# Cloning repository: {self.state.project_url}\n")

        # Check if directory exists
        if Path(self.state.repo_path).exists() and any(
            Path(self.state.repo_path).iterdir()
        ):
            print(f"# Repository directory already exists at {self.state.repo_path}\n")
            subprocess.run(["rm", "-rf", self.state.repo_path])
            print("# Removed existing directory\n")
            # Recreate the directory
            Path(self.state.repo_path).mkdir(exist_ok=True)

        # Clone the repository
        subprocess.run(["git", "clone", self.state.project_url, self.state.repo_path])
        return self.state

    @listen(clone_repo)
    def plan_docs(self):
        print(f"# Planning documentation for: {self.state.repo_path}\n")
        # Convert Path to string for CrewAI
        repo_path_str = str(self.state.repo_path)
        result = planning_crew.kickoff(inputs={"repo_path": repo_path_str})
        print(f"# Planned docs for {self.state.repo_path}:")
        for doc in result.pydantic.docs:
            print(f"    - {doc.title}")
        return result

    @listen(plan_docs)
    def save_plan(self, plan):
        # Create docs directory if it doesn't exist
        docs_dir = Path(self.state.output_path)
        docs_dir.mkdir(exist_ok=True)

        with open(docs_dir / "plan.json", "w") as f:
            f.write(plan.raw)

    @listen(plan_docs)
    def create_docs(self, plan):
        for doc in plan.pydantic.docs:
            print(f"\n# Creating documentation for: {doc.title}")
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
            self.state.docs.append(str(docs_dir / title))
            with open(docs_dir / title, "w") as f:
                f.write(result.raw)
        print(f"\n# Documentation created for: {self.state.repo_path}")


### --> Implementing helper methods to plot and execute the flow in a Jupyter notebook
# Plot the flow
flow = CreateDocumentationFlow()
flow.plot()

# Display the flow visualization using IFrame
from IPython.display import IFrame

# Display the flow visualization
IFrame(src="./crewai_flow.html", width="100%", height=400)

### --> ' Run Documentation Flow
flow = CreateDocumentationFlow()
flow.kickoff()

### --> Plot One of the Documents
# List all files in docs folder and display the first doc using IPython.display
from IPython.display import Markdown, display
import pathlib

docs_dir = output_path
print("Documentation files generated:")
for doc_file in docs_dir.glob("*.mdx"):
    print(f"- {doc_file}")

print("\nDisplaying contents of first doc:\n")
if flow.state.docs:
    first_doc = Path(flow.state.docs[0]).read_text()
    try:
        display(Markdown(first_doc))
    except NameError:
        # If not running in Jupyter, just print the markdown
        print(first_doc)
else:
    print("No documentation files were generated.")
