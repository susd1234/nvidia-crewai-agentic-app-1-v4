import getpass
import os
from dotenv import load_dotenv, find_dotenv
import datetime
import logging
import sys
from pathlib import Path
import time
import threading
import yaml
import subprocess
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import uuid
import json

from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, listen, start
import nest_asyncio

import functools
import inspect
import litellm

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


for handler in logging.root.handlers:
    handler.addFilter(NoLiteLLMFilter())

# Create our logger
logger = logging.getLogger("documentation_agent")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.addFilter(NoLiteLLMFilter())
logger.addHandler(console_handler)

# Override default litellm logger behavior by adding a null handler
litellm_logger = logging.getLogger("litellm")
litellm_logger.setLevel(logging.CRITICAL)
litellm_logger.addHandler(logging.NullHandler())
litellm_logger.propagate = False

# Disable propagation for LiteLLM loggers
for logger_name in [
    "litellm",
    "LiteLLM",
    "litellm.cost_calculator",
    "litellm.llms",
    "litellm.litellm",
    "litellm.utils",
]:
    logger_obj = logging.getLogger(logger_name)
    logger_obj.propagate = False

# Now that logger is defined, we can patch litellm
original_completion = litellm.completion


@functools.wraps(original_completion)
def logged_completion(*args, **kwargs):
    model = kwargs.get("model", "unknown")
    custom_logger = logging.getLogger("documentation_agent")
    custom_logger.info(f"🤖 LLM Call: {model}")
    return original_completion(*args, **kwargs)


litellm.completion = logged_completion

# Monkeypatch LiteLLM cost calculator logging
try:
    # Check if the attribute exists before trying to patch it
    if hasattr(litellm.cost_calculator, "logger"):
        original_logger = litellm.cost_calculator.logger

        # Create a custom logger that doesn't log selected model name messages
        class CustomLogger:
            def __init__(self, original_logger):
                self.original_logger = original_logger

            def __getattr__(self, name):
                orig_attr = getattr(self.original_logger, name)
                if callable(orig_attr):
                    # For methods like info, debug, warning, etc.
                    @functools.wraps(orig_attr)
                    def wrapper(*args, **kwargs):
                        # Skip logging for specific messages
                        if args and isinstance(args[0], str):
                            if "selected model name for cost calculation" in args[0]:
                                return None  # Don't log this message
                        return orig_attr(*args, **kwargs)

                    return wrapper
                return orig_attr

        # Replace the original logger with our filtered version
        litellm.cost_calculator.logger = CustomLogger(original_logger)

        # Also set the logging level to CRITICAL for good measure
        original_logger.setLevel(logging.CRITICAL)

        logger.info("Successfully patched LiteLLM cost calculator logging")
    else:
        logger.info("litellm.cost_calculator.logger not found, skipping patch")
except Exception as e:
    logger.warning(f"Could not patch LiteLLM cost calculator logging: {e}")


class AgentLoggerCallback:
    def __init__(self, logger):
        self.logger = logger

    def on_agent_start(self, agent):
        self.logger.info(f"🧠 Agent starting: {agent.name}")

    def on_agent_end(self, agent, result):
        self.logger.info(f"✅ Agent completed: {agent.name}")


# Create an agent logger callback instance
agent_logger = AgentLoggerCallback(logger)


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


system_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{{ .System }}<|eot_id|>"""
prompt_template = """<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>"""
response_template = (
    """<|start_header_id|>assistant<|end_header_id|>{{ .Response }}<|eot_id|>"""
)

from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
)

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

from crewai.tasks import TaskOutput
import re


def check_mermaid_syntax(task_output: TaskOutput):
    text = task_output.raw
    logger.info("Checking Mermaid syntax in documentation output")

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
from typing import List, Optional, Any, Dict, Union


class DocumentationState(BaseModel):
    """
    State for the documentation flow
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_url: str
    repo_path: str
    docs: List[str] = []
    output_path: str

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
    }

    @classmethod
    def create(cls, project_url: str, repo_path: str, output_path: str):
        """Factory method to create a properly initialized state"""
        return cls(
            project_url=project_url,
            repo_path=repo_path,
            output_path=output_path,
            docs=[],
        )


class CreateDocumentationFlow(Flow[DocumentationState]):
    """Flow for generating documentation from a GitHub repository"""

    def __init__(self):
        """Initialize the flow without state"""
        logger.info("Initializing CreateDocumentationFlow")
        super().__init__()
        logger.info(f"Flow initialized with state type: {type(self.state)}")

    def _create_initial_state(self) -> DocumentationState:
        """Create an initial state with default values"""
        # This method is called by the Flow base class
        logger.info("Creating initial state for CreateDocumentationFlow")
        initial_state = DocumentationState(project_url="", repo_path="", output_path="")
        logger.info(f"Initial state created with id: {initial_state.id}")
        return initial_state

    def kickoff(self, state: Optional[DocumentationState] = None) -> Any:
        """
        Start the flow with a given state

        Args:
            state: State to use for the flow execution
        """
        logger.info(f"Kickoff called with state: {state is not None}")

        if state:
            # Manually set the state attributes
            logger.info("Setting state for flow execution")
            logger.info(f"Current state id: {self.state.id}")
            logger.info(f"Provided state id: {state.id}")

            # Copy all attributes from the provided state to the flow's state
            current_state_data = self.state.model_dump()
            provided_state_data = state.model_dump()

            logger.info(f"Current state keys: {', '.join(current_state_data.keys())}")
            logger.info(f"Provided state keys: {', '.join(provided_state_data.keys())}")

            for key, value in provided_state_data.items():
                if key != "id":  # preserve the id from initial state
                    logger.info(f"Setting state.{key} = {value}")
                    setattr(self.state, key, value)

            logger.info(f"Flow state set: project_url={self.state.project_url}")
        else:
            logger.error("No state provided to kickoff. Cannot continue.")
            raise ValueError("State is required to run the documentation flow")

        logger.info("Starting flow execution with kickoff")
        result = super().kickoff()
        logger.info("Flow execution completed")
        return result

    @start()
    def clone_repo(self):
        logger.info(f"Cloning repository: {self.state.project_url}")
        logger.info(f"State at clone_repo: {self.state.model_dump()}")

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


# When running as a standalone script
if __name__ == "__main__":
    # Get repository URL from user input
    project_url = input(
        "Please enter the GitHub repository URL (e.g., https://github.com/username/repository): "
    ).strip()

    # Set up execution paths
    repo_name = project_url.split("/")[-1]
    timestamp = datetime.datetime.now().strftime("%d%b%y_%H%M%S")
    execution_folder = f"{repo_name}_{timestamp}"

    workdir_path = Path("workdir")
    workdir_path.mkdir(exist_ok=True)

    execution_path = workdir_path / execution_folder
    execution_path.mkdir(exist_ok=True)

    input_path = execution_path / "input"
    input_path.mkdir(exist_ok=True)

    output_path = execution_path / "output"
    output_path.mkdir(exist_ok=True)

    # Create flow instance
    flow = CreateDocumentationFlow()

    # Initialize with state
    state = DocumentationState(
        project_url=project_url, repo_path=str(input_path), output_path=str(output_path)
    )

    # Plot the flow (optional)
    flow.plot()

    # Display flow visualization
    try:
        from IPython.display import IFrame

        IFrame(src="./crewai_flow.html", width="100%", height=400)
    except ImportError:
        pass

    # Run the documentation flow
    logger.info("Starting documentation flow")
    try:
        flow.kickoff(state=state)
        logger.info("Documentation flow completed successfully")
    except Exception as e:
        logger.error(f"Error during documentation flow execution: {e}", exc_info=True)
        raise

    # Display results
    docs_dir = output_path
    logger.info("Documentation files generated:")
    doc_files = list(docs_dir.glob("*.mdx"))
    for doc_file in doc_files:
        logger.info(f"- {doc_file}")

    if doc_files:
        logger.info("Displaying contents of first doc")
        first_doc = Path(doc_files[0]).read_text()
        try:
            from IPython.display import Markdown, display

            display(Markdown(first_doc))
        except (ImportError, NameError):
            logger.info(f"First document available at: {doc_files[0]}")
    else:
        logger.warning("No documentation files were generated.")
