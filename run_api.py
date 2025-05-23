#!/usr/bin/env python3

import uvicorn
from dotenv import load_dotenv, find_dotenv
import logging
import os
import sys
import getpass
import traceback
import nest_asyncio


# Custom filter to remove LiteLLM logs
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
        if "change detected" in record.getMessage():
            return False
        return True


# Set up logging
try:
    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("api_server.log"),
        ],
    )

    # Set higher logging levels for LiteLLM related loggers
    for logger_name in [
        "litellm",
        "LiteLLM",
        "litellm.cost_calculator",
        "litellm.llms",
        "litellm.litellm",
        "litellm.utils",
        "httpx",
        "httpcore",
        "openai",
        "watchdog",
        "watchdog.observers",
        "watchdog.events",
    ]:
        log = logging.getLogger(logger_name)
        log.setLevel(logging.CRITICAL)
        log.propagate = False
        # Remove all handlers
        for handler in log.handlers[:]:
            log.removeHandler(handler)
        # Add null handler
        log.addHandler(logging.NullHandler())

    # Add filter to all root logger handlers
    for handler in logging.root.handlers:
        handler.addFilter(NoLiteLLMFilter())

    # Create logger for API server
    logger = logging.getLogger("api_server")
    logger.setLevel(logging.INFO)

    logger.info("Logging configuration completed successfully")
except Exception as e:
    # Fallback logging setup if the configuration fails
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("api_server")
    logger.error(f"Error setting up logging: {e}")
    logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        logger.info("Starting Documentation Generator API Server...")

        # Apply nest_asyncio to allow nested async loops
        logger.info("Applying nest_asyncio patch...")
        nest_asyncio.apply()

        # Load environment variables
        logger.info("Loading environment variables...")
        _ = load_dotenv(find_dotenv(), override=True)

        # Check for NVIDIA API key
        if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
            logger.info("NVIDIA API key not found in environment variables")
            try:
                nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
                assert nvapi_key.startswith(
                    "nvapi-"
                ), f"{nvapi_key[:5]}... is not a valid key"
                os.environ["NVIDIA_NIM_API_KEY"] = nvapi_key
                os.environ["NVIDIA_API_KEY"] = nvapi_key
                logger.info("NVIDIA API key set successfully")
            except Exception as e:
                logger.error(f"Failed to set NVIDIA API key: {e}")
                sys.exit(1)
        # else:
        #     logger.info("Using NVIDIA API key from environment variables")

        # Create workdir if it doesn't exist
        if not os.path.exists("workdir"):
            logger.info("Creating workdir directory")
            os.makedirs("workdir")

        # Set environment variable to disable certain logging
        os.environ["LITELLM_LOG_LEVEL"] = "CRITICAL"

        # Start the API server with appropriate log level
        logger.info("Starting Uvicorn server...")

        # Try different ports if the default is in use
        ports = [8000, 8001, 8002, 8003, 8004]
        server_started = False

        for port in ports:
            try:
                logger.info(f"Trying to start server on port {port}...")

                # Also update the port in app_ui.py if we're using a non-default port
                if port != 8000:
                    with open("app_ui.py", "r") as f:
                        content = f.read()

                    # Update API base URL
                    updated_content = content.replace(
                        'API_BASE_URL = "http://localhost:8000"',
                        f'API_BASE_URL = "http://localhost:{port}"',
                    )

                    with open("app_ui.py", "w") as f:
                        f.write(updated_content)

                    logger.info(f"Updated app_ui.py to use port {port}")

                uvicorn.run(
                    "api:app",
                    host="0.0.0.0",
                    port=port,
                    reload=True,
                    log_level="error",  # Change log level to error to reduce noise
                    access_log=False,  # Disable access logs
                )

                server_started = True
                break
            except OSError as e:
                if "Address already in use" in str(e):
                    logger.warning(
                        f"Port {port} is already in use, trying next port..."
                    )
                else:
                    logger.error(f"Error starting server on port {port}: {e}")
                    break

        if not server_started:
            logger.error("Could not start server on any of the configured ports")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
