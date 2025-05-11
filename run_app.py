#!/usr/bin/env python3

import subprocess
import sys
import time
import os
import signal
import threading
import webbrowser
from pathlib import Path


def run_api_server():
    """Run the API server in a separate process"""
    print("Starting API server...")
    api_process = subprocess.Popen(
        [sys.executable, "run_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for the API server to start
    time.sleep(3)

    return api_process


def run_app_ui():
    """Run the Gradio UI in a separate process"""
    print("Starting Gradio UI...")
    ui_process = subprocess.Popen(
        [sys.executable, "app_ui.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return ui_process


def monitor_process(process, name):
    """Monitor a process and print its output"""
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(f"[{name}] {output.strip()}")

    # Check for errors
    for line in process.stderr:
        print(f"[{name} ERROR] {line.strip()}")


def main():
    # Create workdir if it doesn't exist
    workdir = Path("workdir")
    workdir.mkdir(exist_ok=True)

    # Start the API server
    api_process = run_api_server()

    # Start the Gradio UI
    ui_process = run_app_ui()

    # Start monitoring threads
    api_monitor = threading.Thread(
        target=monitor_process, args=(api_process, "API"), daemon=True
    )
    ui_monitor = threading.Thread(
        target=monitor_process, args=(ui_process, "UI"), daemon=True
    )

    api_monitor.start()
    ui_monitor.start()

    # Open the browser after a short delay
    def open_browser():
        time.sleep(5)  # Wait for the UI to start
        webbrowser.open("http://localhost:7860")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)

            # Check if either process has terminated
            if api_process.poll() is not None:
                print("API server has terminated unexpectedly")
                break
            if ui_process.poll() is not None:
                print("Gradio UI has terminated unexpectedly")
                break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up processes
        for process in [api_process, ui_process]:
            if process.poll() is None:  # If process is still running
                if os.name == "nt":  # Windows
                    process.terminate()
                else:  # Unix-like
                    os.kill(process.pid, signal.SIGTERM)
                process.wait()


if __name__ == "__main__":
    main()
