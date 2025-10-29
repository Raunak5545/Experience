"""This file contains the prompts for the agent."""

import os
from datetime import datetime

from app.core.config import settings


def load_system_prompt():
    """Load the system prompt from the file."""
    with open(os.path.join(os.path.dirname(__file__), "system.md"), "r") as f:
        return f.read().format(
            agent_name=settings.PROJECT_NAME + " Agent",
            current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

def load_prompt(file_name: str, variables: dict | None = None) -> str:
    """Load a markdown prompt file and inject dynamic variables into it. """
    prompt_path = os.path.join(os.path.dirname(__file__), file_name)
    print(f"values passed to load_prompt: {variables}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Default variables that can be used in all prompts
    base_vars = {
        "current_date_and_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Merge defaults with provided variables
    if variables:
        base_vars.update(**variables)

    # Format the markdown text using the variables
    print("Prompt Variables:", base_vars)
    try:
        formatted_prompt = prompt_template.format(**base_vars)
    except KeyError as e:
        missing_key = e.args[0]
        raise ValueError(f"Missing variable for prompt: '{missing_key}'") from e

    return formatted_prompt
