import os
import time
from typing import (
    Any,
    Dict,
    Optional,
)

from fastapi import HTTPException
from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.logging import logger
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.langgraph.config.model_config import workflow_config
from app.core.prompts import load_prompt


class ExtractionAgent:
    """Agent that extracts travel information from text or multimodal input."""

    def __init__(self):
        # Get the model configuration for this node
        model_config = workflow_config.get_config("extraction")
        
        # Initialize LLM with the configuration
        self.text_llm = ChatGoogleGenerativeAI(
            model=model_config.model_name,
            google_api_key=settings.LLM_API_KEY,
            **model_config.to_dict()
        )

        # Initialize multimodal client with configuration
        self.multimodal_client = genai.Client(
            api_key=settings.LLM_API_KEY,
            **(model_config.multimodal.to_dict() if model_config.multimodal else {})
        )
        # Load extraction prompt template
        self.prompt = load_prompt("extraction.md", {"extra_instructions": ""})

    def extract_from_text(self, text: str) -> str:
        prompt = self.prompt + f"\n\n{text}"
        response = self.text_llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def extract_from_input(
        self,
        state,
        file_input: str,
        extra_prompt: Optional[str] = None,
    ) -> str:
        """
        Extract information from either a file path or URL.
        Args:
            file_input: Path to file or URL
            extra_prompt: Additional prompt text
            is_url: Whether the input is a URL
        """
        from app.utils.file_handler import prepare_content_message

        task_prompt = (
            extra_prompt or "Extract key travel information (dates, destinations, travelers, etc.) from this file."
        )
        full_prompt = load_prompt("extraction.md", {"extra_instructions": task_prompt})
        is_url = state.get("is_url")
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="extraction")
        if is_url:
            content = prepare_content_message(
                full_prompt,
                file_input,
            )
            response = self.text_llm.invoke(
                [HumanMessage(content=content)],
                config={
                    "callbacks": [langfuse_handler],
                    "langfuse_session_id": state.get("session_id"),
                },
            )
            bound_logger.info("extraction_url_invoked", url=file_input)
            response.response_metadata
            return response.content, None

        # For file uploads
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"File not found: {file_input}")

        start_time = time.time()
        uploaded_file = self.multimodal_client.files.upload(file=file_input)
        while True:
            current_file = self.multimodal_client.files.get(name=uploaded_file.name)
            bound_logger.debug("file_upload_state", file_name=uploaded_file.name, state=current_file.state)
            if current_file.state == "ACTIVE":
                break
            elif current_file.state == "FAILED":
                bound_logger.error("file_processing_failed", file_name=uploaded_file.name)
                raise RuntimeError("File processing failed.")
            curr_time = time.time()
            diff_time = curr_time - start_time
            if diff_time > 180:
                bound_logger.error("file_upload_timeout", elapsed=diff_time)
                raise HTTPException(status_code=502, detail="Timed out while trying to upload the file.")
            time.sleep(2)
        response = self.multimodal_client.models.generate_content(
            model=settings.EXTRACTION_MODEL, contents=[uploaded_file, full_prompt]
        )
        return (response.text, uploaded_file)

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:

        raw_input: Optional[str] = state.get("raw_input")
        file_path: Optional[str] = state.get("input_file_path")
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="extraction")
        bound_logger.info("extraction_execute_start", has_file=bool(file_path), has_raw_input=bool(raw_input))

        if file_path:
            extracted_text, uploaded_file = self.extract_from_input(
                state,
                file_path,
                state.get("validation_prompt"),
            )

        elif raw_input:
            extracted_text = self.extract_from_text(raw_input)
        else:
            extracted_text = "No input provided."
        bound_logger.info("extraction_execute_complete", extracted_text_length=len(extracted_text) if isinstance(extracted_text, str) else None)
        return {
            "extracted_text": extracted_text,
            "input_file": uploaded_file,
        }
