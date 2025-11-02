import json
import os
import time
import traceback
from typing import (
    Any,
    Dict,
    Optional,
)

from fastapi import HTTPException
from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import observe

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.langgraph.config.model_config import workflow_config
from app.core.logging import logger
from app.core.prompts import load_prompt
from app.core.langgraph.llm_client import LLMClient
from app.core.langgraph.output_validator import validate_json, repair_json_with_llm


class EvalAgent:
    """Agent that evaluates extracted travel information against the raw input for quality and compliance."""

    def __init__(self):
        # Get the model configuration for this node
        model_config = workflow_config.get_config("evaluation")

        # Initialize LLM with the configuration
        self.eval_llm = ChatGoogleGenerativeAI(
            model=model_config.model_name, google_api_key=settings.LLM_API_KEY, **model_config.to_dict()
        )

        # Initialize multimodal client with configuration
        self.multimodal_client = genai.Client(
            api_key=settings.LLM_API_KEY, **(model_config.multimodal.to_dict() if model_config.multimodal else {})
        )
        # Wrap LLMs with the centralized client for retries/logging
        self.llm_client = LLMClient(llm=self.eval_llm, multimodal_client=self.multimodal_client, name=model_config.model_name)

    def evaluate_from_text(self, state) -> Dict[str, Any]:
        text = state.get("text", "")
        experience = state.get("experience", {})
        session_id = state.get("session_id", "")

        bound_logger = logger.bind(session_id=session_id, node="eval")
        eval_prompt = load_prompt("eval.md", {"text": text, "experience": json.dumps(experience, indent=2)})

        try:
            resp = self.llm_client.invoke([HumanMessage(content=eval_prompt)], session_id=session_id, callbacks=[langfuse_handler])
            content = (resp.get("content") or "").strip()
            parsed = validate_json(content, session_id=session_id)
            if parsed.get("ok"):
                return parsed.get("obj")
            # attempt repair with LLM
            repair = repair_json_with_llm(self.llm_client, content, session_id=session_id, max_attempts=2)
            if repair.get("ok"):
                return repair.get("obj")
            bound_logger.error("eval_from_text_parse_failure", error=parsed.get("error"))
            return self._error_fallback("Model returned invalid JSON for evaluation")
        except Exception as e:
            bound_logger.error("eval_from_text_error", error=str(e), exc_info=True)
            return self._error_fallback(str(e))

    def evaluate_input(self, state) -> Dict[str, Any]:
        """Evaluates input files or URLs along with the extracted experience."""
        from app.utils.file_handler import prepare_content_message

        experience = state.get("experience", {})
        file_input = state.get("input_file_path")
        is_url = state.get("is_url", False)
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="eval")
        eval_prompt = load_prompt("eval.md", {"text": "", "experience": json.dumps(experience, indent=2)})
        if is_url:
            try:
                content = prepare_content_message(eval_prompt, file_input)
                resp = self.llm_client.invoke([HumanMessage(content=content)], session_id=session_id, callbacks=[langfuse_handler])
                content = (resp.get("content") or "").strip()
                parsed = validate_json(content, session_id=session_id)
                if parsed.get("ok"):
                    return parsed.get("obj")
                repair = repair_json_with_llm(self.llm_client, content, session_id=session_id, max_attempts=2)
                if repair.get("ok"):
                    return repair.get("obj")
                bound_logger.error("eval_evaluate_input_url_parse_failure", error=parsed.get("error"))
                return self._error_fallback("Model returned invalid JSON for evaluation of URL input")
            except Exception as e:
                bound_logger.error("eval_evaluate_input_url_error", error=str(e), exc_info=True)
                return self._error_fallback(str(e))

        # For file uploads
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"File not found: {file_input}")

        # Upload file to Google GenAI
        uploaded_file = state.get("input_file")
        if not uploaded_file:
            bound_logger.info("eval_missing_uploaded_file_uploading", file_input=file_input)
            uploaded_file = self.multimodal_client.files.upload(file=file_input)

        # Wait until ACTIVE or timeout
        start_time = time.time()
        while True:
            current_file = self.multimodal_client.files.get(name=uploaded_file.name)
            if current_file.state == "ACTIVE":
                break
            elif current_file.state == "FAILED":
                raise RuntimeError("File processing failed.")
            elif time.time() - start_time > 180:
                raise HTTPException(status_code=502, detail="Timed out while trying to upload the file.")
            time.sleep(2)

        eval_prompt = load_prompt("eval.md", {"text": "", "experience": json.dumps(experience, indent=2)})

        # Send file + prompt to model
        try:
            # use the LLM wrapper for multimodal generation too
            resp = self.llm_client.generate_content(uploaded_file, eval_prompt, session_id=session_id, model=settings.EVALUATION_MODEL)
            content = (resp.get("content") or "").strip()
            parsed = validate_json(content, session_id=session_id)
            if parsed.get("ok"):
                return parsed.get("obj")
            repair = repair_json_with_llm(self.llm_client, content, session_id=session_id, max_attempts=2)
            if repair.get("ok"):
                return repair.get("obj")
            bound_logger.error("eval_evaluate_input_file_parse_failure", error=parsed.get("error"))
            return self._error_fallback("Model returned invalid JSON for evaluation of file input")
        except Exception as e:
            bound_logger.error("eval_evaluate_input_file_error", error=str(e), exc_info=True)
            return self._error_fallback(str(e))

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        """Main entrypoint for evaluation."""
        experience = state.get("experience", {})
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="eval")
        bound_logger.info("eval_execute_receive", experience_preview=(str(experience)[:200] if experience else None))
        raw_input: Optional[str] = state.get("raw_input")
        file_path: Optional[str] = state.get("input_file_path")

        if not experience:
            return {"evaluation": {"error": "No experience data to evaluate"}}

        try:
            if file_path:
                evaluation = self.evaluate_input(state)
            elif raw_input:
                evaluation = self.evaluate_from_text(state)
            else:
                evaluation = {"error": "No input provided for evaluation"}
        except Exception as e:
            bound_logger.error("eval_execute_error", error=str(e), exc_info=True)
            evaluation = self._error_fallback(str(e))
        bound_logger.info("evaluation_result", evaluation=evaluation)
        return {"evaluation": evaluation}

    def _error_fallback(self, message: str) -> Dict[str, Any]:
        return {
            "error": f"Evaluation failed: {message}",
            "hallucination": 0.0,
            "accuracy": 0.0,
            "conciseness": 0.0,
            "structure_compliance": "Fail",
            "overall_score": 0,
            "validation_reason": "Evaluation failed due to runtime error",
        }
