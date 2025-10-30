import json
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
from langfuse import observe

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.logging import logger
from app.core.prompts import load_prompt


class EvalAgent:
    """Agent that evaluates extracted travel information against the raw input for quality and compliance."""

    def __init__(self):
        self.eval_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.2,
            google_api_key=settings.LLM_API_KEY,
        )

        # Multimodal client for files
        self.multimodal_client = genai.Client(api_key=settings.LLM_API_KEY, debug_config={})

    def evaluate_from_text(self, state) -> Dict[str, Any]:
        eval_prompt = load_prompt("eval.md", {"text": text, "experience": json.dumps(experience, indent=2)})
        text = state.get("text", "")
        experience = state.get("experience", {})
        session_id = state.get("session_id", "")

        try:
            response = self.eval_llm.invoke(
                [HumanMessage(content=eval_prompt)],
                config={
                    "callbacks": [langfuse_handler],
                    "langfuse_session_id": session_id,
                },
            )
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            return self._error_fallback(str(e))

    def evaluate_input(self, state) -> Dict[str, Any]:
        """Evaluates input files or URLs along with the extracted experience."""
        from app.utils.file_handler import prepare_content_message

        experience = state.get("experience", {})
        file_input = state.get("input_file_path")
        is_url = state.get("is_url", False)
        session_id = state.get("session_id", "")

        eval_prompt = load_prompt("eval.md", {"text": "", "experience": json.dumps(experience, indent=2)})
        if is_url:
            try:
                content = prepare_content_message(eval_prompt, file_input, is_url=True)
                response = self.eval_llm.invoke(
                    [HumanMessage(content=content)],
                    config={
                        "callbacks": [langfuse_handler],
                        "langfuse_session_id": session_id,
                    },
                )
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(content)
            except Exception as e:
                return self._error_fallback(str(e))

        # For file uploads
        if not os.path.exists(file_input):
            raise FileNotFoundError(f"File not found: {file_input}")

        # Upload file to Google GenAI
        uploaded_file = state.get("input_file")
        if not uploaded_file:
            logger.info("Could not find uploaded_file uploading a new file")
            print("Could not find uploaded_file uploading a new file")
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
            response = self.multimodal_client.models.generate_content(
                model=settings.EVALUATION_MODEL,
                contents=[uploaded_file, eval_prompt],
            )
            # response = self.
            content = response.text.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            return self._error_fallback(str(e))

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        """Main entrypoint for evaluation."""
        experience = state.get("experience", {})
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
            evaluation = self._error_fallback(str(e))
        print("Evaluation Result:", evaluation)
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
