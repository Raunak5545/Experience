import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import HTTPException
from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState


class EvalAgent:
    """Agent that evaluates extracted travel information against the raw input for quality and compliance."""

    def __init__(self):
        self.eval_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.2,
            google_api_key=settings.LLM_API_KEY,
        )

        # Multimodal client for files
        self.multimodal_client = genai.Client(api_key=settings.LLM_API_KEY)

        # Core evaluation prompt
        self.prompt = """
        ## 🎯 Travel Information Evaluation Prompt
        You are a **Travel Information Evaluation Specialist**.
        Your job is to **evaluate the quality and reliability** of a travel information extraction result.

        ### 🧩 Inputs
        You are given:
        1. **Original Input Files** — the user's initial travel data or documents.
        2. **Extracted Experience (Final Output)** — the model-generated narrative of travel details.

        ### 🧠 Evaluation Goals
        You must evaluate how accurate, faithful, and concise the extraction is.

        ### 📊 Evaluation Parameters
        Provide the following metrics (JSON format only):

        - **hallucination** → Proportion of fabricated/unsupported information (float between 0.0 and 1.0)
        - **accuracy** → Correctness of extracted information vs. original input (float between 0.0 and 1.0)
        - **conciseness** → Clarity and brevity without redundancy (float between 0.0 and 1.0)
        - **structure_compliance** → Whether output follows expected schema/format (string: "Pass" or "Fail")
        - **overall_score** → Composite quality score (integer between 0 and 100)
        - **validation_reason** → Explanation if validation is required (string, empty if not required)

        ### ⚙️ Evaluation Logic
        - Compare extracted content with original input.  
        - Identify any fabricated or hallucinated details.  
        - Check if key information is missing or misinterpreted.  
        - Assess clarity, completeness, and logical structure.  
        - Recommend validation if overall_score < 80, hallucination > 0.15, or structure_compliance == "Fail".

        ### ✅ Output Format
        Return strictly in JSON format like this:
        ```json
        {{
          "hallucination": <float>,
          "accuracy": <float>,
          "conciseness": <float>,
          "structure_compliance": <string>,
          "overall_score": <integer>,
          "validation_reason": <string>
        }}
        ```
        """


    def evaluate_from_text(self, experience: Dict[str, Any], text: str) -> Dict[str, Any]:
        eval_prompt = f"""
        {self.prompt}

        ### 🔹 Original Input
        {text}

        ### 🔹 Extracted Experience
        {json.dumps(experience, indent=2)}
        """

        try:
            response = self.eval_llm.invoke([HumanMessage(content=eval_prompt)])
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            return self._error_fallback(str(e))


    def evaluate_from_file(self, experience: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Uploads and evaluates all input files along with the extracted experience."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Upload file to Google GenAI
        uploaded_file = self.multimodal_client.files.upload(file=file_path)

        # Wait until ACTIVE or timeout
        start_time = time.time()
        while True:
            current_file = self.multimodal_client.files.get(name=uploaded_file.name)
            if current_file.state == "ACTIVE":
                break
            elif current_file.state == "FAILED":
                raise RuntimeError("File processing failed.")
            elif time.time() - start_time > 20:
                raise HTTPException(status_code=502, detail="Timed out while trying to upload the file.")
            time.sleep(2)

        # Prepare multimodal evaluation prompt
        eval_prompt = f"""
        {self.prompt}

        ### 🔹 Extracted Experience (Final Output)
        {json.dumps(experience, indent=2)}
        """

        # Send file + prompt to model
        try:
            response = self.multimodal_client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=[uploaded_file, eval_prompt],
            )
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
                evaluation = self.evaluate_from_file(experience, file_path)
            elif raw_input:
                evaluation = self.evaluate_from_text(experience, raw_input)
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
