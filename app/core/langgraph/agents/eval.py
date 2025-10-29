import json
import os
import time
from typing import Any, Dict, Optional
from app.core.logging import logger

from fastapi import HTTPException
from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langfuse import observe
from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler


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

        #### **hallucination** (float, 0.0–1.0)
        **Proportion of fabricated/unsupported info**.
        - **Fabricated** = Any JSON detail **NOT explicitly present** in files (e.g., invented flight number, wrong price).
        - **Unsupported** = Inferred/guessed without direct evidence.
        
        **Calculation Steps**:
        1. List **all atomic facts** in JSON (e.g., 20 facts).
        2. Count **hallucinated facts** (H).
        3. `hallucination = H / total_facts` (round to 2 decimals).
        **Example**: 3/10 facts fabricated → 0.30

        #### **accuracy** (float, 0.0–1.0)
        **Correctness of supported info**.
        - **Correct** = Matches file **exactly** 
        - **Incorrect** = Misread/typo/distorted.
        - **Ignores hallucinations** (those are penalized separately).
        
        **Calculation Steps**:
        1. From **non-hallucinated facts**, count **correct (C)** vs **incorrect (I)**.
        2. `accuracy = C / (C + I)` (or 1.0 if no non-hallucinated facts).
        **Mark 0.0** if all supported facts wrong.

        #### **conciseness** (float, 0.0–1.0)
        **Brevity + clarity without fluff/redundancy**.
        - **High**: Precise, no repeats, essential only.
        - **Low**: Verbose, duplicated keys, irrelevant notes.
        - JSON size should be **minimal viable**.
        
        **Scoring Guidelines**:
        - **1.0**: Perfect (no redundancy, clear keys).
        - **0.8–0.9**: Minor fluff.
        - **0.5–0.7**: Some repeats.
        - **<0.5**: Bloated/unclear.
        **Subjective but evidence-based** (quote redundant parts).

        #### **structure_compliance** (string: "Pass" or "Fail")
        **Adheres to expected JSON schema**.
        - **Pass**: Valid JSON; **required keys present**; correct types (e.g., dates as ISO strings); no extras; nested arrays/objects logical.
        - **Fail**: Invalid JSON, missing keys, wrong types, malformed.
        
        **Logic**:
        1. Parse JSON → valid?
        2. Check keys/types vs schema.
        3. Nested structure logical?
        **Fail** if **ANY** violation.

        #### **overall_score** (integer, 0–100)
        **Composite quality**.
        **Formula** (transparent):
        `overall = round( (accuracy * 35) + ((1 - hallucination) * 35) + (conciseness * 20) + (structure_compliance == "Pass" ? 10 : 0) )`
        
        **Auto-compute** using above.
        **Mark 0** if structure="Fail" AND hallucination > 0.5.

        #### **validation_reason** (string)
        **Why manual/human validation needed**.
        - **Empty** if `overall_score >= 80` AND `hallucination <= 0.15` AND `structure="Pass"`.
        - Else: **Bullet list** of **top 3 issues** + **file evidence**.
        
        **Keep <100 chars**.
        **Example**: "- Hallucinated hotel: Not in any file.\\n- Wrong date: Image shows 2025-10-28."

        ### ⚙️ Evaluation Logic
        - Compare extracted content with original input.  
        - Identify any fabricated or hallucinated details.  
        - Check if key information is missing or misinterpreted.  
        - Assess clarity, completeness, and logical structure.  
        - Recommend validation if overall_score < 80, hallucination > 0.15, or structure_compliance == "Fail".

        ## ALSO PROVIDE REASON FOR YOUR SCORING

        ### ✅ Output Format
        Return strictly in JSON format like this:
        ```json
        {{
          "hallucination": <float>,
          "hallucination_score_reason": <string>,
          "accuracy": <float>,
          "accuracy_score_reason": <string>,
          "conciseness": <float>,
          "conciseness_score_reason": <string>,
          "structure_compliance": <string>,
          "overall_score": <integer>,
          "validation_reason": <string>
        }}
        ```
        """

    def evaluate_from_text(self, state) -> Dict[str, Any]:
        text = state.get("raw_input", "")
        experience = state.get("experience", {})
        session_id = state.get("session_id", "")
        eval_prompt = f"""
        {self.prompt}

        ### Original Input
        {text}

        ###  Extracted Experience
        {json.dumps(experience, indent=2)}
        """

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

    def evaluate_from_file(self, state) -> Dict[str, Any]:
        """Uploads and evaluates all input files along with the extracted experience."""
        experience = state.get("experience", {})
        file_path = state.get("input_file_path")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Upload file to Google GenAI
        uploaded_file = state.get("input_file")
        if not uploaded_file:
            logger.info("Couldnot find uploaded_file uploading a new file")
            print("Couldnot find uploaded_file uploading a new file")
            uploaded_file = self.multimodal_client.files.upload(file=file_path)

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

        # Prepare multimodal evaluation prompt
        eval_prompt = f"""
        {self.prompt}

        ### 🔹 Extracted Experience (Final Output)
        {json.dumps(experience, indent=2)}
        """

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
                evaluation = self.evaluate_from_file(state)
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
