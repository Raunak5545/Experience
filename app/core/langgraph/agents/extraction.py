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
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.prompts import load_prompt


class ExtractionAgent:
    """Agent that extracts travel information from text or multimodal input."""

    def __init__(self):
        self.text_llm = ChatGoogleGenerativeAI(
            model=settings.EVALUATION_LLM,
            temperature=0.3,
            google_api_key=settings.LLM_API_KEY,
        )

        self.multimodal_client = genai.Client(api_key=settings.LLM_API_KEY)
        # Load extraction prompt template
        self.prompt = load_prompt("extraction.md", {"extra_instructions": ""})


    def extract_from_text(self, text: str) -> str:
        # Inject dynamic text into prompt (append for now)
        prompt = self.prompt + f"\n\n{text}"
        response = self.text_llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def extract_from_file(self, file_path: str, extra_prompt: Optional[str] = None) -> str:
        if not os.path.exists(file_path):

            raise FileNotFoundError(f"File not found: {file_path}")
        task_prompt = extra_prompt or "Extract key travel information (dates, destinations, travelers, etc.) from this file."
        full_prompt = load_prompt("extraction.md", {"extra_instructions": task_prompt})
        # Timeout
        start_time = time.time()

        uploaded_file = self.multimodal_client.files.upload(file=file_path)
        while True:
            current_file = self.multimodal_client.files.get(name=uploaded_file.name)
            print("Current state:", current_file.state)
            if current_file.state == "ACTIVE":
                break
            elif current_file.state == "FAILED":
                raise RuntimeError("File processing failed.")
            curr_time = time.time()
            diff_time = curr_time - start_time
            if diff_time > 60:
                raise HTTPException(status_code=502, detail="Timed out while trying to upload the file.")
            time.sleep(2)
        response = self.multimodal_client.models.generate_content(
            model=settings.EXTRACTION_MODEL,
            contents=[uploaded_file, full_prompt]
        )
        return (response.text, uploaded_file)

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        
        raw_input: Optional[str] = state.get("raw_input")
        file_path: Optional[str] = state.get("input_file_path")

        if file_path:
            extracted_text, uploaded_file = self.extract_from_file(file_path, state.get("validation_prompt"))
        elif raw_input:
            extracted_text = self.extract_from_text(raw_input)
        else:
            extracted_text = "No input provided."
        return {
            "extracted_text": extracted_text,
            "extraction_complete": True,
            'input_file': uploaded_file
        }
