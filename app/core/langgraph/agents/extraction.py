import os
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google import genai


from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState


class ExtractionAgent:
    """Agent that extracts travel information from text or multimodal input."""

    def __init__(self):
        self.text_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.3,
            google_api_key=settings.LLM_API_KEY,
        )

        self.multimodal_client = genai.Client(api_key=settings.LLM_API_KEY)
        self.prompt = f"""You are a travel data extraction specialist. Extract ALL travel-related information from the following text.

Focus on identifying:
- Destinations/Cities
- Travel dates (check-in, check-out, departure, arrival)
- Number of travelers (adults, children, infants)
- Activities and experiences
- Accommodations and hotels
- Transportation details
- Budget and pricing
- Special requests or preferences
- Contact information
- Cancellation policies
- Inclusions and exclusions
- Services offered
- Payment terms

Return all relevant information in a clear, structured narrative format.
"""


    def extract_from_text(self, text: str) -> str:
        response = self.text_llm.invoke([HumanMessage(content=self.prompt + f"Text To Analyze from : {text}")])
        return response.content

    def extract_from_file(self, file_path: str, extra_prompt: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(extra_prompt)
        task_prompt = extra_prompt  or  "Extract key travel information (dates, destinations, travelers, etc.) from this file."
        full_prompt = f"{self.prompt}\n\n{task_prompt}"
        print(full_prompt)
        
        uploaded_file = self.multimodal_client.files.upload(file=file_path)
        response = self.multimodal_client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=[uploaded_file, full_prompt]
        )
        return response.text

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        
        raw_input: Optional[str] = state.get("raw_input")
        print(state)
        file_path: Optional[str] = state.get("input_file_path")

        if file_path:
            extracted_text = self.extract_from_file(file_path,state.get("validation_prompt"))
        elif raw_input:
            extracted_text = self.extract_from_text(raw_input)
        else:
            extracted_text = "No input provided."

        print(extracted_text)
        return {
            "extracted_text": extracted_text,
            "extraction_complete": True,
        }
