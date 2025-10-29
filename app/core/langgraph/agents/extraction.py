import os
import time
from typing import Dict, Any, Optional
from fastapi import HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google import genai


from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState

from app.core.langgraph.agents.langfuse_callback import langfuse_handler


class ExtractionAgent:
    """Agent that extracts travel information from text or multimodal input."""

    def __init__(self):
        self.text_llm = ChatGoogleGenerativeAI(
            model=settings.EVALUATION_LLM,
            temperature=0.3,
            google_api_key=settings.LLM_API_KEY,
        )

        self.multimodal_client = genai.Client(api_key=settings.LLM_API_KEY)
        self.prompt = f"""
        ## 🧭 Travel Information Extraction Prompt
        You are an **advanced extraction specialist**.  
        Your task is to **analyze and extract all travel-related information** from the provided input files — which may include **images, videos, PDFs, audio, or raw text**.

        ### 🧩 Input Types
        - **Image:** Describe in detail everything visible — locations, landmarks, dates, signs, activities, and contextual text.
        - **Video:** Combine **visual scene descriptions**, **spoken audio transcripts**, and **text appearing in frames** to extract full travel-related context.
        - **PDF / Documents:** Extract both **text content** and **embedded visual/structural clues** (tables, receipts, itineraries, maps, etc.).
        - **Raw Text:** Parse and interpret natural language information, even if unstructured.

        ---

        ### 🕵️‍♂️ Your Objective
        Provide a **comprehensive, structured narrative summary** covering *all relevant travel information* present in the files.

        Focus especially on:
        1. **Destinations / Cities** — Mention every identifiable place or location.
        2. **Activities and Experiences** — Include sightseeing, adventure, relaxation, events, etc.
        3. **Budget and Pricing** — Include any cost-related details like package price, hotel cost, activity pricing, or transportation fares.

        If available, also include any **additional contextual travel details** (dates, accommodation, travelers, preferences, etc.) found within the data.

        ---

        ### 🧠 Output Format
        Return your findings as a **clear, detailed narrative**, not in a list or category table.

        ---

        ### ⚙️ Instructions
        - Combine and cross-verify information across all input files.  
        - Include **every relevant piece of travel-related data** — even if implied or mentioned briefly.  
        - If any file lacks explicit details, infer them logically based on surrounding context.  
        - Maintain accuracy, fluency, and completeness.
        ---

        **Return only the final structured narrative — no headings, bullet points, or notes.**
        """

    def extract_from_text(self, text: str) -> str:
        response = self.text_llm.invoke(
            [HumanMessage(content=self.prompt + f"Text To Analyze from : {text}")],
            config={"callbacks": [langfuse_handler]},
        )
        return response.content

    def extract_from_file(self, file_path: str, extra_prompt: Optional[str] = None) -> str:
        if not os.path.exists(file_path):

            raise FileNotFoundError(f"File not found: {file_path}")

        task_prompt = (
            extra_prompt or "Extract key travel information (dates, destinations, travelers, etc.) from this file."
        )
        full_prompt = f"{self.prompt}\n\n{task_prompt}"
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
            if diff_time > 180:
                raise HTTPException(status_code=502, detail="Timed out while trying to upload the file.")
            time.sleep(2)
        response = self.multimodal_client.models.generate_content(
            model=settings.EXTRACTION_MODEL, contents=[uploaded_file, full_prompt]
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
            "input_file": uploaded_file,
        }
