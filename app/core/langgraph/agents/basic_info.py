from typing import Any, Dict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.schema.experience import BasicInfo

class BasicInfoAgent:
    """
    Agent that extracts basics info from the text in a structured JSON format.
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model = settings.LLM_MODEL,
            temperature=0.4,
            google_api_key = settings.LLM_API_KEY
        )

        self.prompt = """ 
                    You are an expert at extracting and structuring information from unstructured text sources (such as extracted content from PDFs, video 
                    transcripts, image descriptions, or raw text files) into a standardized JSON format for tour or experience listings. 
                    Your output must be exactly valid JSON matching the structure below—no extra fields, no markdown, no explanations. 
                    Use the input text to populate fields intelligently: infer or extract details where possible, but leave as empty strings
                    /arrays if not explicitly mentioned. For location, if coordinates are not in the text, use approximate ones based on 
                    the place (e.g., via known geography). For Inclusion and Exclusion, create arrays of strings listing key inclusions/exclusions mentioned.
                    For FAQ, generate any number of question-answer pairs (0 or more): questions must be relatable to the scenario (e.g., suitability, 
                    best options, logistics) and directly inspired by details in the source text; answers must be concise paraphrases or direct quotes from the 
                    text supporting the question.
                    Input text: {text}
                    Output JSON structure:
                    {{
                    "caption": "A short, engaging one-line caption summarizing the experience.",
                    "summary": ["Bullet point 1 summary from text.", "Bullet point 2 summary from text."],
                    "location": {{
                    "city": "Extracted or inferred city",
                    "state": "Extracted or inferred state",
                    "country": "Extracted or inferred country",
                    "placeName": "Full place or tour name from text",
                    "coordinates": {{
                    "type": "Point",
                    "coordinates": [longitude, latitude]
                    }}
                    }},
                    "inclusion": ["List item 1 from text.", "List item 2 from text."],
                    "exclusion": ["List item 1 from text.", "List item 2 from text."],
                    "faq": [
                    {{
                    "question": "Relatable question 1 based on text details.",
                    "answer": "Concise answer drawn from text."
                    }},
                    {{
                    "question": "Relatable question 2 based on text details.",
                    "answer": "Concise answer drawn from text."
                    }}
                    // Add more FAQs as needed based on text, or leave empty if no relevant FAQs can be formed
                    ]
                    }}
                    Ensure the JSON is complete and parseable. If the text lacks details for a field, use empty string/array but keep the structure intact.                      """
    def execute(self,state:TravelAgentState) -> Dict[str,Any]:
        extracted_text = state.get("extracted_text")
        if not extracted_text :
            return {
                "next":"extraction"
            }
        llm_structured = self.llm.with_structured_output(BasicInfo)
        response = llm_structured.invoke(
            [
                HumanMessage(self.prompt.format(text = extracted_text)),
            ]
        )
        return {
            "basic_info":response 
        }
