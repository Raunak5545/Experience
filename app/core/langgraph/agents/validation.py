import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState


class ValidationAgent:
    
    REQUIRED_FIELDS = ["destination"]
    MAX_ATTEMPTS = 0
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.2,
            google_api_key=settings.LLM_API_KEY 
        )
    
    def check_completeness(self, extracted_text: str) -> Dict[str, Any]:
        prompt = f"""Analyze the following travel information and determine if it contains:
1. Destination or City (specific location)
2. Travel Dates (at least one date reference)
3. Number of Travelers (explicit or implied)

**Return validated if we have everything that we need**
Extracted Information:

{extracted_text}

Respond in JSON format:
{{
    "has_destination": true/false,
    "is_validated" : true/false,
    "validation_prompt": "Extra instructions for the previous agent to try to fetch these missing information",
    "failed_reason" : "The reason due to which we the validation_failed"
    "confidence": "0-1"
}}"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            result = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            return result
        except:
            return {
                "has_destination": False,
                "failed_reason" : "Error occured",
                "confidence": "0"
            }
    
    
    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        
        extracted_text = state.get("extracted_text", "")
        validation_attempts = state.get("validation_attempts", 0)
        validation_result = self.check_completeness(extracted_text)
        print(validation_result)
        has_destination = validation_result.get("has_destination", [])
        is_validated =  validation_result.get("is_validated",False)
        failed_reason = validation_result.get("failed_reason","")
        validation_prompt = validation_result.get("validation_prompt","")
         
        if has_destination and is_validated:
            return {
                "validated": True,
                "validation_attempts": validation_attempts + 1,
                "missing_fields": [],
                "validation_prompt": "",
                "failed_reason" : "",
                "next": "classification"
            }
        else:
            # Check if we've exceeded max attempts
            if validation_attempts >= self.MAX_ATTEMPTS:
                if has_destination:
                    return {
                        "validated": True,
                        "validation_attempts": validation_attempts + 1,
                        "validation_prompt": "",
                        "failed_reason" : "",
                        "next": "classification",
                        }
                return {
                    "validated": False,
                    "validation_attempts": validation_attempts + 1,
                    "validation_prompt": "No Destination",
                    "next": "end"
                }
            return {
                "validated": False,
                "validation_attempts": validation_attempts + 1,
                "validation_prompt": validation_prompt,
                "failed_reason":failed_reason,
                "next": "extraction"
            }
