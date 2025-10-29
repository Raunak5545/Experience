import json
import time
from typing import (
    Any,
    Dict,
)

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.prompts import load_prompt


class ClassificationAgent:
    """Classifies itinerary as Managed or Unmanaged"""
    
    MANAGED_CRITERIA = [
        "cancellation_policy",
        "contact_info",
        "inclusions_exclusions",
        "services",
        "payment_terms",
        "pricing"
    ]
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.VALIDATION_MODEL,
            temperature=0.1,
            google_api_key=settings.LLM_API_KEY,
        )
    
    def classify(self, extracted_text: str) -> Dict[str, Any]:
        """Classify itinerary based on completeness"""
        prompt = load_prompt("classification.md", {"extracted_text": extracted_text})
        start = time.time()
        response = self.llm.invoke([HumanMessage(content=prompt)])
        duration = time.time() - start
        print(f"[Timing] ClassificationAgent LLM call finished in {duration:.2f} seconds.")
        try:
            result = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            return result
        except:
            return {
                "classification_type": "unmanaged",
                "found_criteria": [],
                "missing_criteria": self.MANAGED_CRITERIA,
                "confidence": "low",
                "reason": "Unable to parse classification"
            }
    
    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        """Main execution method"""
        
        
        extracted_text = state.get("extracted_text", "")
        
        # Classify the itinerary
        classification_result = self.classify(extracted_text)
        
        classification_type = classification_result.get("type", "unmanaged")
        reason = classification_result.get("Explanation", "")
        print(classification_result) 
        return {
            "classification_type": classification_type,
            "classification_reason": reason,
        }



