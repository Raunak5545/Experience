import json
from typing import (
    Any,
    Dict,
)

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState


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
            model=settings.LLM_MODEL,
            temperature=0.1,
            google_api_key=settings.LLM_API_KEY,
        )
    
    def classify(self, extracted_text: str) -> Dict[str, Any]:
        """Classify itinerary based on completeness"""
        prompt = f"""Analyze the following travel information and classify it as either: 
        - MANAGED: Contains at least one of the following elements: cancellation policy, contact info, inclusions/exclusions, services, payment terms, or pricing.
        - UNMANAGED: None of the above elements are present.

        Look for these specific elements:
        1. Cancellation Policy
        2. Contact Information
        3. Inclusions and Exclusions
        4. Services Offered
        5. Payment Terms
        6. Pricing Information

        Extracted Information:
        {extracted_text}


        Respond strictly in JSON format with these fields:
        {{
        "type": "MANAGED" or "UNMANAGED",
        "Explanation": "Explain which elements were found or missing",
        "classification_confidence": 0.88
        }}"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        
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



