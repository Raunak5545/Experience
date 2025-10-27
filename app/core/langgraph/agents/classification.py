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
- MANAGED: Contains detailed booking information including cancellation policy, contact info, inclusions/exclusions, services, payment terms, and pricing
- UNMANAGED: Missing several major components listed above

Look for these specific elements:
1. Cancellation Policy
2. Contact Information
3. Inclusions and Exclusions
4. Services Offered
5. Payment Terms
6. Pricing Information

Extracted Information:
{extracted_text}

Respond in JSON format:
{{
    "classification_type": "managed" or "unmanaged",
    "found_criteria": ["criterion1", "criterion2"],
    "missing_criteria": ["criterion3", "criterion4"],
    "confidence": "high/medium/low",
    "reason": "Brief explanation"
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
        
        classification_type = classification_result.get("classification", "unmanaged")
        reason = classification_result.get("reason", "")
        print(classification_result) 
        return {
            "classification_type": classification_type,
            "classification_reason": reason,
        }



