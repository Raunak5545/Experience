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
from app.core.langgraph.config.model_config import workflow_config
from app.core.prompts import load_prompt
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.logging import logger

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
        # Get the model configuration for this node
        model_config = workflow_config.get_config("classification")
        
        # Initialize LLM with the configuration
        self.llm = ChatGoogleGenerativeAI(
            model=model_config.model_name,
            google_api_key=settings.LLM_API_KEY,
            **model_config.to_dict()
        )
    
    def classify(self, extracted_text: str,session_id :str) -> Dict[str, Any]:
        """Classify itinerary based on completeness"""
        prompt = load_prompt("classification.md", {"extracted_text": extracted_text})
        start = time.time()
        response = self.llm.invoke(
            [HumanMessage(content=prompt)],
            config={
                "callbacks":[langfuse_handler],
                "langfuse_session_id" : session_id,
            }
        )
        duration = time.time() - start
        logger.info("classification_llm_call_finished", session_id=session_id, duration_s=duration)
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
        session_id = state.get("session_id","") 
        logger.info("classification_execute_start", session_id=session_id)
        # Classify the itinerary
        classification_result = self.classify(extracted_text, session_id)

        classification_type = classification_result.get("type", "unmanaged")
        reason = classification_result.get("Explanation", "")
        logger.debug("classification_result", session_id=session_id, result=classification_result)
        logger.info("classification_execute_complete", session_id=session_id, classification_type=classification_type)
        return {
            "classification_type": classification_type,
            "classification_reason": reason,
        }



