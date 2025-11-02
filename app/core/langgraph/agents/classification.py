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
from app.core.langgraph.llm_client import LLMClient
from app.core.langgraph.output_validator import validate_json, repair_json_with_llm
from app.core.logging import logger
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
        # Wrap with LLM client
        self.llm_client = LLMClient(llm=self.llm, name=model_config.model_name)
    
    def classify(self, extracted_text: str,session_id :str) -> Dict[str, Any]:
        """Classify itinerary based on completeness"""
        prompt = load_prompt("classification.md", {"extracted_text": extracted_text})
        start = time.time()
        bound_logger = logger.bind(session_id=session_id, node="classification")
        resp = self.llm_client.invoke([HumanMessage(content=prompt)], session_id=session_id, callbacks=[langfuse_handler])
        duration = time.time() - start
        bound_logger.info("classification_llm_call_finished", duration_s=duration)
        try:
            content = (resp.get("content") or "").strip()
            parsed = validate_json(content, session_id=session_id)
            if parsed.get("ok"):
                result = parsed.get("obj")
            else:
                repair = repair_json_with_llm(self.llm_client, content, session_id=session_id, max_attempts=1)
                if repair.get("ok"):
                    result = repair.get("obj")
                else:
                    # fallback to best-effort
                    result = json.loads(content.replace("```json", "").replace("```", ""))
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
        bound_logger = logger.bind(session_id=session_id, node="classification")
        bound_logger.info("classification_execute_start")
        # Classify the itinerary
        classification_result = self.classify(extracted_text, session_id)

        classification_type = classification_result.get("type", "unmanaged")
        reason = classification_result.get("Explanation", "")
        bound_logger.debug("classification_result", result=classification_result)
        bound_logger.info("classification_execute_complete", classification_type=classification_type)
        return {
            "classification_type": classification_type,
            "classification_reason": reason,
        }



