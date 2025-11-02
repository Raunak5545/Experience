import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.langgraph.config.model_config import workflow_config
from app.core.prompts import load_prompt
from app.core.logging import logger



class ValidationAgent:
    
    REQUIRED_FIELDS = ["destination"]
    MAX_ATTEMPTS = 0
    
    def __init__(self):
        # Get the model configuration for this node
        model_config = workflow_config.get_config("validation")
        
        # Initialize LLM with the configuration
        self.llm = ChatGoogleGenerativeAI(
            model=model_config.model_name,
            google_api_key=settings.LLM_API_KEY,
            **model_config.to_dict()
        )
    
    def check_completeness(self, extracted_text: str,session_id:str) -> Dict[str, Any]:
        prompt = f"""Analyze the following travel information and determine if it contains:
        1. Destination or City (specific location)
        2. Activities or Attractions

        Prompt user and ask for missing information if any of the above are absent.

        **Return validated if we have everything that we need**
        Extracted Information:

        {extracted_text}

        Respond in JSON format:
        {{
        "validated": false,
        "prompt": ""
        }}
        """
        prompt = load_prompt("validation.md", {"extracted_text": extracted_text})
        bound_logger = logger.bind(session_id=session_id, node="validation")
        response = self.llm.invoke([HumanMessage(content=prompt)], config={"callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": session_id,
         }})
        
        try:
            result = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            bound_logger.debug("validation_check_completeness_result", result=result)
            return result
        except Exception as e:
            bound_logger.error("validation_parse_error", error=str(e))
            return {
                "has_destination": False,
                "failed_reason" : "Error occured",
                "confidence": "0"
            }
    
    
    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        
        extracted_text = state.get("extracted_text", "")
        validation_attempts = state.get("validation_attempts", 0)
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="validation")
        bound_logger.info("validation_execute_start", attempts=validation_attempts)
        validation_result = self.check_completeness(extracted_text,state.get("session_id"))
        has_destination = validation_result.get("has_destination", [])
        is_validated =  validation_result.get("is_validated",False)
        failed_reason = validation_result.get("failed_reason","")
        validation_prompt = validation_result.get("validation_prompt","")
         
        if has_destination and is_validated:
            bound_logger.info("validation_success")
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
                    bound_logger.info("validation_success_after_max_attempts")
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
                    "next": "classification"
                }
            bound_logger.info("validation_needs_more_info", attempts=validation_attempts + 1)
            return {
                "validated": False,
                "validation_attempts": validation_attempts + 1,
                "validation_prompt": validation_prompt,
                "failed_reason":failed_reason,
                "next": "extraction"
            }
