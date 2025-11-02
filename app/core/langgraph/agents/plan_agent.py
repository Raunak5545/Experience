import time
from typing import (
    Any,
    Dict,
)

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.langgraph.schema.experience import TravelPlan
from app.core.langgraph.config.model_config import workflow_config
from app.core.prompts import load_prompt
from app.core.logging import logger


class PlanAgent:
    """
    Agent that extracts structured day-by-day travel plans from unstructured text sources
    (e.g., PDF, transcript, or raw text) into a standardized JSON format.
    """

    def __init__(self):
        # Get the model configuration for this node
        model_config = workflow_config.get_config("plan")
        
        # Initialize LLM with the configuration
        self.llm = ChatGoogleGenerativeAI(
            model=model_config.model_name,
            google_api_key=settings.LLM_API_KEY,
            **model_config.to_dict()
        )

        self.prompt = load_prompt("plan_agent.md", {"extracted_text": ""})

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        session_id = state.get("session_id", "")

        llm_structured = self.llm.with_structured_output(TravelPlan)
        start = time.time()
        prompt = load_prompt("plan_agent.md", {"extracted_text": extracted_text})
        response = llm_structured.invoke(
            [
                HumanMessage(prompt),
            ],
            config={
                "callbacks": [langfuse_handler],
                "langfuse_session_id": session_id,
            },
        )
        duration = time.time() - start
        logger.info("plan_agent_llm_finished", session_id=session_id, duration_s=duration)
        logger.info("plan_agent_execute_complete", session_id=session_id)
        return {"travel_plan": response}
