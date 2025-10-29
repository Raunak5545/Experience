import time
from typing import (
    Any,
    Dict,
)

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.schema.experience import TravelPlan
from app.core.prompts import load_prompt


class PlanAgent:
    """
    Agent that extracts structured day-by-day travel plans from unstructured text sources
    (e.g., PDF, transcript, or raw text) into a standardized JSON format.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.PLAN_ITINERARY_MODEL,
            temperature=0.4,
            google_api_key=settings.LLM_API_KEY
        )

        self.prompt = load_prompt("plan_agent.md", {"extracted_text": ""})

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")

        if not extracted_text:
            return {
                "next": "extraction"
            }

        llm_structured = self.llm.with_structured_output(TravelPlan)
        start = time.time()
        prompt = load_prompt("plan_agent.md", {"extracted_text": extracted_text})
        response = llm_structured.invoke([HumanMessage(prompt)])
        duration = time.time() - start
        print(f"[Timing] PlanAgent LLM call finished in {duration:.2f} seconds.")
        print("Returning from plan_agent")
        return {
            "travel_plan": response
        }
