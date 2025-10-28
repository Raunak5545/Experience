from typing import Any, Dict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.schema.experience import TravelPlan

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

        self.prompt = """
        You are an expert at structuring unorganized travel content into a detailed day-by-day itinerary JSON.
        Convert the given text into a valid JSON that matches the structure shown below—no markdown, no commentary,
        only valid JSON. Each day's plan should have a clear 'caption', 'description', and a 'schedule' of activities.

        Use the input text to populate fields intelligently:
        - Infer missing but obvious details from context.
        - Use arrays for multi-line descriptions.
        - Keep empty strings or arrays for missing data.
        - All times should be approximate (Morning, Afternoon, Evening, Night) if not explicitly mentioned.
        - For duration, estimate if not mentioned (e.g., walking tour ~3.5 hrs).
        - Keep JSON strictly valid and complete.

        Input text: {text}

        Output JSON structure:
        {{
          "plan": [
            {{
              "day": "1",
              "caption": "Short engaging title summarizing the day's experience.",
              "description": [
                "Sentence 1 describing the day.",
                "Sentence 2 describing highlights."
              ],
              "schedule": [
                {{
                  "time": "Morning",
                  "timeline": "Morning",
                  "description": [
                    "Sentence 1 describing morning activity.",
                    "Sentence 2 elaborating details."
                  ],
                  "type": {{
                    "name": "activity",
                    "value": {{
                      "name": "Activity name (e.g., Heritage Walk)",
                      "duration in hours": 3.5
                    }},
                    "placename": "Location or area name"
                  }},
                  "caption": "Short catchy summary for this activity"
                }}
                // Add more schedule items as needed
              ]
            }}
            // Add more days as needed
          ]
        }}

        Ensure your final output is valid JSON and matches the schema exactly.
        """

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")

        if not extracted_text:
            return {
                "next": "extraction"
            }

        llm_structured = self.llm.with_structured_output(TravelPlan)
        response = llm_structured.invoke(
            [
                HumanMessage(self.prompt.format(text=extracted_text)),
            ]
        )

        print("Returning from plan_agent")
        return {
            "travel_plan": response
        }
