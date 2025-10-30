import time
from typing import (
    Any,
    Dict,
)

from click import prompt
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.langfuse_callback import langfuse_handler
from app.core.langgraph.data.experience_taxonomy import TAXONOMY
from app.core.langgraph.schema.experience import (
    BasicInfo,
    ExperienceTagsOutputScehma,
)
from app.core.langgraph.tools.experience_types_tags import get_full_experience_taxonomy
from app.core.prompts import load_prompt


class BasicInfoAgent:
    """
    Agent that extracts basics info from the text in a structured JSON format.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.BASIC_INFO_MODEL, temperature=0.4, google_api_key=settings.LLM_API_KEY
        )

        self.tools = [get_full_experience_taxonomy]

        self.agent_executor = create_react_agent(self.llm, self.tools)

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        session_id = state.get("session_id", "")
        llm_structured = self.llm.with_structured_output(BasicInfo)
        start = time.time()
        response = llm_structured.invoke(
            [HumanMessage(load_prompt("basic_info.md", {"extracted_text": extracted_text}))],
            config={
                "callbacks": [langfuse_handler],
                "langfuse_session_id": session_id,
            },
        )
        duration = time.time() - start
        print(f"[Timing] BasicInfoAgent LLM call finished in {duration:.2f} seconds.")
        tags_info = self.extract_tags(state)
        print("Returning from basic-info")
        return {"basic_info": response, "tags_info": tags_info}

    def extract_tags(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        session_id = state.get("session_id", "")
        start = time.time()
        # First invocation - let agent use tools
        result = self.agent_executor.invoke(
            {"messages": [HumanMessage(content=load_prompt("tag.md", {"extracted_text": extracted_text}))]},
            config={
                "max_iterations": 1,  # single reasoning step
                "timeout": 60,
                "stream": False,
                "callbacks": [langfuse_handler],
                "langfuse_session_id": session_id,
            },
        )
        print("Agent result messages:")
        for msg in result["messages"]:
            print(f"Type: {type(msg).__name__}, Content: {msg.content[:200] if msg.content else 'EMPTY'}")

        final_message = result["messages"][-1]

        # If final message is empty, the agent might have finished with a tool call
        # We need to prompt it again to generate the final JSON

        if not final_message.content or (
            isinstance(final_message.content, str) and final_message.content.strip() == ""
        ):
            print("Final message empty, requesting JSON output...")
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=load_prompt("tag.md", {"extracted_text": extracted_text}))]},
                config={
                    "max_iterations": 1,
                    "timeout": 60,
                    "stream": False,
                    "callbacks": [langfuse_handler],
                    "langfuse_session_id": session_id,
                },
            )

        final_message = result["messages"][-1]

        print(f"Final message content: {final_message.content}")

        llm_structured = self.llm.with_structured_output(ExperienceTagsOutputScehma)
        tag_start = time.time()
        tags_info = llm_structured.invoke(
            [HumanMessage(content=f"Convert this to structured format: {final_message.content}")],
            config={
                "callbacks": [langfuse_handler],
                "langfuse_session_id": session_id,
            },
        )
        tag_duration = time.time() - tag_start
        print(f"[Timing] BasicInfoAgent tags LLM call finished in {tag_duration:.2f} seconds.")
        print(tags_info)
        total_duration = time.time() - start
        print(f"[Timing] BasicInfoAgent extract_tags total finished in {total_duration:.2f} seconds.")
        return tags_info
