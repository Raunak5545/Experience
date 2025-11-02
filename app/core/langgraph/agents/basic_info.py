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
from app.core.langgraph.config.model_config import workflow_config
from app.core.prompts import load_prompt
from app.core.logging import logger


class BasicInfoAgent:
    """
    Agent that extracts basics info from the text in a structured JSON format.
    """

    def __init__(self):
        # Get the model configuration for this node
        model_config = workflow_config.get_config("basic_info")
        
        # Initialize LLM with the configuration
        self.llm = ChatGoogleGenerativeAI(
            model=model_config.model_name,
            google_api_key=settings.LLM_API_KEY,
            **model_config.to_dict()
        )

        self.tools = [get_full_experience_taxonomy]
        self.agent_executor = create_react_agent(self.llm, self.tools)

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="basic_info")
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
        bound_logger.info("basic_info_llm_call_finished", duration_s=duration)
        tags_info = self.extract_tags(state)
        bound_logger.info("basic_info_execute_complete")
        return {"basic_info": response, "tags_info": tags_info}

    def extract_tags(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        session_id = state.get("session_id", "")
        bound_logger = logger.bind(session_id=session_id, node="basic_info_tags")
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
        bound_logger.debug("basic_info_agent_messages", message_count=len(result.get("messages", [])))
        for msg in result["messages"]:
            bound_logger.debug(
                "basic_info_agent_message_item",
                type=type(msg).__name__,
                content_preview=(msg.content[:200] if msg.content else "EMPTY"),
            )

        final_message = result["messages"][-1]

        # If final message is empty, the agent might have finished with a tool call
        # We need to prompt it again to generate the final JSON

        if not final_message.content or (
            isinstance(final_message.content, str) and final_message.content.strip() == ""
        ):
            bound_logger.debug("basic_info_final_message_empty")
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

        bound_logger.debug(
            "basic_info_final_message_content",
            content_preview=(final_message.content[:400] if final_message.content else None),
        )

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
        bound_logger.info("basic_info_tags_llm_finished", duration_s=tag_duration)
        bound_logger.debug("basic_info_tags_info", tags_info=tags_info)
        total_duration = time.time() - start
        bound_logger.info("basic_info_extract_tags_complete", total_duration_s=total_duration)
        return tags_info
