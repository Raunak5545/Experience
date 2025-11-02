import json
import uuid
from typing import (
    Any,
    Dict,
)

from fastapi import (
    File,
    HTTPException,
)
from langgraph.graph import (
    END,
    StateGraph,
)

from app.core.langgraph.agents.basic_info import BasicInfoAgent
from app.core.langgraph.agents.classification import ClassificationAgent
from app.core.langgraph.agents.eval import EvalAgent
from app.core.langgraph.agents.extraction import ExtractionAgent
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.plan_agent import PlanAgent
from app.core.langgraph.agents.validation import ValidationAgent
from app.core.logging import logger


def create_travel_workflow():
    """Create the LangGraph workflow

    Workflow Flow:
    1. Extraction Agent - Processes all modalities (text/audio/image/pdf/video) from file or URL
    2. Classification Agent - Classifies content as Managed/Unmanaged
    3. Parallel Processing:
       - Basic Info Agent - Extracts metadata
       - Plan Agent - Creates day-wise itinerary
    4. Combine Node - Merges outputs into final structured JSON

    Input can be either:
    - File upload: Local file processed through multimodal client
    - URL: Remote file processed through text LLM with multimodal content
    """

    # Initialize agents
    extraction_agent = ExtractionAgent()
    validation_agent = ValidationAgent()  # Keeping this but not connecting in graph
    classification_agent = ClassificationAgent()
    basic_info_agent = BasicInfoAgent()
    plan_agent = PlanAgent()
    eval_agent = EvalAgent()

    # Create node functions
    def extraction_node(state: TravelAgentState) -> Dict[str, Any]:
        """Extract travel information from multimodal input"""
        logger.info("node_start", node="extraction", session_id=state.get("session_id"))
        try:
            result = extraction_agent.execute(state)
            logger.info("node_complete", node="extraction", session_id=state.get("session_id"))
            return result
        except Exception as e:
            logger.error("node_error", node="extraction", session_id=state.get("session_id"), error=str(e), exc_info=True)
            raise

    def validation_node(state: TravelAgentState) -> Dict[str, Any]:
        """Validate extracted information (not connected in current workflow)"""
        return validation_agent.execute(state)

    def classification_node(state: TravelAgentState) -> Dict[str, Any]:
        """Classify extracted content as Managed/Unmanaged"""
        logger.info("node_start", node="classification", session_id=state.get("session_id"))
        try:
            result = classification_agent.execute(state)
            logger.info("node_complete", node="classification", session_id=state.get("session_id"))
            return result
        except Exception as e:
            logger.error("node_error", node="classification", session_id=state.get("session_id"), error=str(e), exc_info=True)
            raise

    def basic_info_node(state: TravelAgentState) -> Dict[str, Any]:
        """Extract basic information and metadata"""
        logger.info("node_start", node="basic_info", session_id=state.get("session_id"))
        try:
            result = basic_info_agent.execute(state)
            logger.info("node_complete", node="basic_info", session_id=state.get("session_id"))
            return result
        except Exception as e:
            logger.error("node_error", node="basic_info", session_id=state.get("session_id"), error=str(e), exc_info=True)
            raise

    def plan_agent_node(state: TravelAgentState) -> Dict[str, Any]:
        """Generate day-wise travel itinerary"""
        logger.info("node_start", node="plan", session_id=state.get("session_id"))
        try:
            result = plan_agent.execute(state)
            logger.info("node_complete", node="plan", session_id=state.get("session_id"))
            return result
        except Exception as e:
            logger.error("node_error", node="plan", session_id=state.get("session_id"), error=str(e), exc_info=True)
            raise

    def combine_node(state: "TravelAgentState") -> Dict[str, Any]:
        # Safely extract Pydantic models or default to None
        basic_info = state.get("basic_info")
        tags_info = state.get("tags_info")
        travel_plan = state.get("travel_plan")
        classification_type = state.get("classification_type")

        experience = {
            **(basic_info.model_dump() if basic_info else {}),
            **(travel_plan.model_dump() if travel_plan else {}),
            **(tags_info.model_dump() if tags_info else {}),
            "plan_type": classification_type,
        }
        logger.info("node_complete", node="combine", session_id=state.get("session_id"))
        return {"experience": experience}

    def eval_node(state: TravelAgentState) -> Dict[str, Any]:
        """Evaluate the final structured"""
        logger.info("node_start", node="eval", session_id=state.get("session_id"))
        try:
            result = eval_agent.execute(state)
            logger.info("node_complete", node="eval", session_id=state.get("session_id"))
            return result
        except Exception as e:
            logger.error("node_error", node="eval", session_id=state.get("session_id"), error=str(e), exc_info=True)
            raise

    # Build workflow graph
    workflow = StateGraph(TravelAgentState)

    # Add nodes
    workflow.add_node("extraction", extraction_node)
    # Note: validation node not added to graph per requirements
    workflow.add_node("classification", classification_node)
    workflow.add_node("basic_info_node", basic_info_node)
    workflow.add_node("plan", plan_agent_node)
    workflow.add_node("combine_node", combine_node)
    workflow.add_node("eval_node", eval_node)

    # Set entry point
    workflow.set_entry_point("extraction")

    # Add edges for the workflow flow
    # 1. Extraction → Classification
    workflow.add_edge("extraction", "classification")

    # 2. Classification → Basic Info & Plan (parallel execution)
    workflow.add_edge("classification", "basic_info_node")
    workflow.add_edge("classification", "plan")

    # 3. Both Basic Info and Plan → Combine Node
    workflow.add_edge("basic_info_node", "combine_node")
    workflow.add_edge("plan", "combine_node")

    # 4 Combine Node → Eval Node
    workflow.add_edge("combine_node", "eval_node")

    # 4. Eval Node → End
    workflow.add_edge("eval_node", END)

    return workflow.compile()


def start_agentic_process(file_input: str, is_url: bool = False) -> Dict[str, Any]:
    """
    Start the agentic workflow process.

    Args:
        file_input: Either a file path or URL
        is_url: Whether the input is a URL
    """
    # Initialize workflow
    workflow = create_travel_workflow()

    state = {"input_file_path": file_input, "is_url": is_url, "session_id": str(uuid.uuid4())}
    logger.info("agentic_process_started", session_id=state.get("session_id"), input=file_input, is_url=is_url)
    try:
        result = workflow.invoke(state)
        logger.info("agentic_process_completed", session_id=state.get("session_id"))
        return result
    except Exception as e:
        logger.error("agentic_process_failed", session_id=state.get("session_id"), error=str(e), exc_info=True)
        raise
