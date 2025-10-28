
import json
from typing import Any, Dict

from fastapi import File, HTTPException
from langgraph.graph import END, StateGraph
from app.core.langgraph.agents.basic_info import BasicInfoAgent
from app.core.langgraph.agents.classification import ClassificationAgent
from app.core.langgraph.agents.extraction import ExtractionAgent
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.plan_agent import PlanAgent
from app.core.langgraph.agents.validation import ValidationAgent


def create_travel_workflow():
    """Create the LangGraph workflow"""
    
    # Initialize agents
    extraction_agent = ExtractionAgent()
    validation_agent = ValidationAgent()
    classification_agent = ClassificationAgent()
    basic_info_agent = BasicInfoAgent()
    plan_agent = PlanAgent() 
    # Create node functions
    def extraction_node(state: TravelAgentState) -> Dict[str, Any]:
        return extraction_agent.execute(state)
    
    def validation_node(state: TravelAgentState) -> Dict[str, Any]:
        return validation_agent.execute(state)
    
    def classification_node(state: TravelAgentState) -> Dict[str, Any]:
        return classification_agent.execute(state)
    def basic_info_node (state:TravelAgentState) -> Dict[str,Any]:
        return basic_info_agent.execute(state)
    def plan_agent_node(state:TravelAgentState) -> Dict[str,Any]:
        return plan_agent.execute(state)
    def combine_node(state: "TravelAgentState") -> Dict[str, Any]:
        # Safely extract Pydantic models or default to None
        basic_info = state.get("basic_info")
        tags_info = state.get("tags_info")
        travel_plan = state.get("travel_plan")
        classification_type = state.get("classification_type")

        # Use .model_dump() if model exists, else {}
        experience = {
            **(basic_info.model_dump() if basic_info else {}),
            "plan_type": classification_type,
            "travel_plan": travel_plan.model_dump() if travel_plan else None,
            "tags_info": tags_info.model_dump() if tags_info else None,
        }

        return {"experience": experience}

    # Build workflow graph
    workflow = StateGraph(TravelAgentState)

    # Add nodes
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("classification", classification_node)
    workflow.add_node("basic_info", basic_info_node)
    workflow.add_node("plan",plan_agent_node )
    workflow.add_node("combine_node",combine_node)
    
    # Set entry point
    workflow.set_entry_point("extraction")
    
    # Add edges
    workflow.add_edge("extraction", "validation")
    workflow.add_conditional_edges("validation", lambda x: x["next"],
        {
            "extraction": "extraction",
            "classification": "classification",
        }
)
    workflow.add_edge("classification","basic_info")
    workflow.add_edge("classification","plan")
    workflow.add_edge("basic_info","combine_node")
    workflow.add_edge("plan","combine_node")
    workflow.add_edge("combine_node",END)
    
    return workflow.compile()


def start_agentic_process(file_path:str ): 
    
    # Create workflow
    app = create_travel_workflow()
    
    # Example initial state
    initial_state: TravelAgentState = {
        "input_text": "",
        "input_file_path": file_path,
        "extracted_text": "",
        "extraction_complete": False,
        "validated": False,
        "validation_attempts": 0,
        "missing_fields": [],
        "validation_prompt": "",
        "classification": "",
        "classification_reason": "",
        "final_itinerary": {},
        "next": "",
        "failed_reason": ""
    }
    
    # Run the workflow
    try:
        result = app.invoke(initial_state)
        
        # Display final itinerary 
        print("\n" + "=" * 80)
        print("=" * 80)
        print(f"Classification: {result.get('classification', 'N/A').upper()}")
        print(f"Validated: {result.get('validated', False)}")
        print(f"Validation Attempts: {result.get('validation_attempts', 0)}")
        return result
        
    except Exception as e:
        print(f"\n Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500,detail=e.__str__())

