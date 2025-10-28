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
    """Create the LangGraph workflow
    
    Workflow Flow:
    1. Extraction Agent - Processes all modalities (text/audio/image/pdf/video)
    2. Classification Agent - Classifies content as Managed/Unmanaged
    3. Parallel Processing:
       - Basic Info Agent - Extracts metadata
       - Plan Agent - Creates day-wise itinerary
    4. Combine Node - Merges outputs into final structured JSON
    """
    
    # Initialize agents
    extraction_agent = ExtractionAgent()
    validation_agent = ValidationAgent()  # Keeping this but not connecting in graph
    classification_agent = ClassificationAgent()
    basic_info_agent = BasicInfoAgent()
    plan_agent = PlanAgent()
    
    # Create node functions
    def extraction_node(state: TravelAgentState) -> Dict[str, Any]:
        """Extract travel information from multimodal input"""
        return extraction_agent.execute(state)
    
    def validation_node(state: TravelAgentState) -> Dict[str, Any]:
        """Validate extracted information (not connected in current workflow)"""
        return validation_agent.execute(state)
    
    def classification_node(state: TravelAgentState) -> Dict[str, Any]:
        """Classify extracted content as Managed/Unmanaged"""
        return classification_agent.execute(state)
    
    def basic_info_node(state: TravelAgentState) -> Dict[str, Any]:
        """Extract basic information and metadata"""
        return basic_info_agent.execute(state)
    
    def plan_agent_node(state: TravelAgentState) -> Dict[str, Any]:
        """Generate day-wise travel itinerary"""
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
    # Note: validation node not added to graph per requirements
    workflow.add_node("classification", classification_node)
    workflow.add_node("basic_info", basic_info_node)
    workflow.add_node("plan", plan_agent_node)
    workflow.add_node("combine_node", combine_node)
    
    # Set entry point
    workflow.set_entry_point("extraction")
    
    # Add edges for the workflow flow
    # 1. Extraction → Classification
    workflow.add_edge("extraction", "classification")
    
    # 2. Classification → Basic Info & Plan (parallel execution)
    workflow.add_edge("classification", "basic_info")
    workflow.add_edge("classification", "plan")
    
    # 3. Both Basic Info and Plan → Combine Node
    workflow.add_edge("basic_info", "combine_node")
    workflow.add_edge("plan", "combine_node")
    
    # 4. Combine Node → End
    workflow.add_edge("combine_node", END)
    
    return workflow.compile()


def start_agentic_process(file_path: str = None, raw_input: str = None):
    """
    Start the agentic travel planning process
    
    Args:
        file_path: Path to input file (image/pdf/video/audio)
        raw_input: Raw text input (if no file provided)
    
    Returns:
        Dict containing the final structured travel experience
    """
    
    # Validate input
    if not file_path and not raw_input:
        raise HTTPException(
            status_code=400, 
            detail="Either file_path or raw_input must be provided"
        )
    
    # Create workflow
    app = create_travel_workflow()
    
    # Initialize state
    initial_state: TravelAgentState = {
        "input_text": raw_input or "",
        "input_file_path": file_path,
        "raw_input": raw_input,  # Add raw_input to state
        "extracted_text": "",
        "extraction_complete": False,
        "validated": False,
        "validation_attempts": 0,
        "missing_fields": [],
        "validation_prompt": "",
        "classification_type": "",
        "classification_reason": "",
        "final_itinerary": {},
        "next": "",
        "failed_reason": "",
        "messages": [],
        "basic_info": None,
        "travel_plan": None,
        "experience": None
    }
    
    # Run the workflow
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 80)
        print("WORKFLOW EXECUTION COMPLETED")
        print("=" * 80)
        print(f"Classification Type: {result.get('classification_type', 'N/A').upper()}")
        print(f"Classification Reason: {result.get('classification_reason', 'N/A')}")
        print(f"Extraction Complete: {result.get('extraction_complete', False)}")
        
        # Display final experience/itinerary
        if result.get("experience"):
            print("\n" + "-" * 40)
            print("FINAL STRUCTURED OUTPUT:")
            print("-" * 40)
            print(json.dumps(result.get("experience", {}), indent=2))
        
        return result
        
    except Exception as e:
        print(f"\nError during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )


# Optional: Helper function to visualize the workflow
def visualize_workflow():
    """
    Generate a visual representation of the workflow
    (Requires additional dependencies like matplotlib/graphviz)
    """
    workflow = create_travel_workflow()
    
    print("\nWORKFLOW STRUCTURE:")
    print("-" * 40)
    print("1. EXTRACTION_AGENT")
    print("   ↓")
    print("2. CLASSIFICATION_AGENT")
    print("   ↓")
    print("3. [PARALLEL EXECUTION]")
    print("   ├── BASIC_INFO_AGENT")
    print("   └── PLAN_AGENT")
    print("   ↓")
    print("4. COMBINE_NODE")
    print("   ↓")
    print("5. END")
    print("-" * 40)
    
    return workflow