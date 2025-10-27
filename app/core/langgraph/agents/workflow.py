
import json
from typing import Any, Dict

from fastapi import File, HTTPException
from langgraph.graph import END, StateGraph
from app.core.langgraph.agents.classification import ClassificationAgent
from app.core.langgraph.agents.extraction import ExtractionAgent
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.validation import ValidationAgent


def create_travel_workflow():
    """Create the LangGraph workflow"""
    
    # Initialize agents
    extraction_agent = ExtractionAgent()
    validation_agent = ValidationAgent()
    classification_agent = ClassificationAgent()
    
    # Create node functions
    def extraction_node(state: TravelAgentState) -> Dict[str, Any]:
        return extraction_agent.execute(state)
    
    def validation_node(state: TravelAgentState) -> Dict[str, Any]:
        return validation_agent.execute(state)
    
    def classification_node(state: TravelAgentState) -> Dict[str, Any]:
        return classification_agent.execute(state)
       
    # Build workflow graph
    workflow = StateGraph(TravelAgentState)
    # Add nodes
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("validation", validation_node)
    workflow.add_node("classification", classification_node)
    
    # Set entry point
    workflow.set_entry_point("extraction")
    
    # Add edges
    workflow.add_edge("extraction", "validation")
    workflow.add_conditional_edges("validation", lambda x: x["next"],
        {
            "extraction": "extraction",
            "classification": "classification",
            "end": END
        }
)
    workflow.add_edge("classification", END)
    
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
        print(json.dumps(result.get("final_itinerary", {}), indent=2))
        
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

