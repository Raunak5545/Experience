import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

<<<<<<< Updated upstream
from app.core.langgraph.schema.experience import BasicInfo, Experience, ExperienceTagsOutputScehma, TravelPlan
=======
from app.core.langgraph.schema.experience import BasicInfo, Experience, TravelPlan,Eval
>>>>>>> Stashed changes

class TravelAgentState(TypedDict):
    input_text: str
    input_file_path: list[dict[str, Any]]  # [{type: 'pdf/image/audio/video', path: '...', content: '...'}]
    
    extracted_text: str
    extraction_complete: bool
    
    validated: bool
    validation_attempts: int
    validation_prompt: str
    
    classification_type: str  # 'managed' or 'unmanaged'
    classification_reason: str
    
    final_itinerary: dict[str, Any]
    
    next: str
    messages: Annotated[list[BaseMessage], operator.add]

    failed_reason: str

    basic_information :BasicInfo
    travel_plan :TravelPlan
    experience : Experience
<<<<<<< Updated upstream
    tags_info : ExperienceTagsOutputScehma
=======
    evaluation:Eval
>>>>>>> Stashed changes
