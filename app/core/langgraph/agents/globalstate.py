import operator
from typing import Annotated, Any, TypedDict

from google.ai.generativelanguage_v1beta.types.file import File
from langchain_core.messages import BaseMessage
from regex import E

from app.core.langgraph.schema.experience import BasicInfo, Experience, ExperienceTagsOutputScehma, TravelPlan,Eval

class TravelAgentState(TypedDict):
    session_id : str
    input_text: str
    input_file_path: list[dict[str, Any]]  # [{type: 'pdf/image/audio/video', path: '...', content: '...'}]
    
    input_file :  File
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

    basic_info :BasicInfo
    travel_plan :TravelPlan
    experience : Experience
    tags_info : ExperienceTagsOutputScehma
    evaluation : Eval
