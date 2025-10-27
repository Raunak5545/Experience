from pydantic import BaseModel, Field
from typing import List, Optional

class Coordinates(BaseModel):
    type: str = Field(default="Point", description="The type of coordinates, fixed to 'Point' for GeoJSON compatibility.")
    coordinates: List[float] = Field(description="A list of two floats representing [longitude, latitude].")

class Location(BaseModel):
    city: str = Field(description="The city where the tour or experience takes place, extracted or inferred from the text.")
    state: str = Field(description="The state or region where the tour or experience takes place, extracted or inferred from the text.")
    country: str = Field(description="The country where the tour or experience takes place, extracted or inferred from the text.")
    placeName: str = Field(description="The full name of the place or tour as mentioned in the text.")
    coordinates: Coordinates = Field(description="Geographic coordinates of the location, with type 'Point' and [longitude, latitude].")

class FAQ(BaseModel):
    question: str = Field(description="A question relevant to the tour or experience, inspired by details in the source text.")
    answer: str = Field(description="A concise answer to the question, paraphrased or directly quoted from the source text.")

class BasicInfo(BaseModel):
    caption: str = Field(description="A short, engaging one-line caption summarizing the tour or experience.")
    summary: List[str] = Field(description="A list of bullet points summarizing key aspects of the tour or experience from the text.")
    location: Location = Field(description="Details of the location where the tour or experience takes place.")
    inclusion:Optional[ List[str]] = Field(description="A list of items or services included in the tour or experience, extracted from the text.")
    exclusion: Optional[List[str]] = Field(description="A list of items or services not included in the tour or experience, extracted from the text.")
    faq: List[FAQ] = Field(description="A list of question-answer pairs relevant to the tour or experience, inspired by the source text. Can be empty or contain any number of FAQs.")

class TypeValue(BaseModel):
    name: str = Field(..., description="Name of the activity")
    duration_in_hours: float = Field(..., alias="duration in hours", description="Duration of the activity in hours")

class ActivityType(BaseModel):
    name: str = Field(..., description="Type of item, e.g., 'activity'")
    value: TypeValue = Field(..., description="Detailed information about the activity")
    placename: Optional[str] = Field(None, description="Name of the place where the activity takes place")

class ScheduleItem(BaseModel):
    time: str = Field(..., description="Time of day for the activity, e.g., 'Morning'")
    timeline: str = Field(..., description="Timeline label for the schedule")
    description: List[str] = Field(..., description="List of description paragraphs or sentences")
    type: ActivityType = Field(..., description="Type and details of the scheduled activity")
    caption: Optional[str] = Field(None, description="Short caption for this schedule item")

class PlanItem(BaseModel):
    day: str = Field(..., description="Day number as string, e.g., '1'")
    caption: str = Field(..., description="Caption for the day’s plan")
    description: List[str] = Field(..., description="List of descriptive lines for the day")
    schedule: List[ScheduleItem] = Field(..., description="List of activities scheduled for the day")

class TravelPlan(BaseModel):
    plan: List[PlanItem] = Field(..., description="List of day-by-day travel plans")


class Experience(BasicInfo):
    plan_type:str =  Field(...,description="Type of plan, MANAGED/UNMANAGED")
    travel_plan: TravelPlan

