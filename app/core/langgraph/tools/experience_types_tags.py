from langchain_core.tools import tool
from app.core.langgraph.data.experience_taxonomy import TAXONOMY


@tool
def get_full_experience_taxonomy() -> dict:
    """
    Returns the full experience taxonomy, including categories, types, and subtypes.
    Example:
    {
        "Activity": {
            "Adventure Sports": ["Base Jumping", "Bungee Jumping", ...],
            "Creative Workshops": ["Pottery", "Painting", ...],
            ...
        },
        "Food": {
            "Restaurants": ["Fine Dining", "Street Food", ...],
            ...
        },
        ...
    }
    """
    print("[get_full_experience_taxonomy] Called")

    if isinstance(TAXONOMY, dict):
        print(f"[get_full_experience_taxonomy] Returning taxonomy with {len(TAXONOMY)} top-level categories")
        return TAXONOMY

    print("[get_full_experience_taxonomy] TAXONOMY is not a dict")
    return {}
