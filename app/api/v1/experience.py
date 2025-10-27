from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.agents.workflow import start_agentic_process

router = APIRouter()

@router.post("")
async def create_experience(file: UploadFile = File(...)):
    temp_file_path = f"/tmp/{file.filename}"  # Temporary path
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        
        res = start_agentic_process(temp_file_path)
        return {"filename": file.filename, "experience": res}

    except Exception as e:
        # Handle exceptions
        return {"error": str(e)}

    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
