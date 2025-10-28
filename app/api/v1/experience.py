import traceback
from fastapi import APIRouter, UploadFile, File
import shutil, os, traceback, asyncio

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
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)




@router.post("/create-experience/multiple-files")
async def create_experience(files: list[UploadFile] = File(...)):
    results = []

    async def process_file(file: UploadFile):
        temp_file_path = f"/tmp/{file.filename}"
        try:
            # Save file asynchronously
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Run your processing (can also be made async if start_agentic_process supports it)
            res = await asyncio.to_thread(start_agentic_process, temp_file_path)

            return {"filename": file.filename, "experience": res}

        except Exception as e:
            traceback.print_exc()
            return {"filename": file.filename, "error": str(e)}

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Run all files concurrently
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}
