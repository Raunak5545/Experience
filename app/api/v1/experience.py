import os
import traceback
import tempfile
import asyncio
import shutil
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, HttpUrl
from app.core.langgraph.agents.workflow import start_agentic_process


router = APIRouter()


class ExperienceInput(BaseModel):
    """Input model for URL-based experience creation"""
    file_url: HttpUrl


@router.post("/from-url")
async def create_experience_from_url(input_data: ExperienceInput):
    """Create experience from a URL input"""
    try:
        res = start_agentic_process(str(input_data.file_url), is_url=True)
        return {
            "url": str(input_data.file_url),
            "experience": res.get("experience"),
            "evaluation": res.get("evaluation"),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@router.post("")
async def create_experience(file: UploadFile = File(...)):
    """Create experience from a file upload"""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        res = start_agentic_process(temp_file_path, is_url=False)
        return {
            "filename": file.filename,
            "experience": res.get("experience"),
            "evaluation": res.get("evaluation"),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.post("/create-experience/multiple-files")
async def create_experience(files: list[UploadFile] = File(...)):
    results = []

    async def process_file(file: UploadFile):
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file.filename)
        try:
            # Save file asynchronously
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Run your processing (can also be made async if start_agentic_process supports it)
            res = await asyncio.to_thread(start_agentic_process, temp_file_path)

            return {
                "filename": file.filename,
                "experience": res.get("experience"),
                "evaluation": res.get("evaluation"),
            }

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
