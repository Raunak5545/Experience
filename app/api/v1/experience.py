import asyncio
import os
import shutil
import tempfile
import traceback
from typing import (
    List,
    Optional,
)

from fastapi import (
    APIRouter,
    File,
    UploadFile,
)
from pydantic import (
    BaseModel,
    HttpUrl,
)

from app.core.langgraph.agents.workflow import start_agentic_process
from app.core.logging import logger

router = APIRouter()


class ExperienceInput(BaseModel):
    """Input model for URL-based experience creation"""

    file_url: HttpUrl


@router.post("/from-url")
async def create_experience_from_url(input_data: ExperienceInput):
    """Create experience from a URL input"""
    try:
        logger.info("create_experience_from_url_called", url=str(input_data.file_url))
        res = start_agentic_process(str(input_data.file_url), is_url=True)
        return {
            "url": str(input_data.file_url),
            "experience": res.get("experience"),
            "evaluation": res.get("evaluation"),
        }
    except Exception as e:
        logger.error("create_experience_from_url_failed", url=str(input_data.file_url), error=str(e), exc_info=True)
        return {"error": str(e)}


@router.post("")
async def create_experience(file: UploadFile = File(...)):
    """Create experience from a file upload"""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("create_experience_file_received", filename=file.filename, temp_path=temp_file_path)
        res = start_agentic_process(temp_file_path, is_url=False)
        logger.info("create_experience_file_processed", filename=file.filename)
        return {
            "filename": file.filename,
            "experience": res.get("experience"),
            "evaluation": res.get("evaluation"),
        }
    except Exception as e:
        logger.error("create_experience_file_failed", filename=file.filename, error=str(e), exc_info=True)
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
            logger.info("create_experience_multiple_file_saved", filename=file.filename)
            res = await asyncio.to_thread(start_agentic_process, temp_file_path)

            return {
                "filename": file.filename,
                "experience": res.get("experience"),
                "evaluation": res.get("evaluation"),
            }

        except Exception as e:
            logger.error("create_experience_multiple_file_failed", filename=file.filename, error=str(e), exc_info=True)
            return {"filename": file.filename, "error": str(e)}

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Run all files concurrently
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    return {"results": results}
