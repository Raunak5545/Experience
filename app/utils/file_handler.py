import mimetypes
import os
from typing import Dict, Tuple, Union, Any
import requests
from fastapi import HTTPException

def get_content_type(file_input: str, is_url: bool = False) -> Tuple[str, str]:
    """
    Detect the content type of a file or URL.
    Returns tuple of (main_type, sub_type)
    """
    if is_url:
        try:
            response = requests.head(file_input)
            content_type = response.headers.get('content-type', 'application/octet-stream')
            main_type, sub_type = content_type.split('/')
            return main_type, sub_type
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    else:
        # For local files
        content_type, _ = mimetypes.guess_type(file_input)
        if content_type is None:
            return "application", "octet-stream"
        main_type, sub_type = content_type.split('/')
        return main_type, sub_type

def prepare_content_message(content: str, file_input: str, is_url: bool = False) -> Union[str, list]:
    """
    Prepare the content message for the LLM based on file type.
    Returns either a string for text content or a list for multimodal content.
    """
    main_type, sub_type = get_content_type(file_input, is_url)
    
    if main_type == "text":
        return content
    elif main_type == "image":
        return [
            {"type": "text", "text": content},
            {"type": "image_url" if is_url else "image", "image_url" if is_url else "image": file_input}
        ]
    elif main_type == "video":
        return [
            {"type": "text", "text": content},
            {"type": "video_url" if is_url else "video", "video_url" if is_url else "video": file_input}
        ]
    elif main_type == "audio":
        return [
            {"type": "text", "text": content},
            {"type": "audio_url" if is_url else "audio", "audio_url" if is_url else "audio": file_input}
        ]
    elif main_type == "application" and sub_type in ["pdf", "msword", "vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        return [
            {"type": "text", "text": content},
            {"type": "document_url" if is_url else "document", "document_url" if is_url else "document": file_input}
        ]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {main_type}/{sub_type}")