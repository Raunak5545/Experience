from typing import (
    Tuple,
    Union,
)

import requests
from fastapi import HTTPException


def get_content_type(file_url: str) -> Tuple[str, str]:
    """
    Detect the content type of a remote file (URL).
    Returns a tuple of (main_type, sub_type)
    """
    try:
        response = requests.head(file_url, allow_redirects=True, timeout=5)

        # Handle non-success status codes (e.g., 403 Forbidden, 404 Not Found)
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code, detail=f"Failed to access URL ({response.status_code}): {file_url}"
            )

        # Get content type
        content_type = response.headers.get("content-type", "application/octet-stream")

        # Ensure it’s valid
        if "/" not in content_type:
            raise HTTPException(status_code=400, detail=f"Invalid content-type: {content_type}")

        main_type, sub_type = content_type.split("/", 1)
        return main_type, sub_type

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail=f"Request timed out while accessing URL: {file_url}")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Network error while accessing URL: {file_url}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching URL: {str(e)}")


def prepare_content_message(content: str, file_url: str) -> Union[str, list]:
    """
    Prepare the content message for the LLM based on file type (URL only).
    Returns either a string for text content or a list for multimodal content.
    """
    main_type, sub_type = get_content_type(file_url)

    if main_type == "text":
        return content
    elif main_type == "image":
        return [{"type": "text", "text": content}, {"type": "image_url", "image_url": file_url}]
    elif main_type == "video":
        return [{"type": "text", "text": content}, {"type": "video_url", "video_url": file_url}]
    elif main_type == "audio":
        return [{"type": "text", "text": content}, {"type": "audio_url", "audio_url": file_url}]
    elif main_type == "application" and sub_type in [
        "pdf",
        "msword",
        "vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        return [{"type": "text", "text": content}, {"type": "document_url", "document_url": file_url}]
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {main_type}/{sub_type}")

