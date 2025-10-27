"""API endpoints for itinerary generation."""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import json

from app.api.v1.auth import get_current_session
from app.core.langgraph.orchestrator.workflow import WorkflowOrchestrator
from app.core.langgraph.schemas.itinerary_schemas import ItineraryRequest
from app.core.logging import logger
from app.models.session import Session
from app.core.config import settings
from app.core.limiter import limiter

router = APIRouter()
orchestrator = WorkflowOrchestrator()


@router.post("/generate-itinerary")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS.get("itinerary", ["10/minute"])[0])
async def generate_itinerary(
    request: Request,
    itinerary_request: ItineraryRequest,
    session: Session = Depends(get_current_session),
) -> Dict[str, Any]:
    """Generate a complete itinerary from multi-modal input.
    
    Args:
        request: FastAPI request object
        itinerary_request: Itinerary generation request
        session: Current user session
        
    Returns:
        Generated itinerary with evaluation metrics
    """
    try:
        logger.info(f"Itinerary generation requested for session {session.id}")
        
        result = await orchestrator.process_request(
            itinerary_request,
            session.id,
            session.user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Itinerary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continue-itinerary/{session_id}")
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS.get("itinerary", ["10/minute"])[0])
async def continue_itinerary(
    request: Request,
    session_id: str,
    user_input: Dict[str, str],
    session: Session = Depends(get_current_session),
) -> Dict[str, Any]:
    """Continue itinerary generation with additional user input.
    
    Args:
        request: FastAPI request object
        session_id: Session ID to continue
        user_input: Additional user input
        session: Current user session
        
    Returns:
        Continued itinerary generation result
    """
    try:
        logger.info(f"Continuing itinerary for session {session_id}")
        
        result = await orchestrator.continue_with_input(
            session_id,
            user_input.get("input", "")
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Itinerary continuation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/itinerary-status/{session_id}")
async def get_itinerary_status(
    session_id: str,
    session: Session = Depends(get_current_session),
) -> Dict[str, Any]:
    """Get the current status of an itinerary generation.
    
    Args:
        session_id: Session ID to check
        session: Current user session
        
    Returns:
        Current status and state
    """
    try:
        state = await orchestrator.state_manager.get_state(session_id)
        
        if not state:
            return {"status": "not_found"}
        
        return {
            "status": "in_progress" if not state.final_output else "completed",
            "validated": state.validated,
            "classification": state.classification.value if state.classification else None,
            "has_basic_details": state.basic_details is not None,
            "has_itinerary": state.itinerary is not None,
            "has_evaluation": state.evaluation is not None,
            "errors": state.errors
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))