"""Helpers for validating and repairing LLM JSON outputs.

Provides:
- validate_json(text, session_id) -> {ok: bool, obj/error}
- repair_json_with_llm(llm_client, raw_text, session_id, max_attempts=2) -> {ok: bool, obj/error}
"""
import json
import time
from typing import Any, Dict

from app.core.logging import logger
from langchain_core.messages import HumanMessage


def validate_json(text: str, session_id: str = "") -> Dict[str, Any]:
    """Attempt to parse `text` as JSON and return structured result."""
    bound = logger.bind(session_id=session_id)
    if not isinstance(text, str):
        return {"ok": False, "error": "not-a-string"}
    try:
        # strip possible code fences
        content = text.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        obj = json.loads(content)
        return {"ok": True, "obj": obj}
    except Exception as e:
        bound.debug("json_validation_failed", error=str(e), preview=text[:500])
        return {"ok": False, "error": str(e), "raw": text}


def repair_json_with_llm(llm_client, raw_text: str, session_id: str = "", max_attempts: int = 2) -> Dict[str, Any]:
    """Ask the LLM to repair malformed JSON. Returns parsed obj on success.

    This function issues a targeted prompt asking the LLM to return ONLY valid JSON.
    """
    bound = logger.bind(session_id=session_id)
    # small repair prompt - be explicit
    repair_prompt_template = (
        "The following text is intended to be JSON but may be malformed. "
        "Please extract and return ONLY valid JSON. Do not include any explanation or markdown fences.\n\n"
        "Malformed JSON:\n\n{malformed}\n\n"
        "Return valid JSON only."
    )

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            prompt = repair_prompt_template.format(malformed=raw_text)
            bound.info("repair_attempt", attempt=attempt)
            resp = llm_client.invoke([HumanMessage(content=prompt)], session_id=session_id)
            content = (resp.get("content") or "").strip()
            # try parse
            parsed = validate_json(content, session_id=session_id)
            if parsed.get("ok"):
                bound.info("repair_success", attempt=attempt)
                return {"ok": True, "obj": parsed.get("obj"), "attempts": attempt}
            bound.warning("repair_attempt_failed", attempt=attempt, preview=content[:500])
            raw_text = content  # try to repair the repaired output in next iteration
        except Exception as e:
            bound.error("repair_exception", attempt=attempt, error=str(e), exc_info=True)
    return {"ok": False, "error": "repair_failed", "attempts": attempt, "raw": raw_text}
import json
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from app.core.logging import logger


def _strip_code_fence(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    # remove triple backticks wrappers (```json ... ```)
    if s.startswith("```"):
        parts = s.split("```", 2)
        if len(parts) >= 2:
            # keep the content after the first fence
            s = parts[1]
            # if language token like `json` present, remove it
            if s.startswith("json"):
                s = s[len("json") :].strip()
    return s


def validate_json(raw_text: str, session_id: str = "") -> Dict[str, Any]:
    """Try to parse raw_text as JSON. Returns dict {'ok': bool, 'obj'|'error'}"""
    bound = logger.bind(session_id=session_id)
    try:
        cleaned = _strip_code_fence(raw_text)
        obj = json.loads(cleaned)
        return {"ok": True, "obj": obj}
    except Exception as e:
        bound.warning("json_parse_failed", error=str(e), preview=(raw_text or "")[:500])
        return {"ok": False, "error": str(e)}


def repair_json_with_llm(llm_client, raw_text: str, session_id: str = "", max_attempts: int = 2) -> Dict[str, Any]:
    """Ask the LLM to repair a malformed JSON string and re-validate.

    Returns {'ok': True, 'obj': parsed_obj, 'content': repaired_text} on success,
    otherwise {'ok': False, 'error': '...'}.
    """
    bound = logger.bind(session_id=session_id)
    for attempt in range(1, max_attempts + 1):
        repair_prompt = (
            "The model returned the following text that is supposed to be valid JSON.\n"
            "Please return a corrected JSON object only (no wrapper text).\n\n"
            "Response:\n" + (raw_text or "")
        )
        try:
            resp = llm_client.invoke([HumanMessage(content=repair_prompt)], session_id=session_id)
            content = resp.get("content")
            if not content:
                bound.warning("repair_no_content", attempt=attempt)
                continue
            parsed = validate_json(content, session_id=session_id)
            if parsed.get("ok"):
                bound.info("repair_success", attempt=attempt)
                return {"ok": True, "obj": parsed.get("obj"), "content": content}
            else:
                bound.warning("repair_attempt_failed", attempt=attempt, error=parsed.get("error"))
        except Exception as e:
            bound.error("repair_invoke_failed", attempt=attempt, error=str(e), exc_info=True)
    return {"ok": False, "error": "repair attempts failed"}
