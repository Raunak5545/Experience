"""Centralized wrapper for LLM and multimodal client calls.

Provides unified invoke(...) and generate_content(...) helpers with
retry/backoff, timing, and structured logging/metadata return values.
"""
import time
import json
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger


class LLMClient:
    def __init__(self, llm: Any = None, multimodal_client: Any = None, name: Optional[str] = None):
        self.llm = llm
        self.multimodal_client = multimodal_client
        self.name = name or getattr(llm, "model", "llm")

    def invoke(self, messages: List[Any], *, session_id: str = "", max_retries: Optional[int] = None, callbacks: Optional[List[Any]] = None, **kwargs) -> Dict[str, Any]:
        """Invoke the text/chat LLM with retries, timing and logging.

        Returns a dict with keys: content (str), metadata (dict)
        """
        max_retries = max_retries if max_retries is not None else settings.MAX_LLM_CALL_RETRIES
        bound = logger.bind(session_id=session_id, llm=self.name)

        attempt = 0
        start = time.time()
        while True:
            try:
                attempt += 1
                bound.debug("llm_invoke_attempt", attempt=attempt)
                # The ChatGoogleGenerativeAI API uses invoke(...)
                resp = self.llm.invoke(messages, config={"callbacks": callbacks, "langfuse_session_id": session_id, **kwargs})
                duration = time.time() - start
                content = getattr(resp, "content", None) or getattr(resp, "text", None) or ""
                metadata = getattr(resp, "response_metadata", {}) if hasattr(resp, "response_metadata") else {}
                bound.info("llm_invoke_success", duration_s=duration, attempt=attempt)
                return {"content": content, "metadata": metadata}
            except Exception as e:
                bound.error("llm_invoke_error", attempt=attempt, error=str(e), exc_info=True)
                if attempt >= max_retries:
                    bound.error("llm_invoke_giveup", attempt=attempt)
                    raise
                # simple exponential backoff with cap
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)

    def generate_content(self, uploaded_file: Any, prompt: str, *, session_id: str = "", model: Optional[str] = None, max_retries: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Use the multimodal client to generate content for a file + prompt.

        Returns dict with keys: content, metadata
        """
        if not self.multimodal_client:
            raise RuntimeError("multimodal_client not configured for generate_content")

        max_retries = max_retries if max_retries is not None else settings.MAX_LLM_CALL_RETRIES
        bound = logger.bind(session_id=session_id, llm=self.name, operation="generate_content")
        attempt = 0
        start = time.time()
        while True:
            try:
                attempt += 1
                bound.debug("multimodal_generate_attempt", attempt=attempt)
                resp = self.multimodal_client.models.generate_content(model=model or self.name, contents=[uploaded_file, prompt], **kwargs)
                duration = time.time() - start
                # genai returns a response with .text in many cases
                content = getattr(resp, "text", None) or getattr(resp, "content", "")
                metadata = {}
                bound.info("multimodal_generate_success", duration_s=duration, attempt=attempt)
                return {"content": content, "metadata": metadata}
            except Exception as e:
                bound.error("multimodal_generate_error", attempt=attempt, error=str(e), exc_info=True)
                if attempt >= max_retries:
                    bound.error("multimodal_generate_giveup", attempt=attempt)
                    raise
                backoff = min(2 ** attempt, 30)
                time.sleep(backoff)
import time
import json
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from app.core.logging import logger


class LLMClient:
    """A thin wrapper around LLM / multimodal clients that centralizes
    invoke calls, retries, timing and structured logging.

    This wrapper deliberately keeps payload handling generic: it returns a
    dict with `content` and `metadata` keys so callers can decide how to
    interpret the response.
    """

    def __init__(self, llm=None, multimodal_client=None, name: Optional[str] = None):
        self.llm = llm
        self.multimodal_client = multimodal_client
        self.name = name or getattr(llm, "model", "llm")

    def invoke(
        self,
        messages: list[HumanMessage],
        *,
        session_id: str = "",
        callbacks: Optional[list] = None,
        max_retries: int = 2,
        backoff: bool = True,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        bound = logger.bind(session_id=session_id, llm=self.name)
        attempt = 0
        start = time.time()
        while True:
            try:
                attempt += 1
                resp = self.llm.invoke(
                    messages,
                    config={"callbacks": callbacks, "langfuse_session_id": session_id, **(kwargs or {})},
                )
                duration = time.time() - start
                bound.info("llm_invoke_success", duration_s=duration, attempt=attempt)
                content = getattr(resp, "content", getattr(resp, "text", None))
                meta = getattr(resp, "response_metadata", None) or {}
                return {"content": content, "metadata": meta}
            except Exception as e:
                bound.error("llm_invoke_error", attempt=attempt, error=str(e), exc_info=True)
                if attempt > max_retries:
                    raise
                if backoff:
                    sleep_for = min(2 ** attempt, 30)
                    time.sleep(sleep_for)
                else:
                    time.sleep(1)

    def generate_content(
        self,
        uploaded_file: Any,
        prompt: str,
        *,
        session_id: str = "",
        model: Optional[str] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Wrapper for multimodal `generate_content` calls."""
        bound = logger.bind(session_id=session_id, llm="multimodal")
        attempt = 0
        start = time.time()
        while True:
            try:
                attempt += 1
                response = self.multimodal_client.models.generate_content(
                    model=model, contents=[uploaded_file, prompt]
                )
                duration = time.time() - start
                bound.info("multimodal_generate_success", duration_s=duration, attempt=attempt)
                content = getattr(response, "text", None)
                meta = {}
                return {"content": content, "metadata": meta}
            except Exception as e:
                bound.error("multimodal_generate_error", attempt=attempt, error=str(e), exc_info=True)
                if attempt > max_retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
