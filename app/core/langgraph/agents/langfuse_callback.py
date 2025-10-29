from langfuse._client.get_client import get_client
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()
langfuse_client = get_client()
