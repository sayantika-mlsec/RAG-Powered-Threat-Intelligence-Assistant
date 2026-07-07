"""
Single source of truth for the Gemini client. Router, judge, and generation
all import get_client() instead of building their own — one client, one
quota pool, one place to debug. Switching backends later means editing this
function, not hunting the codebase.
"""
from google import genai
import config

_client = None

def get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
        )
    return _client