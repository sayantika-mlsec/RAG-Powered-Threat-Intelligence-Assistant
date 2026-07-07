"""
Single source of truth for the Gemini client. Every caller (router, judge,
generation) imports get_client() instead of constructing its own — one
client, one quota pool, one place to debug.

LAZY READ, ON PURPOSE: GCP_PROJECT_ID / GCP_LOCATION are read from the
environment INSIDE get_client(), the first time it's actually called — not
at module import time. This is what keeps `import config` (and anything
that transitively imports it, like ingest.py) safe to run in CI, where no
GCP secrets exist. CI's tests/test.py covers ingestion + retrieval only, by
design, and never calls get_client() — so it must never be forced to pay
for a secret it doesn't need just because some other module got imported
along the way.

load_dotenv() is called here too, not borrowed from whichever caller
happens to import this file first — this module owns its own requirement.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

_client = None


def get_client():
    global _client
    if _client is None:
        # KeyError here is intentional and desired: a caller that actually
        # NEEDS a Gemini client with no GCP_PROJECT_ID set should fail loud,
        # right here, not silently construct a broken client. The whole
        # point of the fix is WHEN this line runs, not whether it can fail.
        project = os.environ["GCP_PROJECT_ID"]
        location = os.environ.get("GCP_LOCATION", "us-central1")
        _client = genai.Client(vertexai=True, project=project, location=location)
    return _client