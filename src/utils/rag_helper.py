import os
import requests
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
load_dotenv()

RAG_REGISTRY_URL = os.getenv("RAG_REGISTRY_URL")

if not RAG_REGISTRY_URL:
    raise ValueError("RAG_REGISTRY_URL not set in .env")


def get_rag_base_url():
    try:
        res = requests.get(RAG_REGISTRY_URL, timeout=10)
        res.raise_for_status()
        data = res.json()
        return data.get("base_url").rstrip("/")
    except Exception as e:
        raise Exception(f"Failed to fetch RAG base URL: {e}")


def call_rag(endpoint: str, text: str, case_id: str):
    base_url = get_rag_base_url()
    logger.info(f"Calling RAG at {base_url} with endpoint {endpoint}")
    url = f"{base_url}/{endpoint}"

    res = requests.post(
        url,
        data={
            "text": text,
            "case_id": case_id
        },
        timeout=120
    )

    res.raise_for_status()
    return res.json()