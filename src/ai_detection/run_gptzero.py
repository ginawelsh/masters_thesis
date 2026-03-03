"""
Base script for AI detection with GPTZero API.
Install: pip install -r requirements.txt
API key: https://app.gptzero.me/app/api — set GPTZERO_API_KEY in .env or environment.
"""
import os
from typing import Optional

import requests

try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = "https://api.gptzero.me/v2/predict/text"


def get_api_key():
    """Load GPTZero API key from environment or .env."""
    key = os.environ.get("GPTZERO_API_KEY")
    if not key:
        raise ValueError(
            "GPTZERO_API_KEY not set. Get a key from https://app.gptzero.me/app/api "
            "and set it in the environment or in a .env file in the project root."
        )
    return key.strip()


def detect_text(text: str, api_key: Optional[str] = None, version: Optional[str] = None):
    """
    Send text to GPTZero for AI detection.
    Returns the JSON response (e.g. overall score, sentence-level results).
    """
    key = api_key or get_api_key()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": key,
    }
    payload = {"document": text}
    if version:
        payload["version"] = version

    response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    api_key = get_api_key()

    sample = (
        "Syftet med denna uppsats är att belysa hur visionen om det narkotikafria "
        "samhället uppenbarar sig i diskurser som behandlar narkotikamissbruk."
    )

    print("Sending sample text to GPTZero...")
    result = detect_text(sample, api_key=api_key)
    print("Response:", result)


if __name__ == "__main__":
    main()
