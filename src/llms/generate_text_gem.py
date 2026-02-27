import os
from google import genai

# load .env from project root
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# Google GenAI (Gemini) uses GOOGLE_API_KEY — get a key from https://aistudio.google.com/apikey
_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise ValueError(
        "GOOGLE_API_KEY not set. Get a key from https://aistudio.google.com/apikey then:\n"
        "  set GOOGLE_API_KEY=your-key   (Windows, current session)\n"
        "  export GOOGLE_API_KEY=your-key   (Linux/macOS)\n"
        "Or add GOOGLE_API_KEY=your-key to the .env file in the project root."
    )

client = genai.Client(api_key=_api_key)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Explain how AI works in a few words",
)

print(response.text)
