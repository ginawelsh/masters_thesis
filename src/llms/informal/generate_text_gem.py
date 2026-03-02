"""
Informal: add LLM-generated comment column to reddit_comments.csv (Gemini).
Reads 'question' per row, generates a short Swedish comment in response, writes to same CSV.
"""
import os
import pandas as pd
from google import genai

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise ValueError(
        "GOOGLE_API_KEY not set. Get a key from https://aistudio.google.com/apikey "
        "and set it in the environment or .env in the project root."
    )

client = genai.Client(api_key=_api_key)
_data_dir = os.path.join(_root, "src", "data_collection")
csv_path = os.path.join(_data_dir, "reddit_comments.csv")
out_csv_path = os.path.join(_data_dir, "reddit_comments_gemini.csv")

COMMENT_PROMPT = (
    "Svara på följande fråga med en kort, avslappnad svensk kommentar (som på ett forum). "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)


def generate_comment(question):
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=COMMENT_PROMPT.format(question=question),
    )
    return (response.text or "").strip()


if __name__ == "__main__":
    data = pd.read_csv(csv_path, encoding="utf-8")
    questions = list(data["question"])
    n = len(questions)

    data["Generated_Gemini_Comment"] = pd.NA

    for i in range(n):
        q = questions[i]
        if pd.isna(q) or not str(q).strip():
            continue
        print(f"Processing {i + 1}/{n}...", end=" ", flush=True)
        try:
            data.loc[i, "Generated_Gemini_Comment"] = generate_comment(str(q))
            print("done.")
        except Exception as e:
            print(f"Error: {e}")

    data.to_csv(out_csv_path, index=False, encoding="utf-8")
    print("Finished. Wrote", out_csv_path)
