"""
Informal: add LLM-generated comment column to reddit_comments.csv (OpenAI).
Reads 'question' per row, generates a short Swedish comment in response, writes to same CSV.
"""
import os
import pandas as pd
from openai import OpenAI

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root."
    )

client = OpenAI(api_key=_api_key)
csv_path = os.path.join(_root, "src", "data_collection", "reddit_comments.csv")

COMMENT_PROMPT = (
    "Svara på följande fråga med en kort, avslappnad svensk kommentar (som på ett forum). "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)


def generate_comment(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": COMMENT_PROMPT.format(question=question)}],
    )
    return (response.choices[0].message.content or "").strip()


if __name__ == "__main__":
    data = pd.read_csv(csv_path, encoding="utf-8")
    questions = list(data["question"])
    n = len(questions)

    data["Generated_OpenAI_Comment"] = pd.NA

    for i in range(n):
        q = questions[i]
        if pd.isna(q) or not str(q).strip():
            continue
        print(f"Processing {i + 1}/{n}...", end=" ", flush=True)
        try:
            data.loc[i, "Generated_OpenAI_Comment"] = generate_comment(str(q))
            print("done.")
        except Exception as e:
            print(f"Error: {e}")

    data.to_csv(csv_path, index=False, encoding="utf-8")
    print("Finished. Added column Generated_OpenAI_Comment to", csv_path)
