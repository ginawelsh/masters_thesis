"""
Generates AI comment counterparts for flashback_qa.csv using OpenAI.
Reads the 'question' column, generates a short Swedish forum-style comment per row,
and writes the full CSV with a new 'Generated_OpenAI_Comment' column to
llm_comments/flashback_ai_comments.csv.
"""
import os
import pandas as pd
from openai import OpenAI

# import environment
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# get api key to be able to call gpt
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root."
    )

client = OpenAI(api_key=_api_key)

_src = os.path.join(_root, "src", "1_data_collection")
csv_path = os.path.join(_src, "human_comments", "flashback_qa.csv")
out_csv_path = os.path.join(_src, "llm_comments", "flashback_ai_comments.csv")

COMMENT_PROMPT = (
    "Svara på följande fråga med en svensk kommentar, i stil med en kommentar på ett svenskt forum. "
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
    n = len(data)

    data["Generated_OpenAI_Comment"] = pd.NA

    for i, row in data.iterrows():
        q = row["question"]
        if pd.isna(q) or not str(q).strip():
            continue
        print(f"Processing {i + 1}/{n}...", end=" ", flush=True)
        try:
            data.at[i, "Generated_OpenAI_Comment"] = generate_comment(str(q))
            print("done.")
        except Exception as e:
            print(f"Error: {e}")

    data.to_csv(out_csv_path, index=False, encoding="utf-8")
    print("Finished. Wrote", out_csv_path)
