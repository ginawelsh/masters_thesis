"""
gather reddit comments for select quiz questions
"""

import csv
import os
import sys
from openai import OpenAI

# Project root for .env
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# connect using API key
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root."
    )
client = OpenAI(api_key=_api_key)

# set general prompt for AI-generated questions

COMMENT_PROMPT = (
    "Svara på följande fråga med en kort, avslappnad svensk kommentar (som på ett forum). "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)

questions = ["Någon annan sate här som jobbar natt?", "Vad fick du i julklapp iår sweddit?", "Hur firar ni egentligen juldagen?? Det slog mig att juldagen inte har samma \"självklara\" traditioner som julafton, visst firar många julafton olika men tex julklapparna öppnas ju alltid den 24:e . Vad är juldagen för dig?"]

comments = [""]

# generate AI-generated comment with comment prompt identified above
def generate_comment(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": COMMENT_PROMPT.format(question=question)}],
    )
    return (response.choices[0].message.content or "").strip()

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(data_dir, "reddit_comments_openai_100.csv")

    for question in enumerate(questions):
        comment = generate_comment(question_text)
        comment_rows = []
        comment_rows.append({
            "link_id": link_id,
            "question": question_text,
            "comment": comment,
        })
    except Exception as e:
        print(f"  Error: {e}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "question", "comment"])
        writer.writeheader()
        writer.writerows(comment_rows)

    print(f"Wrote {len(comment_rows)} comments to {out_path} (target {TARGET_QUESTIONS})")


if __name__ == "__main__":
    main()
