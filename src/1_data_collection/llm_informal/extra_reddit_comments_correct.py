"""
gather reddit comments for select quiz questions
"""

import csv
import os
import sys
from openai import OpenAI

# Project root for .env
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    "Svara på följande fråga med en svensk kommentar på 20-60 ord, i stil med en kommentar på ett svenskt forum. "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)

questions = ["Vad hände med Reddit Meetup Day? Det var ju under vår nationaldag har jag för mig. Tycker att det borde styras upp ifall det inte blev något :D", "Vad fick du i julklapp i år Sweddit?.", "Lumpen – har ni gjort den? Jag tycker att vi måste blåsa lite liv i denna reddit, så jag föreslår att vi börjar snacka om det. Gjorde själv inte militärtjänst, var upptagen med andra dumma saker vid den åldern. Dock ångrar jag det väldigt mycket, tror att den hade varit en upplevelse. Åsikter/erfarenheter?"]
comments = [""]

# generate AI-generated comment with comment prompt identified above
def generate_comment(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        # set up prompt generation, insert question into prompt slot
        messages=[{"role": "user", "content": COMMENT_PROMPT.format(question=question)}],
    )
    return (response.choices[0].message.content or "").strip()

def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(data_dir, "reddit_comments_openai_NOT_AVSLAPPNAD_extra")

    comment_rows = []
    for question in questions:
        print(question)
        for n in range(3):
            print(f"generating comment{n}")
            try:
                comment = generate_comment(question)
                comment_rows.append({
                    "question": question,
                    "comment": comment,
                })
            except Exception as e:
                print(f"  Error: {e}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "comment"])
        writer.writeheader()
        writer.writerows(comment_rows)

    print(f"Wrote {len(comment_rows)} comments to {out_path}")


if __name__ == "__main__":
    main()
