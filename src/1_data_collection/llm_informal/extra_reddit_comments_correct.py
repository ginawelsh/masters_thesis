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
    "Svara på följande fråga med en kort, avslappnad svensk kommentar (som på ett forum). "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)

questions = ["Lumpen – har ni gjort den? Jag tycker att vi måste blåsa lite liv i denna reddit, så jag föreslår att vi börjar snacka om det. Gjorde själv inte militärtjänst, var upptagen med andra dumma saker vid den åldern. Dock ångrar jag det väldigt mycket, tror att den hade varit en upplevelse. Åsikter/erfarenheter?", "Snus marknadsförs i USA – kommer det funka den här gången?", "Vad fick du i julklapp i år Sweddit?.", "Det slog mig att juldagen inte har samma \"självklara\" traditioner som julafton, visst firar många julafton olika men tex julklapparna öppnas ju alltid den 24:e. Vad är juldagen för dig?", "Vad är några roliga svenska uttryck som du använder dig av?", "Bokhandlar i Göteborg? Kan någon rekommendera en bra bokhandeln i Göteborg med läsvänlig atmosfär? Letar efter någonstans att spendera några timmar i helgen över en kopp kaffe, utan att vara bråttom att köpa (liksom Barnes & Noble, kanske) ...", "Varför tappade de rödgröna makten till att börja med? Jag är 94a så är rätt ung, men jag undrar hur det kommer sig att de rödgröna har regerat Sverige under en väldigt lång period och S alltid har varit det traditionellt största partiet, så kommer Alliansen från ingenstans och plötsligt vinner, och får dessutom sitta kvar i två mandatperioder. Finns det någon speciell anledning till att Alliansen vann? Typ någon dålig reform som sossarna gjorde eller något liknande.", "Vad är egentligen Sverigedemokraterna?", "Vilka enskilda politiker ger bäst intryck?", "Vad hände med Reddit Meetup Day? Det var ju under vår nationaldag har jag för mig. Tycker att det borde styras upp ifall det inte blev något :D"]
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
    out_path = os.path.join(data_dir, "reddit_comments_openai_NEW.csv")

    comment_rows = []
    for question in questions:
        for _ in range(3):
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
