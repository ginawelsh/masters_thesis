import os
import pandas as pd
import openai
from openai import OpenAI

# load .env from project root (script is in src/llms/formal/)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# check openai api_key is set
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. In this terminal run:\n"
        "  set OPENAI_API_KEY=sk-your-key   (Windows, current session only)\n"
        "  export OPENAI_API_KEY=sk-your-key   (Linux/macOS)\n"
        "If you used setx, open a new terminal so it takes effect, or add OPENAI_API_KEY to a .env file in the project root."
    )
client = OpenAI(api_key=_api_key)

# input: human-collected abstracts with keywords
_human_formal_dir = os.path.join(_root, "src", "1_data_collection", "human_formal")
abstract_data_path = os.path.join(_human_formal_dir, "sv_human_collection_with_kws.csv")

# output: AI-generated titles and abstracts
_out_dir = os.path.join(_root, "src", "1_data_collection", "llm_formal", "abstracts")
out_csv_path = os.path.join(_out_dir, "sv_ai_generated_abstracts.csv")


def generate_title(title, keywords):
    """Generate an AI thesis title in Swedish given the human title and keywords."""
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content":
            f"Titta på titeln och nyckelorden och skapa en ny kandidatuppsatstitel med dina egna ord: {title, keywords}."}],
    )
    return response.choices[0].message.content


def generate_abstract(title, keywords):
    """Generate an AI abstract in Swedish given the human title and keywords."""
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content":
            f"Titta på titeln och nyckelorden och skapa en sammanfattning i kandidatuppsatsstil med dina egna ord: {title, keywords}."}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    data_csv = pd.read_csv(abstract_data_path, encoding="utf-8")

    human_titles = list(data_csv["Title"])
    human_abstracts = list(data_csv["Abstract"])
    keywords = list(data_csv["Keywords"])

    n = len(human_titles)
    print(f"Loaded {n} rows from {abstract_data_path}")

    rows = []
    for i in range(n):
        print(f"Processing {i + 1}/{n}...", end=" ", flush=True)

        ai_title = generate_title(human_titles[i], keywords[i])
        ai_abstract = generate_abstract(human_titles[i], keywords[i])

        rows.append({
            "Human_Thesis_Title": human_titles[i],
            "AI_Thesis_Title": ai_title,
            "Human_Abstract": human_abstracts[i],
            "AI_Abstract": ai_abstract,
        })

        print("done.")
        print(f"  AI title:    {ai_title[:80]}{'...' if len(ai_title) > 80 else ''}")
        print(f"  AI abstract: {ai_abstract[:80]}{'...' if len(ai_abstract) > 80 else ''}")
        print("-" * 60)

    out_df = pd.DataFrame(rows, columns=["Human_Thesis_Title", "AI_Thesis_Title", "Human_Abstract", "AI_Abstract"])
    out_df.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows to {out_csv_path}")
