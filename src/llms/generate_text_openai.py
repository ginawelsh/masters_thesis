import os
import pandas as pd
from openai import OpenAI

# load .env from project root
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# load abstracts data (CSV for reading/writing with new column)
_data_dir = os.path.join("src", "data_collection")
abstract_data_path = os.path.join(_data_dir, "sv_data_collection.xlsx")
data = pd.read_excel(abstract_data_path)
abstracts = list(data["Abstract"])
csv_path = os.path.join(_root, "src", "data_collection", "sv_data_collection_csv.csv")

# LLM-generated abstract creation
def generate_abstract(abstract):
    response = client.chat.completions.create(model="gpt-5.2", messages=[{"role": "user", "content": \
    f"Titta på det här abstraktet och generera ett nytt med egna ord, av samma längd: {abstract}"}],)
    return response.choices[0].message.content

# simple test: call generate_abstract to verify API works (with progress)
if __name__ == "__main__":
    # load CSV so we can add a new column and save back
    data_csv = pd.read_csv(csv_path, encoding="utf-8")
    abstracts_csv = list(data_csv["Abstract"])
    n_test = len(abstracts_csv) if abstracts_csv else 0

    # add new column (empty for rows we don't process)
    data_csv["Generated_Abstract"] = pd.NA

    if n_test == 0:
        test_abstract = "Detta är ett kort testabstrakt."
        print("Processing 1/1...", flush=True)
        result = generate_abstract(test_abstract)
        print("Generated abstract:", result)
        data_csv.loc[0, "Generated_Abstract"] = result
    else:
        for i in range(n_test):
            print(f"Processing {i + 1}/{n_test}...", end=" ", flush=True)
            result = generate_abstract(abstracts_csv[i])
            print("done.")
            print("Generated:", result[:80] + "..." if len(result) > 80 else result)
            print("-" * 40)
            data_csv.loc[i, "Generated_Abstract"] = result

    data_csv.to_csv(csv_path, index=False, encoding="utf-8")
    print("finished. Added column Generated_Abstract to", csv_path)


# create a response
#response = client.chat.completions.create(
#    model="gpt-5.2",
#    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
#)

#print(response.choices[0].message.content)
