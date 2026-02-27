import os
import pandas as pd
from openai import OpenAI

# load .env from project root
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


# load abstracts data
abstract_data_path = os.path.join("src", "data_collection", "sv_data_collection.xlsx")
data = pd.read_excel(abstract_data_path)
abstracts = list(data["Abstract"])

# LLM-generated abstract creation
def generate_abstract(abstract):
    response = client.chat.completions.create(model="gpt-5.2", messages=[{"role": "user", "content": \
    f"Titta på det här abstraktet och generera ett nytt med egna ord, av samma längd: {abstract}"}],)
    return response.choices[0].message.content

# loop through abstracts list
if __name__ == "__main__":
    new_abstracts = []
    for a in abstracts[0]:
        new_abstracts.append(generate_abstract(a))
    for n in new_abstracts:
        print(n)
    print("finished")


# create a response
#response = client.chat.completions.create(
#    model="gpt-5.2",
#    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
#)

#print(response.choices[0].message.content)
