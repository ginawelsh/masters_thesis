"""
Base spaCy script for text analysis.
Install: pip install -r requirements.txt
Swedish: python -m spacy download sv_core_news_sm
English: python -m spacy download en_core_web_sm
"""
import spacy

# Use Swedish small model for thesis data; change to "en_core_web_sm" for English
MODEL = "sv_core_news_sm"


def load_nlp(model_name: str = MODEL):
    """Load spaCy pipeline. Download with: python -m spacy download <model_name>"""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Run: python -m spacy download {model_name}")
        raise


def process_text(nlp, text: str):
    """Return a Doc for the given text."""
    return nlp(text)


def main():
    nlp = load_nlp()

    sample = (
        "Syftet med denna uppsats är att belysa hur visionen om det narkotikafria "
        "samhället uppenbarar sig i diskurser som behandlar narkotikamissbruk."
    )
    doc = process_text(nlp, sample)

    print("Tokens:", [t.text for t in doc[:12]], "...")
    print("POS:", [(t.text, t.pos_) for t in doc[:8]])
    if doc.ents:
        print("Entities:", [(e.text, e.label_) for e in doc.ents])


if __name__ == "__main__":
    main()
