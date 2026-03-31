"""
Multi-layer text analysis for Swedish formal (abstracts) and informal (e.g. Reddit) registers.

Three analysis families:
  1. Morphology & syntax — Stanza (UPOS/lemma/deps) + spaCy (POS, sentences, token stats)
  2. Discourse-oriented semantic space — Sentence-Transformers embeddings + 2D PCA scatter
  3. Content — BERTopic topic model + textstat readability-style metrics (English-oriented; interpret Swedish with care)

Usage:
  pip install -r requirements_analysis.txt
  python -m spacy download sv_core_news_lg
  python src/2_text_analysis_scripts/multi_layer_analysis.py --register formal
  python src/2_text_analysis_scripts/multi_layer_analysis.py --register informal
  python src/2_text_analysis_scripts/multi_layer_analysis.py --register both --max-docs 80

Optional:
  python ... --register both --skip-bertopic --skip-discourse-plots

Stanza Swedish uses tokenize+pos+lemma+depparse only (no mwt). If a previous run failed,
delete the broken folder under %LOCALAPPDATA%\\StanfordNLP\\stanza\\Cache\\...\\sv\\
and re-run so stanza.download can fetch models again.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd

RegisterName = Literal["formal", "informal"]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA = os.path.join(_ROOT, "1_data_collection")
_HUMAN_FORMAL = os.path.join(_DATA, "human_formal")
_HUMAN_INFORMAL = os.path.join(_DATA, "human_informal")

DEFAULT_SOURCES: dict[str, dict] = {
    "formal": {
        "csv_path": os.path.join(_HUMAN_FORMAL, "sv_human_collection_with_kws.csv"),
        "text_column": "Abstract",
        "context_columns": ["Year", "Level", "Title", "Topic Category"],
    },
    "informal": {
        "csv_path": os.path.join(_HUMAN_INFORMAL, "reddit_comments.csv"),
        "text_column": "comment",
        "prepend_column": "question",
        "context_columns": ["link_id"],
    },
}


@dataclass
class TextBundle:
    register: RegisterName
    texts: list[str]
    meta: pd.DataFrame


def _clean_text(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def load_corpus(
    register: RegisterName,
    csv_path: Optional[str] = None,
    text_column: Optional[str] = None,
    prepend_question: bool = True,
    max_docs: Optional[int] = None,
) -> TextBundle:
    cfg = DEFAULT_SOURCES[register]
    path = csv_path or cfg["csv_path"]
    tcol = text_column or cfg["text_column"]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8")
    if tcol not in df.columns:
        raise ValueError(f"Column '{tcol}' missing in {path}. Columns: {list(df.columns)}")

    texts: list[str] = []
    rows_meta: list[dict] = []
    prepend = cfg.get("prepend_column") if prepend_question else None

    for i, row in df.iterrows():
        body = _clean_text(row.get(tcol))
        if prepend and prepend in df.columns:
            q = _clean_text(row.get(prepend, ""))
            if q:
                body = f"{q}\n\n{body}" if body else q
        if not body:
            continue
        texts.append(body)
        meta = {"doc_index": len(texts) - 1, "register": register, "source_row": int(i)}
        for c in cfg.get("context_columns", []):
            if c in df.columns:
                meta[c] = row.get(c)
        rows_meta.append(meta)
        if max_docs is not None and len(texts) >= max_docs:
            break

    meta_df = pd.DataFrame(rows_meta)
    return TextBundle(register=register, texts=texts, meta=meta_df)


def _lazy_spacy():
    import spacy

    return spacy.load("sv_core_news_lg")


def _lazy_stanza():
    import stanza

    # Swedish has no MWT processor in Stanza; including "mwt" breaks the pipeline and
    # can leave a bad cache path (see UnsupportedProcessorError / missing default.pt).
    _processors = "tokenize,pos,lemma,depparse"
    try:
        stanza.download("sv", processors=_processors, verbose=False)
    except TypeError:
        try:
            stanza.download("sv", processors=_processors)
        except TypeError:
            try:
                stanza.download("sv", verbose=False)
            except TypeError:
                stanza.download("sv")
    return stanza.Pipeline(
        "sv",
        processors=_processors,
        use_gpu=False,
        verbose=False,
    )


def analyze_morphology_syntax(
    bundle: TextBundle,
    nlp_spacy=None,
    nlp_stanza=None,
) -> pd.DataFrame:
    """
    Per document: spaCy POS counts, sentence & token counts; Stanza UPOS counts, lemma count, dep-root counts sample.
    """
    if nlp_spacy is None:
        nlp_spacy = _lazy_spacy()
    if nlp_stanza is None:
        nlp_stanza = _lazy_stanza()

    rows = []
    for j, text in enumerate(bundle.texts):
        row: dict = {"doc_index": j, "register": bundle.register}
        doc_sp = nlp_spacy(text)
        pos_c = Counter(t.pos_ for t in doc_sp)
        row["n_tokens_spacy"] = len(doc_sp)
        row["n_sents_spacy"] = len(list(doc_sp.sents))
        row["pos_counts_spacy_json"] = json.dumps(dict(sorted(pos_c.items())), ensure_ascii=False)

        doc_st = nlp_stanza(text)
        upos_c: Counter = Counter()
        lemmas: list[str] = []
        dep_roots: Counter = Counter()
        for sent in doc_st.sentences:
            for w in sent.words:
                if w.upos:
                    upos_c[w.upos] += 1
                if w.lemma:
                    lemmas.append(w.lemma)
                if w.deprel:
                    dep_roots[w.deprel] += 1
        row["n_words_stanza"] = sum(upos_c.values())
        row["upos_counts_json"] = json.dumps(dict(sorted(upos_c.items())), ensure_ascii=False)
        row["n_unique_lemmas_stanza"] = len(set(lemmas))
        row["top_deprel_json"] = json.dumps(
            dict(dep_roots.most_common(12)), ensure_ascii=False
        )
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_discourse_embeddings(
    bundle: TextBundle,
    model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    output_dir: str | None = None,
    save_plot: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Sentence-Transformer embeddings per document; PCA to 2D for exploration (discourse / semantic space).
    Returns (summary DataFrame with paths, full embedding matrix).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        bundle.texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    pca = PCA(n_components=2, random_state=random_state)
    xy = pca.fit_transform(embeddings)

    summary = pd.DataFrame(
        {
            "doc_index": range(len(bundle.texts)),
            "register": bundle.register,
            "pca1": xy[:, 0],
            "pca2": xy[:, 1],
            "explained_var_pc1": pca.explained_variance_ratio_[0],
            "explained_var_pc2": pca.explained_variance_ratio_[1],
        }
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"embeddings_{bundle.register}.npy"), embeddings)
        summary.to_csv(
            os.path.join(output_dir, f"discourse_pca_{bundle.register}.csv"),
            index=False,
            encoding="utf-8",
        )
        if save_plot and len(summary) > 0:
            plt.figure(figsize=(8, 6))
            plt.scatter(summary["pca1"], summary["pca2"], alpha=0.6, s=22)
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.title(f"Discourse space (embeddings → PCA) — {bundle.register}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"discourse_scatter_{bundle.register}.png"),
                dpi=160,
            )
            plt.close()

    return summary, embeddings


def analyze_content_bertopic_textstat(
    bundle: TextBundle,
    embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    output_dir: str | None = None,
    min_topic_size: int = 5,
    skip_bertopic: bool = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    textstat metrics per doc (many are English-calibrated).
    BERTopic over documents using the same multilingual encoder.
    """
    import textstat
    from sentence_transformers import SentenceTransformer

    textstat_rows = []
    for j, text in enumerate(bundle.texts):
        # Heuristic syllables for Swedish: count vowel groups (very rough)
        def rough_syllables_sv(t: str) -> int:
            t = re.sub(r"[^a-zåäöA-ZÅÄÖ]+", " ", t.lower())
            if not t.strip():
                return 0
            return len(re.findall(r"[aeiouyåäö]+", t))

        syll = rough_syllables_sv(text)
        n_words = max(len(text.split()), 1)
        r = {
            "doc_index": j,
            "register": bundle.register,
            "char_count": len(text),
            "word_count": n_words,
            "rough_syllable_count_sv": syll,
        }
        for name in ("flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog", "smog_index"):
            try:
                r[f"textstat_{name}"] = getattr(textstat, name)(text)
            except Exception:
                r[f"textstat_{name}"] = np.nan
        textstat_rows.append(r)
    df_textstat = pd.DataFrame(textstat_rows)

    df_topics = None
    if not skip_bertopic and len(bundle.texts) >= min_topic_size:
        from bertopic import BERTopic

        emb = SentenceTransformer(embedding_model_name)
        topic_model = BERTopic(
            embedding_model=emb,
            language="multilingual",
            min_topic_size=min(min_topic_size, max(2, len(bundle.texts) // 10)),
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(bundle.texts)
        df_topics = pd.DataFrame(
            {
                "doc_index": range(len(bundle.texts)),
                "register": bundle.register,
                "topic": topics,
            }
        )
        if hasattr(probs, "shape") and len(probs.shape) == 2:
            df_topics["topic_prob_max"] = probs.max(axis=1)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            info = topic_model.get_topic_info()
            info.to_csv(
                os.path.join(output_dir, f"bertopic_topic_info_{bundle.register}.csv"),
                index=False,
                encoding="utf-8",
            )
            df_topics.to_csv(
                os.path.join(output_dir, f"bertopic_doc_topics_{bundle.register}.csv"),
                index=False,
                encoding="utf-8",
            )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_textstat.to_csv(
            os.path.join(output_dir, f"textstat_{bundle.register}.csv"),
            index=False,
            encoding="utf-8",
        )

    return df_textstat, df_topics


def run_register(
    register: RegisterName,
    output_base: str,
    max_docs: Optional[int],
    skip_bertopic: bool,
    skip_discourse_plots: bool,
) -> None:
    out_dir = os.path.join(output_base, register)
    os.makedirs(out_dir, exist_ok=True)

    bundle = load_corpus(register, max_docs=max_docs)
    bundle.meta.to_csv(os.path.join(out_dir, "corpus_meta.csv"), index=False, encoding="utf-8")

    print(f"[{register}] Loaded {len(bundle.texts)} documents → {out_dir}")

    print(f"[{register}] Morphology & syntax (spaCy + Stanza)...")
    morph_df = analyze_morphology_syntax(bundle)
    morph_df.to_csv(os.path.join(out_dir, "morphology_syntax.csv"), index=False, encoding="utf-8")

    print(f"[{register}] Discourse (Sentence Transformers + PCA)...")
    analyze_discourse_embeddings(
        bundle,
        output_dir=out_dir,
        save_plot=not skip_discourse_plots,
    )

    print(f"[{register}] Content (textstat + BERTopic)...")
    analyze_content_bertopic_textstat(
        bundle,
        output_dir=out_dir,
        skip_bertopic=skip_bertopic,
    )


def optional_classification_report(
    y_true: Iterable,
    y_pred: Iterable,
    labels: Optional[list] = None,
) -> str:
    """Use when you have gold labels in the CSV (e.g. sentiment column)."""
    from sklearn.metrics import classification_report

    return classification_report(y_true, y_pred, labels=labels)


def parse_args():
    p = argparse.ArgumentParser(description="Formal / informal multi-layer analysis")
    p.add_argument(
        "--register",
        choices=["formal", "informal", "both"],
        default="both",
        help="Which register to analyse",
    )
    p.add_argument(
        "--output-base",
        default=os.path.join(_DATA, "analysis_runs"),
        help="Directory for run outputs (subfolders formal/ informal/)",
    )
    p.add_argument("--max-docs", type=int, default=None, help="Cap documents per register")
    p.add_argument("--skip-bertopic", action="store_true", help="Skip BERTopic (faster)")
    p.add_argument(
        "--skip-discourse-plots",
        action="store_true",
        help="Skip PCA scatter PNG (still saves embeddings .npy and CSV)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    registers: list[RegisterName] = (
        ["formal", "informal"] if args.register == "both" else [args.register]
    )
    for reg in registers:
        run_register(
            reg,
            output_base=args.output_base,
            max_docs=args.max_docs,
            skip_bertopic=args.skip_bertopic,
            skip_discourse_plots=args.skip_discourse_plots,
        )
    print("Done.")


if __name__ == "__main__":
    main()
