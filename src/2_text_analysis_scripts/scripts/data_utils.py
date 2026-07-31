"""
Robust CSV reader for files with unescaped double-quotes inside quoted fields
and embedded newlines inside abstract text.

Two fixes applied before pandas sees the content:
  1. Embedded newlines inside quoted fields are collapsed to a single space so
     multi-line abstract text does not confuse the record parser.
  2. Unescaped " inside a quoted field are escaped as "" so pandas does not
     misread them as field-closing quotes.

For fields that are not the last column, " followed by , is still treated as a
legitimate field close (e.g. quoted title fields).  For the last column only "
followed by newline / EOF closes the field, preventing "term", patterns in
AI-generated text from being split into extra fields.
"""
import io
import os
import sys
import pandas as pd

# force UTF-8 stdout so status prints (→, …, Swedish chars) can't crash on a cp1252 console.
# Applied on import so every analysis script that imports data_utils is covered.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---------------------------------------------------------------------------
# LLM prompt conditions
# ---------------------------------------------------------------------------
# Both corpora hold four independently generated LLM columns: a neutral baseline
# and three adversarial prompts. (An earlier version of this comment said the
# informal corpus had only one generated column -- that has not been true since the
# adversarial conditions were added.)

FORMAL_CONDITIONS = {
    "baseline": "Abstract_baseline",
    "human_like": "Abstract_human_like",
    "detector_aware": "Abstract_detector_aware",
    "detector_evasive": "Abstract_detector_evasive",
}
INFORMAL_CONDITIONS = {
    # NOT `generated_comment`. That column is gpt-4o-mini text, not GPT-5.2: 863 of
    # 1,149 texts trace to gpt-4o-mini output files and the remaining 286 have no
    # locatable source, while no generation cache contains a single baseline cell.
    # `comment_baseline` is a genuine GPT-5.2 baseline, regenerated 2026-07-31 over
    # the same 1,149 documents. See ../../../BASELINE_PROVENANCE.md.
    #
    # Switching to it changes the reported informal baseline results: the new text
    # averages 1,046 chars against the old 550 (human is 269), so effect sizes on
    # length-sensitive features get LARGER -- GPT-5.2 over-writes relative to humans
    # more than the gpt-4o-mini figures suggested.
    #
    # `generated_comment` is retained in the corpus and is still usable as a
    # gpt-4o-mini arm, provided it is labelled as such.
    "baseline": "comment_baseline",
    "human_like": "comment_human_like",
    "detector_aware": "comment_detector_aware",
    "detector_evasive": "comment_detector_evasive",
}
CONDITIONS_BY_DATASET = {"formal": FORMAL_CONDITIONS, "informal": INFORMAL_CONDITIONS}
CONDITION_ARG_CHOICES = list(FORMAL_CONDITIONS) + ["all"]


def add_condition_arg(parser, default="all"):
    """Register a --condition CLI flag (choices: the prompt conditions + 'all')."""
    parser.add_argument(
        "--condition", choices=CONDITION_ARG_CHOICES, default=default,
        help="Which LLM prompt condition(s) to analyse: baseline / human_like / detector_aware / all.",
    )


def resolve_conditions(dataset, requested="all"):
    """Return [(condition_name, llm_column), ...] to iterate over.

    Both registers carry all three prompt conditions (baseline / human_like /
    detector_aware), each mapped to its column in that register's CSV. 'all' returns
    every condition; otherwise just the requested one.
    """
    cols = CONDITIONS_BY_DATASET[dataset]
    names = list(cols) if requested == "all" else [requested]
    return [(name, cols[name]) for name in names]


def condition_tag(dataset, condition):
    """Output-filename tag: '<dataset>_<condition>' for every prompt condition
    (baseline / human_like / detector_aware), so the condition is always explicit
    in the output filename. None (no specific condition) falls back to the bare
    dataset name.
    """
    if condition is None:
        return dataset
    return f"{dataset}_{condition}"


def read_csv_robust(path, encoding="utf-8", **kwargs):
    with open(path, encoding=encoding, errors="replace", newline="") as f:
        raw = f.read()
    num_cols = _count_cols(raw)
    fixed    = _fix_quotes(raw, num_cols)
    df = pd.read_csv(io.StringIO(fixed), engine="python",
                     on_bad_lines="skip", **kwargs)
    # Warn if rows were dropped so the caller knows data is missing
    expected = fixed.count("\n") - 1   # rough: newlines minus header
    if len(df) < expected * 0.95:
        print(f"  Warning: {os.path.basename(path)} — loaded {len(df)} rows "
              f"(~{expected - len(df)} skipped due to malformed lines)")
    return df


def _count_cols(raw):
    end = raw.find("\n")
    header = raw[: end if end != -1 else len(raw)].rstrip("\r")
    return header.count(",") + 1


def _fix_quotes(raw, num_cols):
    out = []
    in_quoted = False
    field_num = 0
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]
        nxt = raw[i + 1] if i + 1 < n else None

        if in_quoted:
            if ch in ("\r", "\n"):
                # Collapse embedded newlines to a space; eat \r\n as one unit
                out.append(" ")
                if ch == "\r" and nxt == "\n":
                    i += 2
                else:
                    i += 1
            elif ch == '"':
                if nxt == '"':
                    out.append('""')
                    i += 2
                elif nxt in ("\r", "\n") or nxt is None:
                    out.append('"')
                    in_quoted = False
                    i += 1
                elif nxt == "," and field_num < num_cols - 1:
                    out.append('"')
                    in_quoted = False
                    i += 1
                else:
                    out.append('""')
                    i += 1
            else:
                out.append(ch)
                i += 1
        else:
            if ch == ",":
                out.append(",")
                field_num += 1
                i += 1
            elif ch in ("\r", "\n"):
                out.append(ch)
                field_num = 0
                i += 1
            elif ch == '"':
                out.append('"')
                in_quoted = True
                i += 1
            else:
                out.append(ch)
                i += 1

    return "".join(out)
