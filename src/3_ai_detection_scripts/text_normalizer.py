"""Uniform text normalizers for the CLEANED detection/feature runs.

Two independent normalizers live here:
  normalize_formal   -- markdown/keyword-footer confound in the formal register.
  normalize_informal -- scrape artefacts in the informal register (see below).

Purpose: neutralise the FORMATTING confound so a cross-generator formal detection
contrast measures Swedish, not markdown. GPT-5.6 and (unstripped) Mistral formal
carry markdown in ~86-90% of docs and a plain-text `Nyckelord:` footer in
34-75%; GPT-5.2 formal was already markdown-stripped at generation time and human
abstracts are essentially markdown-free. Applying ONE normalizer to every source
removes that asymmetry.

Deliberately conservative and defensible:
  1. strip_markdown  -- MIRRORS generation_pipeline.strip_markdown (kept byte-for-byte
     in sync): removes ATX heading markers, horizontal rules, bold/italic/code.
     Leaves bullet/numbered lists and all whitespace/newlines alone. This is exactly
     the normalization the pipeline already applied to the GPT columns, so re-applying
     it is a no-op there and brings Mistral into line.
  2. keyword footer  -- removes a trailing `Nyckelord:/Nyckelbegrepp:/Keywords:/Sökord:`
     line to end-of-text. This is thesis metadata, not abstract prose, and is the one
     structural artefact the followup builder flagged as surviving markdown stripping.

What it does NOT do (by design, to avoid deleting real content or creating a new bias):
  - It does not remove arbitrary "heading-looking" lines (too easy to delete a real
    sentence fragment). After markdown stripping, a `## Bakgrund` becomes a bare
    `Bakgrund` line; that residual is reported, not scrubbed.
  - It NEVER alters human text: `normalize_formal(text, is_human=True)` is a no-op.
    Human abstracts are read-only ground truth and are already markdown-free.
  - It does not touch length or refusals -- those are intrinsic generator behaviour,
    reported separately (see the length/refusal audit).
"""
import re

# --- 1. markdown stripping (in sync with generation_pipeline._MD_SUBS) --------------
_MD_SUBS = [
    (re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+"), ""),                     # ### Heading marker
    (re.compile(r"(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$"), ""),    # horizontal rule
    (re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", re.S), r"\1"),          # **bold**
    (re.compile(r"__(?=\S)(.+?)(?<=\S)__", re.S), r"\1"),              # __bold__
    (re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)"), r"\1"),  # *italic*
    (re.compile(r"`(?=\S)([^`\n]+?)(?<=\S)`"), r"\1"),                 # `code`
]

# --- 2. keyword footer (thesis metadata, not prose) --------------------------------
# GPT-5.6 emits it inline after the last sentence ("... påverka. Nyckelord: A, B, C"),
# NOT on its own line, so anchor to the word rather than to a newline. Removes from the
# keyword label to end-of-text.
_KEYWORD_FOOTER = re.compile(
    r"(?is)\s*\b(?:nyckelord|nyckelbegrepp|nyckelfraser|keywords?|sökord)\b\s*[:：].*\Z"
)

# --- 3. leading section-label heading -----------------------------------------------
# GPT-5.6 prefixes the abstract with a bare heading word ("Sammanfattning <text>").
# ONLY explicit section labels are stripped -- never sentence openers like "Denna"/"Jag"
# -- so a normal abstract that just starts with its first sentence is untouched.
_LEADING_HEADING = re.compile(
    r"(?i)^\s*(?:sammanfattning|sammandrag|abstrakt|abstract|syfte|inledning)\b\s*[:.]?\s+"
)


def strip_markdown(text: str) -> str:
    s = str(text)
    for rx, repl in _MD_SUBS:
        s = rx.sub(repl, s)
    return s


def normalize_formal(text, is_human: bool = False) -> str:
    """Return the normalized text. No-op for human ground truth."""
    s = str(text)
    if is_human:
        return s
    s = strip_markdown(s)
    s = _KEYWORD_FOOTER.sub("", s)
    s = _LEADING_HEADING.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# ===================================================================================
# INFORMAL register
# ===================================================================================
# Purpose: remove PullPush/forum SCRAPE artefacts that survive into the analysed
# comment text. Unlike normalize_formal, this one DOES apply to human text -- that is
# the whole point, since every artefact it targets is human-side only. It is applied
# uniformly to human and LLM columns so the treatment stays symmetric.
#
# Measured over consolidated_informal_comments_adversarial.csv (n=1149):
#   1. `&gt;` HTML entity      13 occurrences / 12 rows, human only, 0 in any LLM column.
#      PullPush returns HTML-escaped comment bodies and the scraper never unescaped
#      them. Consequence for the features: `&gt;` contributes a literal `;` to
#      punct_semicolon (13 of the 28 human semicolons in the whole corpus are this
#      artefact) and an alpha token `gt` to the TTR/MATTR/MTLD vocabulary.
#   2. Blockquote markers       4 literal `>` markers, human only.
#   3. `Mvh <name>` sign-offs   7 rows, human only. Formulaic sign-off plus, in 5 of
#      them, a forum display name or moderator role -- the only author identifiers
#      that survive anywhere in the corpus (no username column was ever collected).
#
# What it does NOT do by default, and why:
#   - It removes the blockquote MARKER, not the quoted LINE. Of the 15 human comments
#     that open with a blockquote, 10 are original content written in blockquote style
#     (a `svenskpolitik` thread where users riff on a quoted list, contributing
#     "Ett land där ..." lines that do NOT appear in the OP), and one is nothing but a
#     quotation, which line-deletion would reduce to an empty document. Deleting lines
#     would therefore destroy more real content than it removes contamination. Pass
#     drop_quoted=True to measure that variant; it is not the default.
#   - It does not drop the 2 moderator thread-locking notices ("Denna låses då det
#     finns en annan tråd med samma fråga"), which are thread management rather than
#     answers. Removing rows changes n and cascades into every paired test, so they
#     are reported for a separate exclusion decision, not silently dropped.

# 1. HTML entities. An explicit whitelist rather than html.unescape(): only `&gt;`
# actually occurs, and a whitelist cannot mangle a literal ampersand a user typed.
_ENTITIES = [
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&#0?39;|&apos;"), "'"),
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&amp;"), "&"),      # last, so &amp;gt; -> &gt; -> > is not re-entered
]

# 2. Blockquote markers. Two branches, both guarded so arithmetic comparisons survive:
# the char after the marker must not be a space, digit or `=`. This keeps `>10%` (LLM
# baseline), `Menu > Labels`, `Litteratur->Språk`, `låna >85%` and `<a href>` intact.
_QUOTE_MARKERS = [
    re.compile(r"(?m)^[ \t]*>[ \t]?(?=[^\s\d=])"),   # line-initial `>` / `> `
    re.compile(r"(?<=\s)>(?=[^\s\d=])"),             # mid-line `>` after whitespace
]

# 3. Trailing sign-off: `Mvh` to end of text, tail capped at 5 short words so the
# pattern can never eat a real sentence. End-anchored, so the LLM's metalinguistic
# mention of “mvh” mid-sentence is untouched.
_SIGNOFF = re.compile(
    r"(?is)\s*\bm\.?v\.?h\b[\s,.:/-]*(?:[\w.åäöÅÄÖ/-]+[ \t]?){0,5}\Z"
)

# Optional (drop_quoted=True): whole blockquote lines, marker included.
_QUOTED_LINES = re.compile(r"(?m)^[ \t]*>[ \t]?(?=[^\s\d=]).*$")


def normalize_informal(text, drop_quoted: bool = False) -> str:
    """Return the normalized comment. Applied to human AND LLM columns alike."""
    s = str(text)
    for rx, repl in _ENTITIES:
        s = rx.sub(repl, s)
    if drop_quoted:
        s = _QUOTED_LINES.sub("", s)
    else:
        for rx in _QUOTE_MARKERS:
            s = rx.sub("", s)
    s = _SIGNOFF.sub("", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s


# quick self-audit when run directly: how much each source changes
if __name__ == "__main__":
    import os
    import pandas as pd

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.dirname(here)
    ab = pd.read_csv(os.path.join(src, "1_data_collection", "llm_abstracts",
                                 "abstracts", "sv_abstracts_adversarial.csv"))
    mis = pd.read_csv(os.path.join(src, "2_text_analysis_scripts",
                                   "mistral_temperature_ab",
                                   "generated_corpus_mistral_temps.csv"))
    misf = mis[(mis["register"] == "formal") & (mis["temp"] == 1.0)]

    def md_rate(series, human=False):
        n = series.map(lambda t: normalize_formal(t, is_human=human))
        before = series.astype(str).str.contains(r"[*#`]|Nyckelord", regex=True).mean()
        after = n.str.contains(r"[*#`]|Nyckelord", regex=True).mean()
        changed = (n != series.astype(str).map(lambda t: normalize_formal(t, is_human=human) if False else str(t))).mean()
        return before, after

    print("source                     markdown/Nyckelord  before -> after")
    for label, s, human in [
        ("human (Abstract)",           ab["Abstract"], True),
        ("gpt5.2 baseline",            ab["Abstract_baseline"], False),
        ("gpt5.2 human_like",          ab["Abstract_human_like"], False),
        ("gpt5.2 detector_evasive",    ab["Abstract_detector_evasive"], False),
        ("mistral (all formal cond.)", misf["text"], False),
    ]:
        b, a = md_rate(s.dropna(), human)
        print(f"  {label:26s} {b:6.0%} -> {a:4.0%}")
