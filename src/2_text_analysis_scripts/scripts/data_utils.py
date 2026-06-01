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
import pandas as pd


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
