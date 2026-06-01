"""
Visualise reddit_comments_openai_NOT_AVSLAPPNAD as an HTML table in the browser.
"""

import csv
import os
import webbrowser
import tempfile

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reddit_comments_openai_NOT_AVSLAPPNAD")


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def build_html(rows: list[dict]) -> str:
    def esc(text: str) -> str:
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )

    header_cells = "".join(f"<th>{esc(col)}</th>" for col in rows[0].keys()) if rows else ""

    body_rows = []
    for i, row in enumerate(rows):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        cells = "".join(f"<td>{esc(str(v))}</td>" for v in row.values())
        body_rows.append(f'<tr style="background:{bg}">{cells}</tr>')

    return f"""<!DOCTYPE html>
<html lang="sv">
<head>
  <meta charset="UTF-8">
  <title>reddit_comments_openai_NOT_AVSLAPPNAD</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      font-size: 13px;
      padding: 20px;
      color: #333;
    }}
    h1 {{
      font-size: 16px;
      margin-bottom: 12px;
    }}
    p.meta {{
      color: #666;
      margin-bottom: 16px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
    }}
    th {{
      background: #2c2c2c;
      color: #fff;
      padding: 8px 12px;
      text-align: left;
      position: sticky;
      top: 0;
    }}
    th:first-child  {{ width: 45%; }}
    th:last-child   {{ width: 55%; }}
    td {{
      padding: 8px 12px;
      border-bottom: 1px solid #e0e0e0;
      vertical-align: top;
      word-wrap: break-word;
    }}
    tr:hover td {{
      background: #eef4ff !important;
    }}
  </style>
</head>
<body>
  <h1>reddit_comments_openai_NOT_AVSLAPPNAD</h1>
  <p class="meta">{len(rows)} rows</p>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>"""


def main():
    rows = load_rows(DATA_FILE)
    html = build_html(rows)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()

    print(f"Wrote table to {tmp.name}")
    webbrowser.open(f"file:///{tmp.name.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
