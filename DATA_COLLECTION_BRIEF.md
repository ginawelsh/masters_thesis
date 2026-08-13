# Data collection brief — Swedish blog posts and newspaper editorials

**Collect 75 of each.** Two spreadsheets. Details below fit on one page.

We're studying whether AI-written Swedish can be told apart from human-written Swedish.
You're collecting the **human** texts only.

---

## The three rules

**1. Published before 1 January 2017.**
AI writing tools became public after this, so anything later might not be fully human.
No date shown → skip it. "Last updated 2019" is not a publication date → skip it.

**2. Copy the text exactly.**
No fixing spelling, grammar, capitalisation, spacing or line breaks. Keep typos, slang,
dialect spellings and emoji. Those are the data. Paste as **plain text**, not formatted.

**3. Every text needs a title.**
We generate the AI comparison text from the title, so a text without one is unusable.

---

## What to collect

**Editorials** — opinion writing from Swedish newspapers: *ledare*, *krönika*, signed
commentary. Not straight news, not wire copy (TT), not letters to the editor.

**Blog posts** — personal Swedish blogs, someone writing in their own voice. Not corporate
or marketing blogs, not photo galleries or link roundups.

**Length:** whatever the piece is. Copy it in full — don't count or trim anything.

**Spread it around:** max 3 posts per blog (aim for 30+ different blogs), max 5 pieces per
publication.

**Skip:** sexually explicit material, or abuse of a named private person. Unsure → put it in
the `notes` column and leave it to Gina.

---

## Spreadsheet columns

`editorials.csv` — `id`, `title`, `standfirst`, `text`, `publication`, `author`,
`pub_date`, `url`, `notes`

`blogs.csv` — `id`, `title`, `text`, `blog_name`, `author`, `pub_date`, `url`, `notes`

`id` = `ed_001` / `bl_001` onwards. `pub_date` = `YYYY-MM-DD`, or `YYYY-MM` if the day
isn't shown. `author` = `unsigned` / `anonymous` if none.

**Save as CSV UTF-8** (Excel: *Save As → CSV UTF-8*). Reopen and check å ä ö look right —
the wrong setting turns them into garbage.

---

## Two last things

**Don't pad the numbers.** If only 60 editorials genuinely meet the date rule, deliver 60
and say so. A clean smaller set is much more useful than a full one with dates we can't
trust.

**Ask rather than guess** if you can't find enough, a source's terms are unclear, or
something sits on the line between news and opinion.
