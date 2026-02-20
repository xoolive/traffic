from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_quote = False

    for ch in text:
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
            continue

        if not in_quote:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
            elif ch == sep and depth == 0:
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _clean_value(value: str) -> str:
    value = value.strip().rstrip(",")
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]

    value = value.replace("{{", "").replace("}}", "")
    value = value.replace(r"\&", "&")
    value = value.replace("``", '"').replace("''", '"')

    latex_fixes = {
        r"{\'e}": "é",
        r"{\'E}": "É",
        r"{\'a}": "á",
        r"{\'i}": "í",
        r"{\'o}": "ó",
        r"{\`o}": "ò",
        r"{\'u}": "ú",
        r"{\^e}": "ê",
        r"{\^o}": "ô",
        r"{\^u}": "û",
        r"{\^i}": "î",
        r"{\`e}": "è",
        r"{\`a}": "à",
        r"{\v c}": "č",
        r"{\'c}": "ć",
        r"{\c c}": "ç",
        r"{\"u}": "ü",
        r"{\"o}": "ö",
        r"{\"a}": "ä",
        r"{\"e}": "ë",
        r"{\~n}": "ñ",
        r"--": "—",
        r"---": "—",
    }
    for src, dst in latex_fixes.items():
        value = value.replace(src, dst)

    value = re.sub(r"\s+", " ", value).strip()
    return value


def _parse_entry(entry_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in _split_top_level(entry_text):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip().lower()] = _clean_value(v)
    return out


def parse_bibtex(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    i = 0
    n = len(text)

    while i < n:
        at = text.find("@", i)
        if at == -1:
            break

        brace = text.find("{", at)
        if brace == -1:
            break

        etype = text[at + 1 : brace].strip().lower()
        depth = 1
        j = brace + 1
        while j < n and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1

        block = text[brace + 1 : j - 1].strip()
        if "," not in block:
            i = j
            continue

        key, fields = block.split(",", 1)
        data = _parse_entry(fields)
        data["entrytype"] = etype
        data["key"] = key.strip()
        entries.append(data)
        i = j

    return entries


def _format_author(name: str) -> str:
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{first} {last}".strip()
    return name.strip()


def _format_authors(authors: str) -> str:
    parts = [_format_author(a) for a in authors.split(" and ") if a.strip()]
    if not parts:
        return "Unknown authors"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    if len(parts) < 5:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    return f"{parts[0]} et al"  # noqa: RUF001


def _author_list(authors: str) -> list[str]:
    return [_format_author(a) for a in authors.split(" and ") if a.strip()]


def _venue_name(entry: dict[str, str]) -> str:
    return (
        entry.get("journal")
        # or entry.get("booktitle")
        # or entry.get("publisher")
        # or entry.get("school")
        # or entry.get("howpublished")
        or "Unknown"
    )


def _build_citation(entry: dict[str, str]) -> str:
    title = entry.get("title", "Untitled")
    authors = _format_authors(entry.get("author", ""))
    # year = entry.get("year", "n.d.")
    venue = (
        entry.get("journal")
        or entry.get("booktitle")
        or entry.get("publisher")
        or entry.get("school")
        or entry.get("howpublished")
        or entry.get("entrytype", "")
    )

    details: list[str] = []
    if entry.get("volume"):
        details.append(entry["volume"])
    if entry.get("number"):
        details.append(f"({entry['number']})")
    if entry.get("pages"):
        details.append(f"pp. {entry['pages']}")

    links: list[str] = []
    if doi := entry.get("doi"):
        doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
        links.append(f"[{doi}]({doi_url})")
    elif url := entry.get("url"):
        links.append(f"[↗]({url})")

    pieces = [
        f"**{title}**",
        authors.replace("{", "").replace("}", ""),
        f"*{venue}*",  # str(year)
    ]
    if details:
        pieces.append(" ".join(details))
    if links:
        pieces.append(" · ".join(links))

    citation = ". ".join(pieces) + "."

    return citation


def render_publications(entries: list[dict[str, str]]) -> str:
    by_year: dict[int, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        try:
            year = int(entry.get("year", "0"))
        except ValueError:
            year = 0
        by_year[year].append(entry)

    lines: list[str] = [
        "# Publications",
        "",
        "!!! warning",
        "    The following published papers claimed that they used the traffic "
        "library for their experiments.",
        "    The entries are automatically generated from ResearchGate and "
        "Google Scholar records. They may be incomplete or contain errors.",
        "",
        (
            f"_Last generated: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
        ),
        "",
    ]

    # Stats block
    type_counter = Counter(
        entry.get("entrytype", "unknown") for entry in entries
    )
    article_count = type_counter.get("article", 0)
    inproceedings_count = type_counter.get("inproceedings", 0)
    preprint_count = type_counter.get("preprint", 0) + type_counter.get(
        "misc", 0
    )
    book_chapter_count = type_counter.get("incollection", 0) + type_counter.get(
        "inbook", 0
    )

    author_counter: Counter[str] = Counter()
    for entry in entries:
        if any(
            x in entry.get("author", "")
            for x in ["Xavier Olive", "X. Olive", "Olive, Xavier"]
        ):
            continue
        for author in _author_list(entry.get("author", "")):
            author_counter[author] += 1

    venue_counter: Counter[str] = Counter(
        _venue_name(entry)
        for entry in entries
        if _venue_name(entry) != "Unknown"
    )

    lines.extend(
        [
            "## At a glance",
            "",
            f"- Total publications: **{len(entries)}**",
            f"- Articles: **{article_count}**",
            f"- Conference papers: **{inproceedings_count}**",
            f"- Preprints: **{preprint_count}**",
            f"- Book chapters: **{book_chapter_count}**",
            "",
            "### Top 5 citing authors",
            "This list excludes papers written by the main contributor of the "
            "library.",
            "",
        ]
    )

    for author, count in author_counter.most_common(5):
        lines.append(f"- {author}: **{count}**")

    lines.extend(["", "### Top 5 citing journals", ""])

    for venue, count in venue_counter.most_common(5):
        lines.append(f"- {venue}: **{count}**")

    lines.append("")

    for year in sorted(by_year.keys(), reverse=True):
        if year <= 0:
            continue
        lines.append(f"## {year}")
        lines.append("")

        items = sorted(
            by_year[year],
            key=lambda e: (
                e.get("month", ""),
                e.get("author", ""),
                e.get("title", ""),
            ),
            reverse=True,
        )

        for entry in items:
            lines.append(f"- {_build_citation(entry)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    docs_dir = root / "docs"
    bib_path = docs_dir / "assets" / "biblio.bib"
    out_path = docs_dir / "publications.md"

    entries = parse_bibtex(bib_path.read_text(encoding="utf-8"))
    out_path.write_text(render_publications(entries), encoding="utf-8")
    print(f"Generated {out_path} with {len(entries)} entries")


if __name__ == "__main__":
    main()
