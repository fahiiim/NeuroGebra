"""Check Python fence syntax and links in a built MkDocs site (stdlib only)."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.links = []
        self.ids = set()
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])


def check_snippets():
    errors = []
    count = 0
    pattern = re.compile(
        r"^([ \t]*)```(?:python|py)[ \t]*\n(.*?)^\1```[ \t]*$", re.M | re.S
    )
    for path in sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]:
        content = path.read_text(encoding="utf-8")
        for match in pattern.finditer(content):
            count += 1
            line = content[: match.start(2)].count("\n") + 1
            try:
                ast.parse(textwrap.dedent(match[2]))
            except SyntaxError as exc:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line + (exc.lineno or 1) - 1}: {exc.msg}"
                )
    return count, errors


def check_links(site):
    site = site.resolve()
    pages = {
        p.resolve(): Page(p.read_text(encoding="utf-8")) for p in site.rglob("*.html")
    }
    errors = set()
    count = 0
    for path, page in pages.items():
        for link in page.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            count += 1
            decoded = unquote(parsed.path)
            target = (
                (
                    site / decoded.lstrip("/")
                    if decoded.startswith("/")
                    else path.parent / decoded
                ).resolve()
                if decoded
                else path
            )
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.add(f"{path.relative_to(site)}: missing target {link}")
            elif (
                parsed.fragment
                and target in pages
                and unquote(parsed.fragment) not in pages[target].ids
            ):
                errors.add(f"{path.relative_to(site)}: missing anchor {link}")
    return count, sorted(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site", type=Path, help="Also validate local links/anchors in this built site"
    )
    args = parser.parse_args()
    count, errors = check_snippets()
    print(f"Checked {count} Python code blocks")
    if args.site:
        if not (args.site / "index.html").is_file():
            errors.append(f"Missing built homepage: {args.site}")
        else:
            links, link_errors = check_links(args.site)
            print(f"Checked {links} local links and assets")
            errors.extend(link_errors)
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
