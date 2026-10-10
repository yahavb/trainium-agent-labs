#!/usr/bin/env python3
"""Check dashboard HTML structure against a comparison report, without a browser.

This is static validation only; it does not verify layout or JavaScript behavior.
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path


class Dashboard(HTMLParser):
    def __init__(self):
        super().__init__()
        self.data = []
        self.capture = False
        self.ids = []
        self.dots = []
        self.external = []
        self.fake = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "script" and attrs.get("id") == "attempt-data":
            self.capture = True
        if "dot" in attrs.get("class", "").split():
            self.dots.append(int(attrs["data-id"]))
        if "fake" in attrs.get("class", "").split():
            self.fake = True
        if tag in ("script", "img", "iframe", "link"):
            url = attrs.get("src") or attrs.get("href")
            if url and not url.startswith(("data:", "#")):
                self.external.append(url)

    def handle_endtag(self, tag):
        if tag == "script":
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.data.append(data)


def check(html_path, report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    parser = Dashboard()
    parser.feed(Path(html_path).read_text(encoding="utf-8"))
    details = json.loads("".join(parser.data))
    expected = report["attempts"]
    if len(details) != expected or sorted(parser.dots) != list(range(expected)):
        raise ValueError("Timeline/data count does not match report attempts")
    if len(parser.ids) != len(set(parser.ids)):
        raise ValueError("Duplicate HTML element ids")
    if not {"attempt-data", "detail", "tip"}.issubset(parser.ids):
        raise ValueError("Missing dashboard interaction elements")
    if parser.external:
        raise ValueError(f"Unexpected external resource dependencies: {parser.external}")
    if parser.fake:
        raise ValueError("Dashboard is flagged as fake/stub data")
    return {"status": "passed", "report_status": report["status"], "attempts": expected,
            "timeline_marks": len(parser.dots), "embedded_attempts": len(details),
            "visual_qa": "not performed; static checks only"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html", type=Path)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()
    print(json.dumps(check(args.html, args.report), indent=2))


if __name__ == "__main__":
    main()
