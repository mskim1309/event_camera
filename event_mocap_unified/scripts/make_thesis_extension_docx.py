#!/usr/bin/env python3
"""Append the generated Korean thesis-extension draft to a *new* DOCX file.

The original submission is never modified.  The output is deliberately a
review copy: equations and tables remain readable plain text so the author
can apply the institution's final Word styles after checking every claim.
"""

from __future__ import annotations

import argparse
import copy
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("w", W)


def qn(tag: str) -> str:
    return f"{{{W}}}{tag}"


def make_text_paragraph(text: str, heading_level: int | None = None) -> ET.Element:
    paragraph = ET.Element(qn("p"))
    if heading_level is not None:
        properties = ET.SubElement(paragraph, qn("pPr"))
        ET.SubElement(properties, qn("pStyle"), {qn("val"): f"Heading{heading_level}"})
    run = ET.SubElement(paragraph, qn("r"))
    text_node = ET.SubElement(run, qn("t"))
    text_node.set(f"{{{XML}}}space", "preserve")
    text_node.text = text
    return paragraph


def make_page_break() -> ET.Element:
    paragraph = ET.Element(qn("p"))
    run = ET.SubElement(paragraph, qn("r"))
    ET.SubElement(run, qn("br"), {qn("type"): "page"})
    return paragraph


def draft_paragraphs(markdown: str):
    """Small, dependency-free Markdown-to-readable-Word conversion."""

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or line == "***":
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            yield make_text_paragraph(heading.group(2), min(len(heading.group(1)), 3))
            continue
        if line.startswith("> "):
            yield make_text_paragraph(line[2:])
            continue
        if line.startswith("|"):
            # Keep data compact and reviewable; the final author can convert
            # this line to a Word table with the thesis template's styling.
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            yield make_text_paragraph("    ".join(cells))
            continue
        line = re.sub(r"^[-*]\s+", "• ", line)
        line = re.sub(r"^(\d+)\.\s+", r"\1. ", line)
        line = line.replace("**", "").replace("`", "")
        yield make_text_paragraph(line)


def append_draft(source_docx: Path, draft_path: Path, output_docx: Path) -> None:
    draft = draft_path.read_text(encoding="utf-8")
    with zipfile.ZipFile(source_docx, "r") as source:
        document = ET.fromstring(source.read("word/document.xml"))
        body = document.find(qn("body"))
        if body is None:
            raise ValueError("word/document.xml has no w:body")

        section = body.find(qn("sectPr"))
        if section is not None:
            body.remove(section)
        body.append(make_page_break())
        for paragraph in draft_paragraphs(draft):
            body.append(paragraph)
        if section is not None:
            body.append(section)

        output_docx.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_docx, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                if info.filename == "word/document.xml":
                    continue
                target.writestr(copy.copy(info), source.read(info.filename))
            target.writestr("word/document.xml", ET.tostring(document, encoding="utf-8", xml_declaration=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-docx", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-docx", type=Path, required=True)
    args = parser.parse_args()
    append_draft(args.source_docx, args.draft, args.output_docx)
    print(args.output_docx)


if __name__ == "__main__":
    main()
