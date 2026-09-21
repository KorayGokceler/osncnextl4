#!/usr/bin/env python
'''
Render a weekly update script (markdown) as a PDF to read from while speaking.

    python presentation/make_update_pdf.py presentation/updates/2026-09-21.md

WHY THE MARKDOWN IS THE SOURCE
  A PDF with no source in the repository cannot be corrected, reviewed or
  diffed -- and a talk script gets corrected right up to the meeting.  So the
  markdown is what is versioned and the PDF is built from it.  `.gitignore`
  excludes `*.pdf` with named exceptions, and the built file is one of them,
  so the two cannot drift silently: a stale PDF shows up as a diff.

WHAT IT UNDERSTANDS
  Deliberately a small subset -- `# title`, a following plain line as the
  subtitle, `## heading`, paragraphs, `` `code` `` spans, and two-column
  `| key | value |` tables.  Enough for this format and nothing more; a real
  markdown engine would be a dependency for no gain here.

Dependencies: reportlab.  Nothing else, and no icetray.
'''
from __future__ import print_function

import os
import re
import argparse

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)


def styles():
    ss = getSampleStyleSheet()
    return dict(
        title=ParagraphStyle("t", parent=ss["Title"], fontSize=17, leading=21,
                             spaceAfter=2),
        sub=ParagraphStyle("s", parent=ss["Normal"], fontSize=9.5, leading=13,
                           textColor=colors.HexColor("#666666"),
                           spaceAfter=14),
        head=ParagraphStyle("h", parent=ss["Heading2"], fontSize=12.5,
                            leading=16, spaceBefore=13, spaceAfter=5,
                            textColor=colors.HexColor("#1f3d5c")),
        body=ParagraphStyle("b", parent=ss["Normal"], fontSize=10.8,
                            leading=16.5, spaceAfter=8),
        note=ParagraphStyle("n", parent=ss["Normal"], fontSize=9.5, leading=14,
                            leftIndent=8,
                            textColor=colors.HexColor("#444444"), spaceAfter=6),
    )


def inline(text):
    """Escape for reportlab, then turn `code` into a monospace span."""
    text = (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;"))
    return re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)


def blocks(md):
    """The markdown as a list of (kind, payload), in order."""
    out, para, rows = [], [], []

    def flush_para():
        if para:
            out.append(("body", " ".join(para)))
            del para[:]

    def flush_rows():
        if rows:
            out.append(("table", list(rows)))
            del rows[:]

    for raw in md.split("\n"):
        line = raw.rstrip()
        if line.startswith("| "):
            flush_para()
            cells = [c.strip() for c in line.strip("|").split("|")]
            # a separator row (|---|---|) carries no text
            if cells and not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells[:2])
            continue
        flush_rows()
        if line.startswith("# "):
            flush_para()
            out.append(("title", line[2:].strip()))
        elif line.startswith("## "):
            flush_para()
            out.append(("head", line[3:].strip()))
        elif not line:
            flush_para()
        else:
            para.append(line)
    flush_para()
    flush_rows()

    # A plain line straight after the title is the subtitle.
    for i, (kind, payload) in enumerate(out):
        if kind == "title" and i + 1 < len(out) and out[i + 1][0] == "body":
            out[i + 1] = ("sub", out[i + 1][1])
            break
    return out


def build(md_path, pdf_path):
    with open(md_path) as fh:
        md = fh.read()
    st = styles()
    story, seen_table = [], False

    for kind, payload in blocks(md):
        if kind == "title":
            story.append(Paragraph(inline(payload), st["title"]))
        elif kind == "sub":
            story.append(Paragraph(inline(payload), st["sub"]))
        elif kind == "head":
            # the reference block at the end is set apart from the script
            if not seen_table and payload.lower().startswith("numbers"):
                story += [Spacer(1, 16),
                          HRFlowable(width="100%", thickness=0.7,
                                     color=colors.HexColor("#cccccc"))]
            story.append(Paragraph(inline(payload), st["head"]))
        elif kind == "body":
            story.append(Paragraph(inline(payload),
                                   st["note"] if seen_table else st["body"]))
        elif kind == "table":
            seen_table = True
            data = [[Paragraph("<b>%s</b>" % inline(a), st["note"]),
                     Paragraph(inline(b), st["note"])] for a, b in payload]
            tbl = Table(data, colWidths=[52 * mm, 108 * mm])
            tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3,
                 colors.HexColor("#dddddd")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]))
            story += [tbl, Spacer(1, 8)]

    SimpleDocTemplate(pdf_path, pagesize=A4,
                      leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=20 * mm, bottomMargin=20 * mm,
                      title=os.path.basename(pdf_path),
                      author="").build(story)
    print("-> %s" % pdf_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("markdown")
    ap.add_argument("-o", "--output", default=None,
                    help="default: the markdown path with .pdf")
    args = ap.parse_args()
    out = args.output or os.path.splitext(args.markdown)[0] + ".pdf"
    build(args.markdown, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
