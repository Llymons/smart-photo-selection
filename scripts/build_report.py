"""Build a formatted Word (.docx) report from docs/design-report.md.

The Markdown file is the single source of truth.  This script implements the
small subset of Markdown used by the report (headings, paragraphs, bullet and
numbered lists, pipe tables, images, fenced code blocks) and adds a cover page,
a Word table of contents field, Chinese typography and page numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "design-report.md"
DOCX_PATH = ROOT / "docs" / "智能选片设计报告_期中.docx"

BODY_CN = "宋体"
BODY_EN = "Times New Roman"
HEAD_CN = "黑体"
MONO = "Consolas"
INK = RGBColor(0x1F, 0x29, 0x37)
GRAY = RGBColor(0x55, 0x5F, 0x6D)


# --------------------------------------------------------------------------- #
# Low-level XML helpers
# --------------------------------------------------------------------------- #
def set_run_font(run, cn: str = BODY_CN, en: str = BODY_EN, size: float | None = None,
                 bold: bool | None = None, color: RGBColor | None = None) -> None:
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_style_font(style, cn: str, en: str, size: float, bold: bool = False,
                   color: RGBColor = INK) -> None:
    style.font.name = en
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), en)
    rfonts.set(qn("w:hAnsi"), en)
    rfonts.set(qn("w:eastAsia"), cn)


def shade_cell(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def shade_paragraph(paragraph, hex_color: str) -> None:
    ppr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    ppr.append(shd)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9, color=GRAY)


def add_toc(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title = paragraph.add_run("目  录")
    set_run_font(title, cn=HEAD_CN, size=16, bold=True)

    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = "（在 Word 中右键“更新域”生成目录）"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(placeholder)
    run._r.append(fld_end)
    set_run_font(run, size=10.5, color=GRAY)

    settings = doc.settings.element
    update = OxmlElement("w:updateFields")
    update.set(qn("w:val"), "true")
    settings.append(update)


# --------------------------------------------------------------------------- #
# Markdown parsing helpers
# --------------------------------------------------------------------------- #
def parse_front_matter(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    if not lines or lines[0].strip() != "---":
        return {}, lines
    meta: dict[str, str] = {}
    end = 1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
        if ":" in lines[i]:
            key, value = lines[i].split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, lines[end + 1:]


def is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\|?\s*:?-{2,}", line.strip())) and "-" in line


def parse_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(paragraph, text: str, size: float = 12, cn: str = BODY_CN,
               en: str = BODY_EN, color: RGBColor = INK) -> None:
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            set_run_font(run, cn=cn, en=en, size=size, bold=True, color=color)
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            set_run_font(run, cn=MONO, en=MONO, size=size - 1.5, color=color)
        else:
            run = paragraph.add_run(part)
            set_run_font(run, cn=cn, en=en, size=size, color=color)


# --------------------------------------------------------------------------- #
# Document construction
# --------------------------------------------------------------------------- #
def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    set_style_font(normal, BODY_CN, BODY_EN, 12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)

    for name, size, before, after in (
        ("Heading 1", 16, 18, 12),
        ("Heading 2", 14, 14, 8),
        ("Heading 3", 12, 10, 6),
    ):
        style = doc.styles[name]
        set_style_font(style, HEAD_CN, BODY_EN, size, bold=True)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.line_spacing = 1.2

    for name in ("List Bullet", "List Number"):
        style = doc.styles[name]
        set_style_font(style, BODY_CN, BODY_EN, 12)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.left_indent = Cm(0.74)

    # Compact TOC entries so the table of contents fits on one page.
    for name in ("toc 1", "toc 2", "toc 3", "TOC 1", "TOC 2", "TOC 3"):
        try:
            style = doc.styles[name]
            set_style_font(style, BODY_CN, BODY_EN, 9.5)
            style.paragraph_format.line_spacing = 1.1
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(0)
        except KeyError:
            pass

    # Footer page number only; no running header text.
    add_page_number(section.footer.paragraphs[0])
    return doc


def add_cover(doc: Document, meta: dict[str, str]) -> None:
    for _ in range(2):
        doc.add_paragraph()
    course = doc.add_paragraph()
    course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = course.add_run(meta.get("course", "机器视觉"))
    set_run_font(run, cn=HEAD_CN, size=18, bold=True)

    report_type = doc.add_paragraph()
    report_type.alignment = WD_ALIGN_PARAGRAPH.CENTER
    report_type.paragraph_format.space_before = Pt(18)
    run = report_type.add_run(meta.get("report_type", "期中初步设计报告"))
    set_run_font(run, cn=HEAD_CN, size=16, color=GRAY)

    for _ in range(3):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(meta.get("title", "智能选片"))
    set_run_font(run, cn=HEAD_CN, size=36, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(10)
    run = subtitle.add_run(meta.get("subtitle", ""))
    set_run_font(run, cn=HEAD_CN, size=17, color=GRAY)

    for _ in range(5):
        doc.add_paragraph()

    info = [
        ("组　　号", meta.get("group", "第 __ 组")),
        ("组　　员", meta.get("members", "待填写")),
        ("代码托管", "GitHub"),
        ("仓库地址", meta.get("repo", "")),
        ("日　　期", meta.get("date", "")),
    ]
    table = doc.add_table(rows=len(info), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for row, (label, value) in zip(table.rows, info):
        row.cells[0].width = Cm(2.8)
        row.cells[1].width = Cm(11.0)
        p_label = row.cells[0].paragraphs[0]
        p_label.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_label.paragraph_format.line_spacing = 1.8
        p_label.paragraph_format.space_after = Pt(0)
        set_run_font(p_label.add_run(f"{label}："), size=12)

        p_value = row.cells[1].paragraphs[0]
        p_value.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_value.paragraph_format.line_spacing = 1.8
        p_value.paragraph_format.space_after = Pt(0)
        set_run_font(p_value.add_run(value), size=11 if label == "仓库地址" else 12)

    doc.add_page_break()


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    n_cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            cell = table.cell(r_idx, c_idx)
            text = row[c_idx] if c_idx < len(row) else ""
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            add_inline(p, text, size=10.5)
            if r_idx == 0:
                shade_cell(cell, "E8EEF7")
                for run in p.runs:
                    run.bold = True

    # Repeat header row across pages.
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(tbl_header)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def render_markdown(doc: Document, lines: list[str]) -> None:
    in_code = False
    code_lines: list[str] = []
    inserted_toc = False
    paragraph_count = 0

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                for code_line in code_lines:
                    p = doc.add_paragraph()
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.left_indent = Cm(0.6)
                    shade_paragraph(p, "F3F4F6")
                    run = p.add_run(code_line if code_line else " ")
                    set_run_font(run, cn=MONO, en=MONO, size=9)
                doc.add_paragraph().paragraph_format.space_after = Pt(2)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped == "<!-- pagebreak -->":
            doc.add_page_break()
            i += 1
            continue

        # Table: current line starts with | and next line is a separator.
        if stripped.startswith("|") and i + 1 < len(lines) and is_table_separator(lines[i + 1]):
            rows = [parse_table_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(parse_table_row(lines[i].strip()))
                i += 1
            add_table(doc, rows)
            paragraph_count = 0
            continue

        # Image.
        image_match = re.match(r"^!\[(.*?)\]\((.*?)\)\s*$", stripped)
        if image_match:
            alt, rel = image_match.groups()
            image_path = (MD_PATH.parent / rel).resolve()
            if image_path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                run = p.add_run()
                run.add_picture(str(image_path), width=Cm(15.4))
            if alt:
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap.paragraph_format.space_after = Pt(12)
                add_inline(cap, alt, size=10.5, color=GRAY)
            i += 1
            continue

        # Headings.
        heading_match = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1)) - 1  # ## -> Heading 1
            text = heading_match.group(2).strip()
            if not inserted_toc and text.startswith("1 "):
                doc.add_page_break()
                add_toc(doc)
                doc.add_page_break()
                inserted_toc = True
            heading = doc.add_heading(level=level)
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_inline(heading, text, size={1: 16, 2: 14, 3: 12}[level],
                       cn=HEAD_CN, color=INK)
            for run in heading.runs:
                run.bold = True
            paragraph_count = 0
            i += 1
            continue

        # Bullet / numbered list.
        if re.match(r"^[-*]\s+", stripped):
            text = re.sub(r"^[-*]\s+", "", stripped)
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, text)
            paragraph_count = 0
            i += 1
            continue
        numbered_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered_match:
            number, text = numbered_match.groups()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.first_line_indent = Cm(-0.74)
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_after = Pt(0)
            add_inline(p, f"{number}. {text}")
            paragraph_count = 0
            i += 1
            continue

        # Regular paragraph (one Markdown line = one paragraph).
        p = doc.add_paragraph()
        if re.match(r"^\[\d+\]", stripped):
            # Reference entry: hanging indent, no first-line indent.
            p.paragraph_format.left_indent = Cm(0.74)
            p.paragraph_format.first_line_indent = Cm(-0.74)
        else:
            p.paragraph_format.first_line_indent = Pt(24)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(0)
        add_inline(p, stripped)
        paragraph_count += 1
        i += 1


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    meta, body = parse_front_matter(lines)

    doc = setup_document()
    doc.core_properties.title = f"{meta.get('title', '')}：{meta.get('subtitle', '')}"
    doc.core_properties.author = meta.get("members", "Smart Photo Selection Team")
    doc.core_properties.subject = meta.get("course", "机器视觉")

    add_cover(doc, meta)
    render_markdown(doc, body)

    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(DOCX_PATH)
    print(f"Wrote {DOCX_PATH}")


if __name__ == "__main__":
    main()
