"""Render docs/PROJECT_EXPLAINER.md to docs/PROJECT_EXPLAINER.pdf."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "PROJECT_EXPLAINER.md"
TARGET = ROOT / "docs" / "PROJECT_EXPLAINER.pdf"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


class ExplainerPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "PayGuard  |  Ashwath Bajaj  |  Interview explainer", align="left")
        self.ln(4)
        self.set_draw_color(210, 210, 210)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, str(self.page_no()), align="center")
        self.set_text_color(0, 0, 0)


def _clean(text: str) -> str:
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u00a0", " ")


def _styled_runs(text: str, size: float) -> list[tuple[str, str, float, str]]:
    runs: list[tuple[str, str, float, str]] = []
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            family, style, run_size, body = "DejaVu", "B", size, part[2:-2]
        elif part.startswith("`") and part.endswith("`"):
            family, style, run_size, body = "Mono", "", size - 0.5, part[1:-1]
        else:
            family, style, run_size, body = "DejaVu", "", size, part
        for word in re.findall(r"\S+\s*", body):
            runs.append((family, style, run_size, word))
    return runs


def _write_inline(pdf: ExplainerPDF, text: str, size: float, height: float) -> None:
    right = pdf.w - pdf.r_margin
    for family, style, run_size, word in _styled_runs(text, size):
        pdf.set_font(family, style, run_size)
        width = pdf.get_string_width(word)
        if pdf.get_x() + width > right and pdf.get_x() > pdf.l_margin + 1:
            pdf.ln(height)
            pdf.set_x(pdf.l_margin)
            word = word.lstrip()
            pdf.set_font(family, style, run_size)
        pdf.write(height, word)


def _paragraph(pdf: ExplainerPDF, text: str, size: float = 10.5, height: float = 5.2) -> None:
    pdf.set_x(pdf.l_margin)
    _write_inline(pdf, _clean(text), size, height)
    pdf.ln(height + 1.4)


def _heading(pdf: ExplainerPDF, line: str) -> None:
    level = len(line) - len(line.lstrip("#"))
    text = _clean(line[level:].strip())
    if level == 1:
        pdf.ln(2)
        pdf.set_font("DejaVu", "B", 18)
        pdf.multi_cell(0, 9, text)
        pdf.ln(1)
        return
    if level == 2:
        pdf.ln(2)
        pdf.set_font("DejaVu", "B", 14)
        pdf.multi_cell(0, 7, text)
        pdf.ln(1)
        return
    pdf.ln(1)
    pdf.set_font("DejaVu", "B", 12)
    pdf.multi_cell(0, 6, text)
    pdf.ln(0.6)


def _bullet(pdf: ExplainerPDF, text: str, numbered: str | None = None) -> None:
    indent = pdf.l_margin + 4
    pdf.set_x(indent)
    pdf.set_font("DejaVu", "", 10.5)
    marker = f"{numbered} " if numbered else "- "
    pdf.write(5.2, marker)
    _write_inline(pdf, _clean(text), 10.5, 5.2)
    pdf.ln(6)


def _code(pdf: ExplainerPDF, text: str) -> None:
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Mono", "", 8)
    for line in text.splitlines() or [""]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 4.2, _clean(line) if line else " ", fill=True)
    pdf.ln(1.5)


def _fit_cell(pdf: ExplainerPDF, text: str, width: float) -> str:
    label = _clean(INLINE.sub(lambda match: match.group(0).strip("*`"), text.strip()))
    if pdf.get_string_width(label) <= width - 1.2:
        return label
    while label and pdf.get_string_width(label + ".") > width - 1.2:
        label = label[:-1]
    return label.rstrip() + "."


def _table(pdf: ExplainerPDF, rows: list[list[str]]) -> None:
    if not rows:
        return
    columns = max(len(row) for row in rows)
    width = pdf.epw / columns
    for index, row in enumerate(rows):
        if pdf.get_y() > 270:
            pdf.add_page()
        pdf.set_font("DejaVu", "B" if index == 0 else "", 8)
        padded = row + [""] * (columns - len(row))
        for cell in padded:
            pdf.cell(width, 6, _fit_cell(pdf, cell, width), border=1)
        pdf.ln(6)
    pdf.ln(2)


def _blocks(lines: list[str]) -> list[tuple[str, object]]:
    blocks: list[tuple[str, object]] = []
    paragraph: list[str] = []
    code: list[str] = []
    in_code = False
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("p", " ".join(paragraph)))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(("table", [row[:] for row in table]))
            table.clear()

    for raw in lines:
        line = raw.rstrip("\n")
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            if in_code:
                blocks.append(("code", "\n".join(code)))
                code = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            flush_table()
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} or not cell for cell in cells):
                continue
            table.append(cells)
            continue
        flush_table()
        if stripped.startswith("#"):
            flush_paragraph()
            blocks.append(("h", stripped))
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            blocks.append(("ul", stripped[2:]))
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            flush_paragraph()
            blocks.append(("ol", (numbered.group(1), numbered.group(2))))
            continue
        paragraph.append(stripped)
    flush_paragraph()
    flush_table()
    if code:
        blocks.append(("code", "\n".join(code)))
    return blocks


def render(source: Path, target: Path) -> None:
    pdf = ExplainerPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 16, 16)
    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("Mono", "", str(FONT_DIR / "DejaVuSansMono.ttf"))
    pdf.set_title("PayGuard interview explainer")
    pdf.set_author("Ashwath Bajaj")
    pdf.add_page()

    for kind, payload in _blocks(source.read_text(encoding="utf-8").splitlines()):
        if kind == "h":
            _heading(pdf, str(payload))
        elif kind == "p":
            _paragraph(pdf, str(payload))
        elif kind == "ul":
            _bullet(pdf, str(payload))
        elif kind == "ol":
            number, text = payload  # type: ignore[misc]
            _bullet(pdf, str(text), numbered=f"{number}.")
        elif kind == "code":
            _code(pdf, str(payload))
        elif kind == "table":
            _table(pdf, payload)  # type: ignore[arg-type]

    target.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(target))


def main() -> None:
    render(SOURCE, TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
