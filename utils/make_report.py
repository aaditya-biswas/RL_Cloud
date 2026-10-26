"""Build the two-page submission report as a PDF.

The layout is two columns on A4, the way the assignment asks for: a full-width
title block, then two text frames per page with the content flowing from the
first to the second and on to the next page. That is ReportLab's frame
machinery, which keeps figures, tables and captions together; the HTML/CSS
route through WeasyPrint was tried first and cannot place images inside CSS
columns (it drops them), so it is not used here.

Two artefacts come out of a run:

* ``report/fig_reward.png`` - the one figure that is not in ``plots/``, because
  it is about the *shape of the reward* rather than about a run; and
* ``report/AI_Assignment2_Report.pdf`` - the report itself.

The text lives in :mod:`utils.report_content`; this module only decides how it
is set.

Usage
-----
    python utils/make_report.py              # figure + PDF
    python utils/make_report.py --check      # ... and fail if the body spills past 2 pages
    python utils/make_report.py --skip-figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import HRFlowable  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python utils/make_report.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import report_content as content  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "report"
PDF_PATH = REPORT_DIR / "AI_Assignment2_Report.pdf"
FIGURE_PATH = REPORT_DIR / "fig_reward.png"

# ---- reward v3 constants; keep in step with env/design_wrappers.py ----------
RAMP_START_MS = 150.0   # penalty starts above the 150 ms "healthy" band
SLA_MS = 200.0          # the acceptance threshold
CAP_AT_MS = 400.0       # 4 units/ms from 150 ms hits the 1,000-unit cap here
CAP_UNITS = 1_000.0
SLOPE = 4.0             # units per ms over the ramp
POD_STEP_COST = 1.5     # reward units per pod per 10 s step
STEPS_PER_DAY = 8_640
POD_DAY = POD_STEP_COST * STEPS_PER_DAY  # 12,960 units

DPI = 200
PAGE_W, PAGE_H = A4
MARGIN_X = 12 * mm
MARGIN_TOP = 10.5 * mm
MARGIN_BOTTOM = 9 * mm
COL_GAP = 6.2 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X
COL_W = (CONTENT_W - COL_GAP) / 2
FRAME_H = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM

INK = colors.HexColor("#111111")
BLUE = colors.HexColor("#1a3a5c")
DEEP = colors.HexColor("#12263a")
RED = colors.HexColor("#b03a2e")
GREY = colors.HexColor("#444444")
RULE = colors.HexColor("#dde3e9")

FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
FONTS = {
    "Body": "LiberationSerif-Regular.ttf",
    "Body-Bold": "LiberationSerif-Bold.ttf",
    "Body-Italic": "LiberationSerif-Italic.ttf",
    "Body-BoldItalic": "LiberationSerif-BoldItalic.ttf",
    "Head": "LiberationSans-Bold.ttf",
    "Head-Reg": "LiberationSans-Regular.ttf",
    "Head-Italic": "LiberationSans-Italic.ttf",
    "Mono": "LiberationMono-Regular.ttf",
}


def register_fonts() -> None:
    """Register the Liberation faces so the PDF carries the fonts it shows."""
    for name, filename in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Body-Bold",
        italic="Body-Italic", boldItalic="Body-BoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "Head", normal="Head-Reg", bold="Head", italic="Head-Italic", boldItalic="Head",
    )
    pdfmetrics.registerFontFamily(
        "Mono", normal="Mono", bold="Mono", italic="Mono", boldItalic="Mono",
    )



def _style() -> None:
    """Match ``utils/make_plots.py`` so the report figure looks like the rest."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "legend.frameon": False,
    })


def make_reward_figure(path: Path = FIGURE_PATH) -> Path:
    """Two panels: the penalty is flat in depth, linear in count."""
    _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.65))

    # -- left: one step's penalty as a function of its latency ---------------
    latency = np.linspace(0.0, 1_000.0, 2_001)
    penalty = np.minimum(CAP_UNITS, SLOPE * np.maximum(0.0, latency - RAMP_START_MS))
    left.plot(latency, penalty, color="#b03a2e", lw=2.0, zorder=3)
    left.axvspan(SLA_MS, 1_000.0, color="#b03a2e", alpha=0.07, zorder=0)
    left.axvline(SLA_MS, color="#1a3a5c", ls="--", lw=1.0)
    left.axvline(CAP_AT_MS, color="#666666", ls=":", lw=1.0)
    left.axhline(CAP_UNITS, color="#666666", ls=":", lw=1.0)
    left.annotate("SLA 200 ms\n(200 units)", xy=(SLA_MS, 200), xytext=(230, 430),
                  fontsize=7, color="#1a3a5c",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color="#1a3a5c"))
    left.annotate("cap: 400 ms and 3,000 ms\ncost the same 1,000",
                  xy=(700, CAP_UNITS), xytext=(400, 640), fontsize=7, color="#444444",
                  arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"))
    left.text(290, 110, "slope 4 units / ms", fontsize=7, color="#b03a2e")
    left.set_xlim(0, 1_000)
    left.set_ylim(0, 1_180)
    left.set_xlabel("latency in the step (ms) - flat out to the 3,000 ms clamp")
    left.set_ylabel("penalty (reward units)")
    left.set_title("(a) one breached step: flat in depth", fontsize=8.5, loc="left")

    # -- right: total penalty as a function of how many steps breach ---------
    breaches = np.arange(0, 17)
    right.plot(breaches, CAP_UNITS * breaches, color="#b03a2e", lw=2.0,
               label="breach penalty = 1,000 x N")
    right.axhline(POD_DAY, color="#1a3a5c", ls="--", lw=1.2,
                  label=f"one pod-day = {POD_DAY:,.0f}")
    break_even = POD_DAY / CAP_UNITS
    right.plot([break_even], [POD_DAY], "o", ms=5, color="#1a3a5c", zorder=4)
    right.annotate(f"break-even:\n{break_even:.0f} breaches / day",
                   xy=(break_even, POD_DAY), xytext=(0.8, 14_000), fontsize=7,
                   color="#1a3a5c",
                   arrowprops=dict(arrowstyle="->", lw=0.8, color="#1a3a5c"))
    right.set_xlim(0, 16)
    right.set_ylim(0, 17_500)
    right.set_xlabel("breached steps (one day)")
    right.set_ylabel("cost (reward units)")
    right.set_title("(b) many breached steps: linear in count", fontsize=8.5, loc="left")
    right.legend(loc="lower right", fontsize=7)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_styles() -> dict[str, ParagraphStyle]:
    """Every style the report uses, in one place."""
    body_kwargs = dict(fontName="Body", fontSize=8.2, leading=10.2, textColor=INK,
                       alignment=TA_JUSTIFY, hyphenationLang="en_US")
    return {
        "kicker": ParagraphStyle("kicker", fontName="Head-Reg", fontSize=6.8, leading=8.2,
                                 textColor=BLUE),
        "title": ParagraphStyle("title", fontName="Head", fontSize=15.5, leading=16.6,
                                textColor=DEEP),
        "subtitle": ParagraphStyle("subtitle", fontName="Body-Italic", fontSize=8.9,
                                   leading=10.6, textColor=colors.HexColor("#333333")),
        "meta": ParagraphStyle("meta", fontName="Head-Reg", fontSize=7.5, leading=9.4,
                               textColor=colors.HexColor("#1b1b1b")),
        "body": ParagraphStyle("body", spaceAfter=2.6, **body_kwargs),
        "abstract": ParagraphStyle("abstract", fontName="Body", fontSize=8.05, leading=10.0,
                                   textColor=INK, alignment=TA_JUSTIFY,
                                   hyphenationLang="en_US"),
        "h2": ParagraphStyle("h2", fontName="Head", fontSize=8.8, leading=10.4, textColor=BLUE,
                             spaceBefore=5.0, spaceAfter=2.1, keepWithNext=1),
        "eq": ParagraphStyle("eq", fontName="Body", fontSize=8.15, leading=10.6,
                             alignment=TA_CENTER, textColor=INK, spaceBefore=1.5,
                             spaceAfter=4.5, backColor=colors.HexColor("#f7f8fa"),
                             borderPadding=(4, 5, 4, 5)),
        "caption": ParagraphStyle("caption", fontName="Body", fontSize=6.9, leading=8.35,
                                  textColor=colors.HexColor("#2c2c2c"), alignment=TA_LEFT,
                                  spaceBefore=2.2, spaceAfter=1.0),
        "cell": ParagraphStyle("cell", fontName="Body", fontSize=7.0, leading=8.25,
                               textColor=INK, alignment=TA_LEFT),
        "cell-muted": ParagraphStyle("cell-muted", fontName="Body", fontSize=7.0, leading=8.25,
                                     textColor=GREY, alignment=TA_LEFT),
        "head": ParagraphStyle("head", fontName="Head", fontSize=6.8, leading=8.2,
                               textColor=BLUE, alignment=TA_LEFT),
        "ref": ParagraphStyle("ref", fontName="Body", fontSize=7.1, leading=8.7, textColor=INK,
                              alignment=TA_JUSTIFY, leftIndent=10, firstLineIndent=-10,
                              spaceAfter=1.8, hyphenationLang="en_US"),
    }



def header_flowables(styles: dict) -> list:
    """The full-width title block that sits above the first page's columns."""
    meta = content.META
    return [
        Paragraph(meta["kicker"], styles["kicker"]),
        Spacer(1, 2.0),
        Paragraph(meta["title"], styles["title"]),
        Spacer(1, 2.4),
        Paragraph(meta["subtitle"], styles["subtitle"]),
        Spacer(1, 3.2),
        HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceBefore=0, spaceAfter=3.2),
        Paragraph(
            f"<b>{meta['authors']}</b> &nbsp;·&nbsp; "
            f"<a href='https://{meta['url']}'><font color='#1a3a5c'>{meta['url']}</font></a> &nbsp;·&nbsp; "
            f"{meta['note']}",
            styles["meta"],
        ),
        Spacer(1, 3.0),
        HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#b9c2cc"),
                   spaceBefore=0, spaceAfter=0),
    ]


def abstract_flowable(styles: dict) -> KeepTogether:
    """The abstract: a shaded box with a blue left rule, kept on one piece."""
    box = Table([[Paragraph(content.ABSTRACT, styles["abstract"])]], colWidths=[COL_W])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f9")),
        ("LINEBEFORE", (0, 0), (0, -1), 1.6, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    return KeepTogether([box, Spacer(1, 6)])


def _cell_style(styles: dict, align: str, muted: bool) -> ParagraphStyle:
    base = styles["cell-muted" if muted else "cell"]
    style = ParagraphStyle(f"{base.name}-{align}", parent=base)
    style.alignment = TA_RIGHT if align == "right" else TA_LEFT
    return style


def table_flowable(key: str, styles: dict) -> KeepTogether:
    """One of ``report_content.TABLES``: caption above, rules between rows."""
    spec = content.TABLES[key]
    muted = set(spec.get("muted", ()))
    mono_col = spec.get("mono_col")
    rows = [[Paragraph(h, styles["head"]) for h in spec["header"]]]
    for index, row in enumerate(spec["rows"]):
        row_styles = [_cell_style(styles, align, index in muted) for align in spec["align"]]
        cells = []
        for column, (cell, style) in enumerate(zip(row, row_styles)):
            if column == mono_col:
                cell = f"<font face='Mono' size='6.5'>{cell}</font>"
            cells.append(Paragraph(cell, style))
        rows.append(cells)

    table = Table(rows, colWidths=spec["widths"], hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7),
        ("TOPPADDING", (0, 0), (-1, 0), 0.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLUE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
    ]
    for row, fill in spec.get("highlight", {}).items():
        commands.append(("BACKGROUND", (0, row + 1), (-1, row + 1), colors.HexColor(fill)))
    table.setStyle(TableStyle(commands))

    return KeepTogether([
        Paragraph(spec["caption"], styles["caption"]),
        Spacer(1, 1.5),
        table,
        Spacer(1, 4.5),
    ])


def figure_flowable(path: str, caption: str, styles: dict, scale: float = 1.0) -> KeepTogether:
    """A column-width figure with its caption, never split across frames."""
    source = PROJECT_ROOT / path
    px_w, px_h = ImageReader(str(source)).getSize()
    width = COL_W * scale
    height = width * px_h / px_w
    return KeepTogether([
        Spacer(1, 2.5),
        Image(str(source), width=width, height=height),
        Paragraph(caption, styles["caption"]),
        Spacer(1, 4.5),
    ])



def build_story(styles: dict) -> list:
    """Walk ``report_content.BLOCKS`` and emit flowables, in order."""
    story: list = [NextPageTemplate("rest")]
    story.extend(header_flowables(styles))
    story.append(FrameBreak())
    story.append(abstract_flowable(styles))

    for block in content.BLOCKS:
        kind = block[0]
        if kind == "h2":
            number, _, rest = block[1].partition(". ")
            story.append(Paragraph(
                f"<font color='#b03a2e'>{number}.</font> {rest}", styles["h2"]))
        elif kind == "p":
            story.append(Paragraph(block[1], styles["body"]))
        elif kind == "eq":
            story.append(Paragraph(block[1], styles["eq"]))
        elif kind == "table":
            story.append(table_flowable(block[1], styles))
        elif kind == "fig":
            scale = block[3] if len(block) > 3 else 1.0
            story.append(figure_flowable(block[1], block[2], styles, scale))
        elif kind == "refs":
            story.append(Paragraph("References", styles["h2"]))
            for entry in block[1]:
                story.append(Paragraph(entry, styles["ref"]))
        else:  # pragma: no cover - a typo in the content module
            raise ValueError(f"unknown block kind: {kind!r}")
    return story


def _footer(total_pages: int | None):
    """Page number (``n`` or ``n / N``) centred at the foot of every page."""

    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Body", 6.8)
        canvas.setFillColor(colors.HexColor("#9a9a9a"))
        text = f"{doc.page}" if total_pages is None else f"{doc.page} / {total_pages}"
        canvas.drawCentredString(PAGE_W / 2, 4.6 * mm, text)
        canvas.restoreState()

    return draw


def build(pdf_path: Path = PDF_PATH, total_pages: int | None = None) -> int:
    """Render the report and return its page count."""
    register_fonts()
    styles = make_styles()
    header = header_flowables(styles)
    # +10pt of slack: without it the last hairline rule overflows into column 1
    header_height = sum(f.wrap(CONTENT_W, PAGE_H)[1] for f in header) + 10

    def frames(with_header: bool) -> list:
        col_height = FRAME_H - header_height if with_header else FRAME_H
        made = []
        if with_header:
            made.append(Frame(MARGIN_X, PAGE_H - MARGIN_TOP - header_height, CONTENT_W,
                              header_height, leftPadding=0, rightPadding=0,
                              topPadding=0, bottomPadding=0, id="head"))
        made.append(Frame(MARGIN_X, MARGIN_BOTTOM, COL_W, col_height,
                          leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                          id="col1"))
        made.append(Frame(MARGIN_X + COL_W + COL_GAP, MARGIN_BOTTOM, COL_W, col_height,
                          leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                          id="col2"))
        return made

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN_X, rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title=content.META["title"],
        author=content.META["authors"],
        subject="AI Assignment 2 (Phase 3): PPO autoscaling on real HTTP traffic",
        creator="utils/make_report.py",
    )
    footer = _footer(total_pages)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=frames(with_header=True), onPage=footer),
        PageTemplate(id="rest", frames=frames(with_header=False), onPage=footer),
    ])
    doc.build(build_story(styles))
    return doc.page


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-figure", action="store_true",
                        help="reuse report/fig_reward.png instead of redrawing it")
    parser.add_argument("--check", action="store_true",
                        help="fail when the body needs more than 2 pages plus references")
    parser.add_argument("--out", type=Path, default=PDF_PATH,
                        help=f"PDF destination (default: {PDF_PATH})")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.skip_figure:
        figure = make_reward_figure()
        print(f"figure  {figure.relative_to(PROJECT_ROOT)}")

    pages = build(args.out)                     # pass 1: how many pages?
    pages = build(args.out, total_pages=pages)  # pass 2: print "n / N"
    size_kb = args.out.stat().st_size / 1_024
    print(f"report  {args.out.relative_to(PROJECT_ROOT)}  {pages} pages  {size_kb:.0f} kB")

    if args.check and pages > 3:
        print("FAIL: the body must fit on 2 pages (references may spill to page 3)",
              file=sys.stderr)
        return 1
    if args.check:
        print("OK: body fits on 2 pages, references may follow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
