#!/usr/bin/env python3
"""
Build SVD Image Compression Lab Report PDF
Course: Signal Processing and Systems Design Laboratory (EE69205)
Experiment 6: Gray-Scale Image Compression using SVD
Student: Bagada Vishal Jayantilal (Roll No: 23EE38004)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable
)

# ---------------------------------------------------------------- paths
FIG_DIR = "extracted_figures"
OUT_PDF = "SVD_Image_Compression_Lab_Report_Bagada_Vishal_23EE38004.pdf"

# ---------------------------------------------------------------- styles
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "title", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=18, leading=22,
    alignment=TA_CENTER, textColor=colors.HexColor("#1a3a5c"),
    spaceAfter=6,
)
subtitle_style = ParagraphStyle(
    "subtitle", parent=styles["Normal"],
    fontName="Helvetica", fontSize=12, leading=16,
    alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
    spaceAfter=2,
)
info_style = ParagraphStyle(
    "info", parent=styles["Normal"],
    fontName="Helvetica", fontSize=10, leading=14,
    alignment=TA_CENTER, textColor=colors.HexColor("#333333"),
)
h1 = ParagraphStyle(
    "h1", parent=styles["Heading1"],
    fontName="Helvetica-Bold", fontSize=14, leading=18,
    textColor=colors.HexColor("#1a3a5c"), spaceBefore=14, spaceAfter=6,
    borderPadding=3,
)
h2 = ParagraphStyle(
    "h2", parent=styles["Heading2"],
    fontName="Helvetica-Bold", fontSize=11.5, leading=15,
    textColor=colors.HexColor("#2c5f8a"), spaceBefore=10, spaceAfter=4,
)
body = ParagraphStyle(
    "body", parent=styles["Normal"],
    fontName="Helvetica", fontSize=9.5, leading=13.5,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
body_left = ParagraphStyle(
    "body_left", parent=body, alignment=TA_LEFT,
)
bullet = ParagraphStyle(
    "bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=3,
)
caption = ParagraphStyle(
    "caption", parent=styles["Normal"],
    fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
    alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
    spaceBefore=2, spaceAfter=10,
)
cell_style = ParagraphStyle(
    "cell", parent=styles["Normal"],
    fontName="Helvetica", fontSize=8, leading=10,
)
cell_bold = ParagraphStyle(
    "cell_bold", parent=cell_style, fontName="Helvetica-Bold",
)

def P(text, style=body, **kwargs):
    return Paragraph(text, style, **kwargs)

# ---------------------------------------------------------------- header/footer
def header_footer(canvas, doc):
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(colors.HexColor("#1a3a5c"))
    canvas.setLineWidth(0.8)
    canvas.line(2 * cm, A4[1] - 1.6 * cm, A4[0] - 2 * cm, A4[1] - 1.6 * cm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.HexColor("#1a3a5c"))
    canvas.drawString(2 * cm, A4[1] - 1.45 * cm,
                      "EE69205 - Signal Processing and Systems Design Laboratory")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.45 * cm,
                           "Experiment 6: SVD Image Compression")
    # Footer
    canvas.setStrokeColor(colors.HexColor("#1a3a5c"))
    canvas.setLineWidth(0.8)
    canvas.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(2 * cm, 1.2 * cm,
                      "Bagada Vishal Jayantilal  |  Roll No: 23EE38004")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm,
                           f"Page {doc.page}")
    canvas.restoreState()

# ---------------------------------------------------------------- document
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    title="SVD Image Compression Lab Report",
    author="Bagada Vishal Jayantilal",
)
story = []

# ================================================================ TITLE BLOCK
story.append(Spacer(1, 6))
story.append(P("Laboratory Report", title_style))
story.append(P("Gray-Scale Image Compression using Singular Value Decomposition (SVD)", subtitle_style))
story.append(Spacer(1, 4))
story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#1a3a5c"), spaceAfter=8))

info_table = Table([
    [P("<b>Course:</b> Signal Processing and Systems Design Laboratory (EE69205)", info_style),
     P("<b>Experiment No.:</b> 6", info_style)],
    [P("<b>Name:</b> Bagada Vishal Jayantilal", info_style),
     P("<b>Roll No.:</b> 23EE38004", info_style)],
    [P("<b>Topic:</b> Whole-Image and Patch-wise SVD Compression", info_style),
     P("<b>Date:</b> October 2026", info_style)],
], colWidths=[8.6 * cm, 8.6 * cm])
info_table.setStyle(TableStyle([
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(info_table)
story.append(Spacer(1, 8))

# ================================================================ ABSTRACT
story.append(P("Abstract", h1))
story.append(P(
    "This experiment investigates gray-scale image compression using Singular Value "
    "Decomposition (SVD). A 140 &times; 140 pixel crop of the standard Cameraman image is "
    "compressed using two approaches: (i) whole-image low-rank SVD approximation and "
    "(ii) patch-wise SVD compression, in which the image is divided into 25 non-overlapping "
    "28 &times; 28 patches, each compressed independently. Reconstruction quality is assessed "
    "using Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural "
    "Similarity Index (SSIM), plotted against the compression ratio. Results show that "
    "increasing the retained rank K improves quality monotonically, and that patch-wise "
    "compression better preserves local structural detail (higher SSIM) at comparable "
    "compression ratios, while whole-image SVD provides a compact global approximation.",
    body))
story.append(Spacer(1, 4))

# ================================================================ OBJECTIVE
story.append(P("1. Objective and Goals", h1))
story.append(P(
    "<b>Objective:</b> To implement and analyze Singular Value Decomposition (SVD) for "
    "gray-scale image compression, and to compare whole-image versus patch-wise low-rank "
    "approximation in terms of compression ratio and reconstruction quality.", body))
story.append(P("Goals:", body))
for g in [
    "Decompose a gray-scale image matrix using SVD and construct its rank-K approximation "
    "A<sub>K</sub> = \u2211<sub>k=1</sub><sup>K</sup> \u03c3<sub>k</sub> u<sub>k</sub> v<sub>k</sub><sup>T</sup>.",
    "Study how the number of retained singular components (K) affects compression ratio "
    "(CR), MSE, PSNR, and SSIM.",
    "Implement whole-image SVD compression on a 140 &times; 140 Cameraman crop.",
    "Implement patch-wise SVD compression over 25 non-overlapping 28 &times; 28 patches.",
    "Compare both approaches using metric-versus-compression-ratio curves and a combined "
    "comparison table.",
]:
    story.append(P(g, bullet, bulletText="\u2022"))
story.append(Spacer(1, 4))

# ================================================================ THEORY
story.append(P("2. Theory: SVD for Image Compression", h1))
story.append(P(
    "Every real matrix <b>A</b> of size m &times; n and rank r admits the Singular Value "
    "Decomposition:", body))
story.append(Spacer(1, 2))
story.append(P(
    "<i>A = U \u0393 V<sup>T</sup></i>", body_left))
story.append(Spacer(2, 2))
story.append(P(
    "where <b>U</b> (m &times; m) contains orthonormal left singular vectors, "
    "<b>\u0393</b> (m &times; n) is diagonal with singular values "
    "\u03c3<sub>1</sub> \u2265 \u03c3<sub>2</sub> \u2265 \u2026 \u2265 \u03c3<sub>r</sub> &gt; 0, "
    "and <b>V</b><sup>T</sup> (n &times; n) contains orthonormal right singular vectors. "
    "Equivalently, the outer-product expansion is:", body))
story.append(Spacer(1, 2))
story.append(P(
    "<i>A = \u2211<sub>i=1</sub><sup>r</sup> \u03c3<sub>i</sub> u<sub>i</sub> v<sub>i</sub><sup>T</sup></i>", body_left))
story.append(Spacer(2, 2))
story.append(P(
    "Keeping only the first K terms yields the best rank-K approximation (Eckart\u2013Young\u2013Mirsky "
    "theorem):", body))
story.append(Spacer(1, 2))
story.append(P(
    "<i>A<sub>K</sub> = \u2211<sub>k=1</sub><sup>K</sup> \u03c3<sub>k</sub> u<sub>k</sub> v<sub>k</sub><sup>T</sup></i>", body_left))
story.append(Spacer(4, 2))
story.append(P(
    "<b>Compression ratio.</b> Storing a rank-K approximation of an H &times; W image requires "
    "K(1 + H + W) numbers (K singular values, K columns of U of length H, and K rows of V<sup>T</sup> "
    "of length W) instead of H &middot; W pixels. Hence:", body))
story.append(Spacer(1, 2))
story.append(P(
    "<i>CR = K(1 + H + W) / (H &middot; W)</i>", body_left))
story.append(Spacer(2, 2))
story.append(P(
    "CR &lt; 1 indicates compression; CR &gt; 1 means the approximation uses more numbers than "
    "the original. For patch-wise compression the same formula is applied per patch with patch "
    "dimensions H<sub>p</sub> &times; W<sub>p</sub>.", body))
story.append(P(
    "<b>Quality metrics</b> (pixel values normalized to [0, 1], data_range = 1):", body))
story.append(Spacer(1, 2))
metrics_rows = [
    [P("<b>Metric</b>", cell_bold), P("<b>Definition / Note</b>", cell_bold), P("<b>Direction</b>", cell_bold)],
    [P("MSE", cell_style), P("Mean Squared Error", cell_style), P("Lower is better", cell_style)],
    [P("PSNR (dB)", cell_style), P("10 log<sub>10</sub>(1 / MSE)", cell_style), P("Higher is better", cell_style)],
    [P("SSIM", cell_style), P("Structural Similarity Index (max = 1)", cell_style), P("Higher is better", cell_style)],
]
mt = Table(metrics_rows, colWidths=[2.2 * cm, 8.2 * cm, 3.2 * cm])
mt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
]))
story.append(mt)
story.append(Spacer(1, 6))

# ================================================================ DATA EDA
story.append(P("3. Data and Exploratory Data Analysis (EDA)", h1))
story.append(P(
    "<b>Dataset.</b> The standard gray-scale <b>Cameraman</b> image is used. The notebook first "
    "attempts to download the image linked in the lab manual "
    "(Google Drive: id=1nlTgGDd0dLn5Kg7ZvcW8CEHwnyzH-tFW); if unavailable, it falls back to "
    "<i>skimage.data.camera()</i>. The loaded image is 148 &times; 180 pixels.", body))
story.append(P(
    "<b>Preprocessing.</b> The image is converted to gray-scale, pixel values are normalized "
    "from [0, 255] to [0, 1], and a center crop of size <b>140 &times; 140</b> is taken so that "
    "it divides evenly into 25 patches of <b>28 &times; 28</b> pixels.", body))

# Image 1: original
story.append(P("Figure 1: Original Cameraman image as loaded (148 &times; 180)", caption))
img_path = os.path.join(FIG_DIR, "fig_01_cell3_out1.png")
story.append(Image(img_path, width=8.6 * cm, height=8.6 * cm * 423 / 485))

# Image 2: crop
story.append(P("Figure 2: Center 140 &times; 140 crop used for the experiment", caption))
img_path = os.path.join(FIG_DIR, "fig_02_cell5_out0.png")
story.append(Image(img_path, width=8.6 * cm, height=8.6 * cm * 426 / 405))

# Image 6: patches (EDA)
story.append(P("Figure 3: EDA \u2014 25 non-overlapping 28 &times; 28 patches (5 &times; 5 grid)", caption))
img_path = os.path.join(FIG_DIR, "fig_06_cell13_out1.png")
story.append(Image(img_path, width=12 * cm, height=12 * cm * 789 / 783))
story.append(PageBreak())

# ================================================================ METHODOLOGY
story.append(P("4. Methodology", h1))
story.append(P(
    "<b>Whole-image SVD compression.</b> The compact SVD of the 140 &times; 140 crop is computed "
    "once using NumPy (<i>np.linalg.svd</i> with <i>full_matrices=False</i>). For each "
    "K &in; {5, 10, 20, 40, 50, 100}, the image is reconstructed from the top-K singular "
    "triplets, and CR, MSE, PSNR, and SSIM are evaluated against the original crop. A "
    "full-rank check at K = 140 confirms exact reconstruction to floating-point precision.", body))
story.append(P(
    "<b>Patch-wise SVD compression.</b> The crop is split into 25 non-overlapping 28 &times; 28 "
    "patches. Each patch is decomposed and independently reconstructed for "
    "K &in; {5, 10, 15, 20}. The reconstructed patches are reassembled into a 140 &times; 140 "
    "image, and CR<sub>p</sub>, MSE, PSNR, and SSIM are computed against the original crop.", body))
story.append(Spacer(1, 4))

# ================================================================ RESULTS
story.append(P("5. Results", h1))

# ---- 5.1 full rank check
story.append(P("5.1 Full-Rank Sanity Check (K = 140)", h2))
fr_rows = [
    [P("<b>Parameter</b>", cell_bold), P("<b>Value</b>", cell_bold)],
    [P("K", cell_style), P("140", cell_style)],
    [P("MSE", cell_style), P("\u2248 1.11 &times; 10<sup>\u221230</sup>", cell_style)],
    [P("PSNR (dB)", cell_style), P("299.55", cell_style)],
    [P("SSIM", cell_style), P("1.0000", cell_style)],
]
ft = Table(fr_rows, colWidths=[4.4 * cm, 4.4 * cm])
ft.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(ft)
story.append(P("Table 1: Full-rank reconstruction (K = 140) reproduces the crop to numerical precision.", caption))
story.append(P(
    "Figure 4 shows the original crop beside its K = 140 reconstruction; the two are visually "
    "indistinguishable, confirming the correctness of the SVD implementation.", body))
img_path = os.path.join(FIG_DIR, "fig_03_cell7_out2.png")
story.append(Image(img_path, width=10 * cm, height=10 * cm * 390 / 839))
story.append(P("Figure 4: Original crop vs. K = 140 reconstruction.", caption))

# ---- 5.2 whole image table
story.append(P("5.2 Whole-Image Compression Results", h2))
wi_rows = [
    [P("<b>K</b>", cell_bold), P("<b>CR</b>", cell_bold), P("<b>MSE</b>", cell_bold),
     P("<b>PSNR (dB)</b>", cell_bold), P("<b>SSIM</b>", cell_bold)],
    [P("5", cell_style), P("0.0717", cell_style), P("0.014199", cell_style),
     P("18.48", cell_style), P("0.4776", cell_style)],
    [P("10", cell_style), P("0.1434", cell_style), P("0.007549", cell_style),
     P("21.22", cell_style), P("0.5948", cell_style)],
    [P("20", cell_style), P("0.2867", cell_style), P("0.002746", cell_style),
     P("25.61", cell_style), P("0.7455", cell_style)],
    [P("40", cell_style), P("0.5735", cell_style), P("0.000633", cell_style),
     P("31.99", cell_style), P("0.8840", cell_style)],
    [P("50", cell_style), P("0.7168", cell_style), P("0.000299", cell_style),
     P("35.24", cell_style), P("0.9285", cell_style)],
    [P("100", cell_style), P("1.4337", cell_style), P("1.0 &times; 10<sup>\u22126</sup>", cell_style),
     P("58.92", cell_style), P("0.9994", cell_style)],
]
wit = Table(wi_rows, colWidths=[1.8 * cm, 1.8 * cm, 2.6 * cm, 2.8 * cm, 2.2 * cm])
wit.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(wit)
story.append(P("Table 2: Whole-image SVD compression results for the 140 &times; 140 Cameraman crop.", caption))
img_path = os.path.join(FIG_DIR, "fig_04_cell9_out1.png")
story.append(Image(img_path, width=11 * cm, height=11 * cm * 821 / 1155))
story.append(P("Figure 5: Whole-image reconstructions for K = 5, 10, 20, 40, 50, 100.", caption))
img_path = os.path.join(FIG_DIR, "fig_05_cell11_out0.png")
story.append(Image(img_path, width=12 * cm, height=12 * cm * 390 / 1490))
story.append(P("Figure 6: Whole-image metric curves (MSE, PSNR, SSIM vs. compression ratio).", caption))
story.append(PageBreak())

# ---- 5.3 patch-wise table
story.append(P("5.3 Patch-wise Compression Results", h2))
pw_rows = [
    [P("<b>K</b>", cell_bold), P("<b>CR<sub>p</sub></b>", cell_bold), P("<b>MSE</b>", cell_bold),
     P("<b>PSNR (dB)</b>", cell_bold), P("<b>SSIM</b>", cell_bold)],
    [P("5", cell_style), P("0.3635", cell_style), P("0.001889", cell_style),
     P("27.24", cell_style), P("0.8865", cell_style)],
    [P("10", cell_style), P("0.7270", cell_style), P("0.000311", cell_style),
     P("35.07", cell_style), P("0.9740", cell_style)],
    [P("15", cell_style), P("1.0906", cell_style), P("4.8 &times; 10<sup>\u22125</sup>", cell_style),
     P("43.17", cell_style), P("0.9961", cell_style)],
    [P("20", cell_style), P("1.4541", cell_style), P("3.0 &times; 10<sup>\u22126</sup>", cell_style),
     P("54.88", cell_style), P("0.9998", cell_style)],
]
pwt = Table(pw_rows, colWidths=[1.8 * cm, 2.2 * cm, 2.6 * cm, 2.8 * cm, 2.2 * cm])
pwt.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(pwt)
story.append(P("Table 3: Patch-wise SVD compression results (25 patches of 28 &times; 28).", caption))
img_path = os.path.join(FIG_DIR, "fig_07_cell14_out1.png")
story.append(Image(img_path, width=10 * cm, height=10 * cm * 790 / 800))
story.append(P("Figure 7: Patch-wise reconstructions for K = 5, 10, 15, 20.", caption))
img_path = os.path.join(FIG_DIR, "fig_08_cell16_out0.png")
story.append(Image(img_path, width=12 * cm, height=12 * cm * 390 / 1490))
story.append(P("Figure 8: Patch-wise metric curves (MSE, PSNR, SSIM vs. patch compression ratio).", caption))

# ---- 5.4 comparison table
story.append(P("5.4 Whole-Image vs. Patch-wise Comparison", h2))
comp_rows = [
    [P("<b>Method</b>", cell_bold), P("<b>K</b>", cell_bold), P("<b>CR</b>", cell_bold),
     P("<b>MSE</b>", cell_bold), P("<b>PSNR (dB)</b>", cell_bold), P("<b>SSIM</b>", cell_bold)],
    [P("Patch-wise", cell_style), P("5", cell_style), P("0.3635", cell_style),
     P("0.001889", cell_style), P("27.24", cell_style), P("0.8865", cell_style)],
    [P("Patch-wise", cell_style), P("10", cell_style), P("0.7270", cell_style),
     P("0.000311", cell_style), P("35.07", cell_style), P("0.9740", cell_style)],
    [P("Patch-wise", cell_style), P("15", cell_style), P("1.0906", cell_style),
     P("4.8 &times; 10<sup>\u22125</sup>", cell_style), P("43.17", cell_style), P("0.9961", cell_style)],
    [P("Patch-wise", cell_style), P("20", cell_style), P("1.4541", cell_style),
     P("3.0 &times; 10<sup>\u22126</sup>", cell_style), P("54.88", cell_style), P("0.9998", cell_style)],
    [P("Whole-image", cell_style), P("5", cell_style), P("0.0717", cell_style),
     P("0.014199", cell_style), P("18.48", cell_style), P("0.4776", cell_style)],
    [P("Whole-image", cell_style), P("10", cell_style), P("0.1434", cell_style),
     P("0.007549", cell_style), P("21.22", cell_style), P("0.5948", cell_style)],
    [P("Whole-image", cell_style), P("20", cell_style), P("0.2867", cell_style),
     P("0.002746", cell_style), P("25.61", cell_style), P("0.7455", cell_style)],
    [P("Whole-image", cell_style), P("40", cell_style), P("0.5735", cell_style),
     P("0.000633", cell_style), P("31.99", cell_style), P("0.8840", cell_style)],
    [P("Whole-image", cell_style), P("50", cell_style), P("0.7168", cell_style),
     P("0.000299", cell_style), P("35.24", cell_style), P("0.9285", cell_style)],
    [P("Whole-image", cell_style), P("100", cell_style), P("1.4337", cell_style),
     P("1.0 &times; 10<sup>\u22126</sup>", cell_style), P("58.92", cell_style), P("0.9994", cell_style)],
]
ct = Table(comp_rows, colWidths=[2.6 * cm, 1.4 * cm, 1.6 * cm, 2.6 * cm, 2.6 * cm, 2.0 * cm],
           repeatRows=1)
ct.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b8")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
]))
story.append(ct)
story.append(P("Table 4: Combined comparison of whole-image and patch-wise SVD compression.", caption))
img_path = os.path.join(FIG_DIR, "fig_09_cell18_out1.png")
story.append(Image(img_path, width=12 * cm, height=12 * cm * 390 / 1490))
story.append(P("Figure 9: Comparison curves \u2014 whole-image vs. patch-wise (MSE, PSNR, SSIM vs. CR).", caption))

# ---- 5.5 patch verification
story.append(P("5.5 Patch Tiling Verification", h2))
story.append(P(
    "Reassembling the 25 original 28 &times; 28 patches reproduces the center crop exactly "
    "(<i>np.allclose</i> returned <b>True</b>), confirming that the patch layout and tiling order "
    "are correct.", body))
img_path = os.path.join(FIG_DIR, "fig_10_cell21_out1.png")
story.append(Image(img_path, width=9 * cm, height=9 * cm * 390 / 1490))
story.append(P("Figure 10: Original crop vs. full image assembled from 25 crops (match verified).", caption))
img_path = os.path.join(FIG_DIR, "fig_11_cell21_out2.png")
story.append(Image(img_path, width=9.5 * cm, height=9.5 * cm * 890 / 867))
story.append(P("Figure 11: Full assembled images from reconstructed patches (patch-wise K = 5, 10, 15, 20).", caption))
story.append(PageBreak())

# ================================================================ OBSERVATIONS
story.append(P("6. Observations and Conclusion", h1))
for obs in [
    "<b>Effect of K.</b> Increasing the retained rank K retains more singular components and "
    "monotonically improves reconstruction quality: MSE decreases, while PSNR and SSIM increase.",
    "<b>Full-rank reconstruction.</b> At K = 140 the crop is reconstructed to numerical precision "
    "(MSE \u2248 10<sup>\u221230</sup>, SSIM = 1.0), validating the SVD implementation.",
    "<b>Patch-wise advantage in structure.</b> Patch-wise SVD applies a separate low-rank basis to "
    "each 28 &times; 28 region, preserving local edges and textures better than one global basis. At "
    "comparable compression ratios it achieves markedly higher SSIM. For example, patch-wise "
    "K = 5 (CR<sub>p</sub> = 0.364) gives SSIM = 0.887, whereas whole-image K = 20 (CR = 0.287) "
    "gives SSIM = 0.746.",
    "<b>PSNR vs. SSIM.</b> PSNR and MSE track each other inversely, but SSIM is more discriminating "
    "for perceptual/structural quality; two methods can have similar PSNR at similar CR yet differ "
    "substantially in SSIM.",
    "<b>No universal winner.</b> Patch-wise compression is not always superior at every compression "
    "ratio; the outcome depends on image content and the chosen K, so actual metric values should "
    "be compared before declaring a winner.",
    "<b>Compression-ratio definition.</b> The reported CR values are parameter-count ratios only and "
    "exclude file-format headers, entropy coding, and other encoding overhead.",
]:
    story.append(P(obs, bullet, bulletText="\u2022"))
story.append(Spacer(1, 4))
story.append(P(
    "<b>Conclusion.</b> SVD provides an effective, mathematically optimal framework for low-rank "
    "gray-scale image compression. Whole-image SVD yields a compact global approximation whose "
    "quality scales smoothly with K. Patch-wise SVD better preserves local structural detail at "
    "comparable compression ratios, as reflected by higher SSIM, making it preferable when "
    "perceptual structure is important. The choice between methods should be guided by the target "
    "compression ratio, image content, and the quality metric that matters most for the application.",
    body))
story.append(Spacer(1, 8))

# ================================================================ REFERENCES
story.append(P("7. References", h1))
refs = [
    "Lab Manual \u2014 Signal Processing and Systems Design Laboratory (EE69205), Experiment 6: "
    "Gray-Scale Image Compression using Singular Value Decomposition (SVD).",
    "ACME BYU Applied Mathematics Lab \u2014 \u201cSVD and Image Compression\u201d (low-rank matrix "
    "approximation, Eckart\u2013Young\u2013Mirsky theorem, image compression workflow).",
    "NumPy documentation \u2014 <i>numpy.linalg.svd</i> (compact SVD computation).",
    "scikit-image documentation \u2014 <i>skimage.metrics</i>: mean_squared_error, "
    "peak_signal_noise_ratio, structural_similarity.",
    "Trefethen & Bau, \u201cNumerical Linear Algebra,\u201d SIAM, 1997 (SVD theory and computation).",
]
for i, r in enumerate(refs, 1):
    story.append(P(f"{i}. {r}", body))
story.append(Spacer(1, 10))
story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#1a3a5c"), spaceAfter=6))
story.append(P(
    "Prepared by <b>Bagada Vishal Jayantilal</b> (Roll No: 23EE38004) \u2014 "
    "EE69205 Signal Processing and Systems Design Laboratory, Experiment 6.", info_style))

# ---------------------------------------------------------------- build
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"PDF created: {OUT_PDF}")
