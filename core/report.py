"""PDF report for an E-PROM analysis (ReportLab only, no external fonts)."""

from __future__ import annotations
import io
from datetime import datetime

import numpy as np
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

NAVY = colors.HexColor("#1F3864")
BLUE = colors.HexColor("#305496")
LIGHT = colors.HexColor("#F4F6FB")


def _safe(value):
    """Make text encodable by the standard PDF fonts."""
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .encode("cp1252", errors="replace").decode("cp1252")
    )


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=22, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=ss["Heading2"], textColor=BLUE, fontSize=14, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=ss["Heading3"], textColor=NAVY, fontSize=11, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12),
        "small": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444")),
        "cell": ParagraphStyle("c", parent=ss["BodyText"], fontSize=7.5, leading=9),
        "mono": ParagraphStyle("m", parent=ss["Code"], fontSize=6.5, leading=8),
    }


def _table(rows, col_widths=None, header=True, font=7.5, zebra=True):
    st = _styles()
    data = [
        [Paragraph(_safe(c), st["cell"]) if isinstance(c, str) else c for c in row]
        for row in rows
    ]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        # header cells are Paragraphs: recolour their text
        for j, cell in enumerate(data[0]):
            if isinstance(cell, Paragraph):
                cell.style = ParagraphStyle("hc", parent=cell.style, textColor=colors.white, fontName="Helvetica-Bold")
                data[0][j] = Paragraph(cell.text, cell.style)
        t = Table(data, colWidths=col_widths, repeatRows=1)
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def _bar_chart(labels, values, title, width=470, height=170, horizontal=False):
    d = Drawing(width, height)
    d.add(String(width / 2, height - 10, _safe(title), fontSize=9, fontName="Helvetica-Bold", textAnchor="middle", fillColor=NAVY))
    chart = HorizontalBarChart() if horizontal else VerticalBarChart()
    chart.x, chart.y = 70 if horizontal else 40, 28
    chart.width, chart.height = width - (110 if horizontal else 70), height - 50
    vals = [float(v) for v in values]
    chart.data = [vals]
    chart.bars[0].fillColor = BLUE
    chart.categoryAxis.categoryNames = [_safe(l)[:22] for l in labels]
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labelAxisMode = "low"
    chart.valueAxis.labels.fontSize = 7
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = (hi - lo) * 0.1 or 0.1
    chart.valueAxis.valueMin, chart.valueAxis.valueMax = lo - (pad if lo < 0 else 0), hi + pad
    chart.valueAxis.labelTextFormat = "%.3f"
    if not horizontal:
        chart.categoryAxis.labels.angle = 30 if len(labels) > 6 else 0
        chart.categoryAxis.labels.boxAnchor = "ne" if len(labels) > 6 else "n"
    d.add(chart)
    return d


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(15 * mm, 9 * mm, "E-PROM report")
    canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_report(data, result, config_df, fahp, extras=None, config_json=None, version="E-PROM"):
    """Return the PDF bytes.

    extras may contain: mc, mc_params, sensitivity, agreement, loo, reversal,
    contributions, flowsort, portfolio (each optional).
    """
    extras = extras or {}
    st = _styles()
    story = []
    ids = [str(x) for x in data["ids"]]
    names = list(data["criteria"])
    n, k = len(ids), len(names)

    # ------------------------------------------------------------ title
    story.append(Paragraph("E-PROM — Analysis report", st["title"]))
    story.append(Paragraph(
        _safe(f"Express Fuzzy Preference Ranking with PROMETHEE · generated {datetime.now():%Y-%m-%d %H:%M} · {version}"),
        st["small"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        _safe(
            f"{data['n_experts']} decision maker(s) ({', '.join(map(str, data['expert_names']))}), "
            f"{n} alternatives, {k} criteria. Criterion weights come from FAHP-Express (fuzzy, kept fuzzy "
            "through the analysis); alternatives are ranked by the defuzzified net flow of Fuzzy PROMETHEE, "
            "aggregating the decision makers in the fuzzy domain."
        ),
        st["body"],
    ))

    # ------------------------------------------------------------ ranking
    story.append(Paragraph("1. Ranking", st["h1"]))
    order = np.argsort(np.asarray(result["rank_e_prom"], dtype=float), kind="mergesort")
    rows = [["Rank", "ID", "Description", "Phi+", "Phi-", "Net Phi"]]
    for i in order:
        rows.append([
            _fmt(result["rank_e_prom"][i], 1).rstrip("0").rstrip("."), ids[i], str(data["descriptions"][i]),
            _fmt(result["phi_plus"][i]), _fmt(result["phi_minus"][i]), _fmt(result["phi_net"][i]),
        ])
    story.append(_table(rows, [30, 40, 200, 55, 55, 60]))
    story.append(Spacer(1, 6))
    story.append(_bar_chart([ids[i] for i in order], [result["phi_net"][i] for i in order], "Net flow (Phi) by alternative"))

    if n <= 12:
        from .promethee import promethee_i_relation
        story.append(Paragraph("PROMETHEE I relations (row vs column: P outranks, P- is outranked, I indifferent, R incomparable)", st["h2"]))
        rel = [[""] + ids]
        for i in range(n):
            rel.append([ids[i]] + [
                "I" if i == j else promethee_i_relation(result["phi_plus"][i], result["phi_minus"][i], result["phi_plus"][j], result["phi_minus"][j])
                for j in range(n)
            ])
        story.append(_table(rel, [40] + [min(40, 400 // n)] * n))

    # ------------------------------------------------------------ configuration
    story.append(Paragraph("2. Configuration", st["h1"]))
    cfg_rows = [list(map(str, config_df.columns))]
    for _, r in config_df.iterrows():
        cfg_rows.append([_safe(v) if not isinstance(v, float) else f"{v:g}" for v in r.tolist()])
    story.append(_table(cfg_rows))

    ew = np.asarray(result.get("expert_weights", []), dtype=float)
    if len(ew) and not np.allclose(ew, ew[0]):
        story.append(Paragraph("Decision-maker weights", st["h2"]))
        story.append(_table([["Decision maker", "Weight"]] + [[str(a), _fmt(b, 3)] for a, b in zip(result["expert_names"], ew)], [150, 60]))

    story.append(Paragraph("Attribute precision (H = High, M = Medium, L = Low)", st["h2"]))
    prec_rows = [["Decision maker"] + names]
    for e, row in enumerate(fahp["precision_levels"]):
        prec_rows.append([str(fahp["expert_names"][e])] + [str(x)[0] for x in row])
    story.append(_table(prec_rows))

    # ------------------------------------------------------------ weights
    story.append(Paragraph("3. Criteria weights (FAHP-Express)", st["h1"]))
    wf = np.asarray(fahp["w_fuzzy_group"], dtype=float)
    wc = np.asarray(fahp["w_crisp_group"], dtype=float)
    ind = np.asarray(fahp["w_crisp_individual"], dtype=float)
    rows = [["Criterion", "a", "b", "c", "d", "Crisp (group)"] + [str(x) for x in fahp["expert_names"]]]
    for i in range(k):
        rows.append([names[i]] + [_fmt(v) for v in wf[i]] + [_fmt(wc[i])] + [_fmt(ind[i, e]) for e in range(ind.shape[1])])
    story.append(_table(rows))
    story.append(Spacer(1, 4))
    story.append(_bar_chart(names, wc, "Group crisp weight by criterion", horizontal=True, height=max(120, 22 * k + 40)))
    refs = fahp.get("reference_criteria")
    if refs:
        story.append(Paragraph(_safe("Reference criteria: " + ", ".join(f"{a}: {b}" for a, b in zip(fahp["expert_names"], refs))), st["small"]))

    # ------------------------------------------------------------ explain
    contrib = extras.get("contributions")
    if contrib is not None:
        story.append(Paragraph("4. Contribution of each criterion to the net flow", st["h1"]))
        rows = [["Alternative"] + names + ["Sum", "Net Phi"]]
        for i in order:
            rows.append([ids[i]] + [_fmt(v, 3) for v in contrib[i]] + [_fmt(contrib[i].sum(), 3), _fmt(result["phi_net"][i], 3)])
        story.append(_table(rows, font=6.5))
        story.append(Paragraph("Contributions are COA(Phi+_k) - COA(Phi-_k) per criterion; defuzzification is not exactly additive, so the sum can differ slightly from the net flow.", st["small"]))

    # ------------------------------------------------------------ agreement
    ag = extras.get("agreement")
    if ag is not None:
        story.append(Paragraph("5. Agreement among decision makers", st["h1"]))
        story.append(Paragraph(_safe(f"Kendall W = {_fmt(ag['kendall_w'], 3)} ({ag['kendall_w_label']} agreement)."
                                      + (f" Low agreement with the others: {', '.join(ag['flagged'])}." if ag["flagged"] else "")), st["body"]))
        m = len(ag["expert_names"])
        rows = [["Spearman"] + [str(x) for x in ag["expert_names"]]]
        for a in range(m):
            rows.append([str(ag["expert_names"][a])] + [_fmt(ag["spearman"][a, b], 2) for b in range(m)])
        story.append(_table(rows))
        loo = extras.get("loo")
        if loo:
            story.append(Paragraph("Leave-one-decision-maker-out", st["h2"]))
            rows = [["Removed", "Same full ranking", "Same first place", "Spearman vs full"]]
            for r in loo:
                rows.append([str(r["removed"]), "yes" if r["same_order"] else "no", "yes" if r["same_first"] else "no", _fmt(r["spearman_vs_full"], 3)])
            story.append(_table(rows))

    # ------------------------------------------------------------ sensitivity
    sens = extras.get("sensitivity")
    if sens:
        story.append(Paragraph("6. Weight sensitivity", st["h1"]))
        story.append(Paragraph("Range of each criterion's weight share (other weights rescaled proportionally) inside which the ranking does not change.", st["body"]))
        rows = [["Criterion", "Current share", "Full ranking stable", "First place stable"]]
        for s in sens:
            rows.append([
                names[s["criterion"]], f"{s['share_now']:.1%}",
                f"{s['order_lower']:.1%} - {s['order_upper']:.1%}",
                f"{s['top_lower']:.1%} - {s['top_upper']:.1%}",
            ])
        story.append(_table(rows))
        rev = extras.get("reversal")
        if rev:
            story.append(Paragraph("Rank-reversal test (removing one alternative)", st["h2"]))
            rows = [["Removed", "Reversed pairs"]]
            for r in rev:
                pairs = "; ".join(f"{ids[a]} vs {ids[b]}" for a, b in r["reversals"]) or "none"
                rows.append([ids[r["removed"]], pairs])
            story.append(_table(rows, [80, 380]))

    # ------------------------------------------------------------ Monte Carlo
    mc = extras.get("mc")
    if mc is not None:
        story.append(Paragraph("7. Monte Carlo stability", st["h1"]))
        params = extras.get("mc_params", {})
        story.append(Paragraph(_safe(
            f"{mc['n_mc']:,} simulations · seed {mc['seed']} · weight sampling: {mc.get('weight_sampling', '')}. "
            + (f"Continuous variation (%): {params.get('variation_text', '')}" if params.get("variation_text") else "")
        ), st["body"]))
        order_mc = np.argsort(mc["mean_rank"])
        rows = [["Alternative", "Mean position", "Position SD", "1st-place %", "Mean Phi", "Phi SD"]]
        for i in order_mc:
            rows.append([ids[i], _fmt(mc["mean_rank"][i], 2), _fmt(mc["std_rank"][i], 2), _fmt(100 * mc["freq_top1"][i], 1), _fmt(mc["mean_phi"][i]), _fmt(mc["std_phi"][i])])
        story.append(_table(rows))
        story.append(Paragraph("Rank acceptability (% of simulations in each position)", st["h2"]))
        acc = mc["rank_acceptability"]
        rows = [["Alternative"] + [str(r + 1) for r in range(n)]]
        for i in order_mc:
            rows.append([ids[i]] + [_fmt(acc[i, r], 1) for r in range(n)])
        story.append(_table(rows, [60] + [max(22, 380 // n)] * n))

    # ------------------------------------------------------------ FlowSort
    fs = extras.get("flowsort")
    if fs is not None:
        story.append(Paragraph("8. Classification (FlowSort)", st["h1"]))
        labels = fs["class_names"]
        prof = np.asarray(fs["profiles"], dtype=float)
        story.append(Paragraph("Limiting profiles (worst boundary first)", st["h2"]))
        rows = [["Profile"] + names]
        for h in range(prof.shape[0]):
            rows.append([f"P{h + 1}"] + [f"{v:g}" for v in prof[h]])
        story.append(_table(rows))
        story.append(Paragraph("Assignment", st["h2"]))
        head = ["ID", "Class", "Net flow in R+{a}"]
        acc = fs.get("class_acceptability")
        if acc is not None:
            head += [f"% {l}" for l in labels]
        rows = [head]
        for i in range(n):
            row = [ids[i], labels[int(fs["class_index"][i])], _fmt(fs["phi"][i], 3)]
            if acc is not None:
                row += [_fmt(acc[i, c], 1) for c in range(len(labels))]
            rows.append(row)
        story.append(_table(rows))

    # ------------------------------------------------------------ portfolio
    pf = extras.get("portfolio")
    if pf is not None:
        story.append(Paragraph("9. Portfolio selection (PROMETHEE V)", st["h1"]))
        if not pf["solution"]["feasible"]:
            story.append(Paragraph("No feasible portfolio for the stated constraints.", st["body"]))
        else:
            sol, base = pf["solution"], pf.get("greedy")
            shifted = pf.get("mode") == "shifted"
            story.append(Paragraph(_safe(
                "Objective: " + ("shifted net flow (net flow minus the lowest net flow)." if shifted else "net flow (classic PROMETHEE V).")
            ), st["small"]))
            story.append(Paragraph(_safe(
                f"Optimal portfolio: {sol['count']} alternative(s), total {'value' if shifted else 'net flow'} {sol['objective']:.4f}"
                + (f" (ranking-order baseline: {base['objective']:.4f})." if base else ".")
            ), st["body"]))
            for label, use, bud in zip(pf["resource_names"], sol["usage"], sol["budgets"]):
                story.append(Paragraph(_safe(f"{label}: {use:,.2f} used of {bud:,.2f}"), st["small"]))
            rows = [["ID", "Description", "Net Phi", "Selected"]]
            for i in order:
                rows.append([ids[i], str(data["descriptions"][i]), _fmt(result["phi_net"][i], 3), "yes" if sol["selected"][i] else ""])
            story.append(_table(rows, [40, 260, 60, 60]))

    # ------------------------------------------------------------ reproducibility
    if config_json:
        story.append(Paragraph("Reproducibility — configuration file", st["h1"]))
        text = config_json.decode("utf-8") if isinstance(config_json, bytes) else str(config_json)
        story.append(Paragraph(_safe(text).replace("\n", "<br/>").replace(" ", "&nbsp;"), st["mono"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm, bottomMargin=16 * mm, title="E-PROM report")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
