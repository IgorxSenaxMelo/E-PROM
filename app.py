from pathlib import Path
import hashlib
import io
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------
# Self-check of the core/ package. If core/ and app.py come from different
# releases (partial upload) or Streamlit kept an old module in memory, show a
# clear message instead of a cryptic ImportError.
# ---------------------------------------------------------------
import importlib

CORE_VERSION_REQUIRED = "8.0.1"

# Modules in dependency order, with the names app.py needs from each.
_CORE_REQUIRED = [
    ("core.labels", ["CONTINUOUS", "ORDINAL"]),
    ("core.fuzzy_numbers", ["defuzz_coa"]),
    ("core.promethee", ["promethee_i_relation"]),
    ("core.fahp", []),
    ("core.e_prom_core", ["run_e_prom"]),
    ("core.flowsort", ["run_flowsort", "suggest_profiles"]),
    ("core.monte_carlo", ["run_e_prom_monte_carlo"]),
    ("core.analysis", [
        "criterion_contributions", "expert_agreement", "flow_terms",
        "leave_one_expert_out", "pairwise_comparison", "rank_reversal_test",
        "weight_sensitivity",
    ]),
    ("core.data_loader", ["read_expert_workbook", "read_fahp_workbook"]),
    ("core.validation", [
        "summarize", "validate_against_config",
        "validate_evaluation_workbook", "validate_fahp_workbook",
    ]),
    ("core.portfolio", [
        "OBJECTIVE_MODES", "greedy_by_rank", "portfolio_values", "solve_portfolio",
    ]),
    ("core.config_io", [
        "build_config", "criteria_defaults", "dumps",
        "expert_weight_defaults", "parse_config",
    ]),
    ("core.thresholds", ["preference_curve", "suggest_thresholds"]),
    ("core.report", ["build_report"]),
]


def _core_problems(reload=False):
    problems = []
    try:
        core_pkg = importlib.import_module("core")
        if reload:
            core_pkg = importlib.reload(core_pkg)
    except Exception as exc:
        return [f"The core/ package could not be imported: {exc}"]

    found = getattr(core_pkg, "__version__", "unknown")
    if found != CORE_VERSION_REQUIRED:
        problems.append(
            f"core/__init__.py reports version {found}; app.py needs {CORE_VERSION_REQUIRED}."
        )

    for module_name, names in _CORE_REQUIRED:
        try:
            module = importlib.import_module(module_name)
            if reload:
                module = importlib.reload(module)
        except Exception as exc:
            problems.append(f"{module_name}: cannot be imported ({type(exc).__name__}: {exc}).")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(
                f"{module_name} ({getattr(module, '__file__', '?')}) is missing: "
                + ", ".join(missing)
            )
    return problems


_core_issues = _core_problems()
if _core_issues:
    # Streamlit keeps imported modules in memory between reruns: try to reload them.
    _core_issues = _core_problems(reload=True)

if _core_issues:
    st.set_page_config(page_title="E-PROM", page_icon="⚖️", layout="wide")
    st.error(
        "The files of this app are inconsistent (app.py and the core/ folder come "
        "from different versions)."
    )
    st.markdown(
        "**What to do:** upload the **whole** `core/` folder from the latest release "
        "(not only some files) together with `app.py`, `GUIDE.md`, `requirements.txt` "
        "and the `templates/` folder, commit, and then use *Manage app → Reboot app*."
    )
    st.code("\n".join(_core_issues))
    st.stop()


from core.analysis import (
    criterion_contributions,
    expert_agreement,
    flow_terms,
    leave_one_expert_out,
    pairwise_comparison,
    rank_reversal_test,
    weight_sensitivity,
)
from core.config_io import (
    build_config,
    criteria_defaults,
    dumps,
    expert_weight_defaults,
    parse_config,
)
from core.data_loader import (
    read_expert_workbook,
    read_fahp_workbook,
)
from core.e_prom_core import run_e_prom
from core.flowsort import run_flowsort, suggest_profiles
from core.fuzzy_numbers import defuzz_coa
from core.labels import CONTINUOUS, ORDINAL
from core.monte_carlo import run_e_prom_monte_carlo
from core.portfolio import (
    OBJECTIVE_MODES,
    greedy_by_rank,
    portfolio_values,
    solve_portfolio,
)
from core.promethee import promethee_i_relation
from core.report import build_report
from core.thresholds import preference_curve, suggest_thresholds
from core.validation import (
    summarize,
    validate_against_config,
    validate_evaluation_workbook,
    validate_fahp_workbook,
)

APP_VERSION = "E-PROM v8.0.1"


st.set_page_config(
    page_title="E-PROM — Fuzzy Preference Ranking",
    page_icon="⚖️",
    layout="wide",
)


PREF_CODES = {
    "Usual": 1,
    "U-shape": 2,
    "V-shape": 3,
    "Level": 4,
    "Linear": 5,
}


def save_uploaded(f):
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(f.name).suffix or ".xlsx",
    )
    tmp.write(f.getvalue())
    tmp.close()
    return Path(tmp.name)


ISSUE_ICONS = {"error": "❌ error", "warning": "⚠️ warning", "info": "ℹ️ info"}


def show_issues(issues, expanded=None):
    """Show the input-validation report."""
    if not issues:
        return
    counts = summarize(issues)
    table = pd.DataFrame(issues).rename(
        columns={"severity": "Severity", "sheet": "Sheet", "cell": "Cell", "message": "Problem"}
    )
    table["Severity"] = table["Severity"].map(ISSUE_ICONS)
    title = (
        f"🔎 Input validation — {counts['errors']} error(s), "
        f"{counts['warnings']} warning(s), {counts['info']} note(s)"
    )
    with st.expander(
        title, expanded=bool(counts["errors"]) if expanded is None else expanded
    ):
        st.dataframe(table, use_container_width=True, hide_index=True)


def result_df(data, result):
    return pd.DataFrame({
        "Rank": np.asarray(
            result["rank_e_prom"],
            dtype=int,
        ),
        "Alternative": data["ids"],
        "Description": data["descriptions"],
        "Phi+": result["phi_plus"],
        "Phi-": result["phi_minus"],
        "Net Phi": result["phi_net"],
    }).sort_values(
        ["Rank", "Alternative"]
    ).reset_index(drop=True)


def excel_bytes(data, result, config, fahp):
    out = io.BytesIO()

    with pd.ExcelWriter(
        out,
        engine="openpyxl",
    ) as writer:

        result_df(
            data,
            result,
        ).to_excel(
            writer,
            sheet_name="Ranking",
            index=False,
        )

        phi = pd.DataFrame(
            result["phi_net_experts"],
            index=data["ids"],
            columns=result["expert_names"],
        )
        phi.index.name = "Alternative"

        phi.to_excel(
            writer,
            sheet_name="Phi_by_Decision_Maker",
        )

        config.to_excel(
            writer,
            sheet_name="Configuration",
            index=False,
        )

        fuzzy_w = pd.DataFrame(
            fahp["w_fuzzy_group"],
            index=data["criteria"],
            columns=["a", "b", "c", "d"],
        )
        fuzzy_w.index.name = "Criterion"

        fuzzy_w.to_excel(
            writer,
            sheet_name="Fuzzy_Weights",
        )

        precision_matrix = fahp.get(
            "precision_levels",
            [["High"] * len(data["criteria"]) for _ in fahp.get("expert_names", [])],
        )
        # Keep the exported precision table dimensionally consistent.
        # Some legacy reader paths provide fewer expert names than precision
        # rows; in that case use stable fallback names rather than changing
        # the underlying precision data.
        precision_rows = len(precision_matrix)
        expert_names = list(fahp.get("expert_names", []))

        if len(expert_names) != precision_rows:
            fallback_names = [
                f"Decision maker {i + 1}"
                for i in range(precision_rows)
            ]
            expert_names = [
                expert_names[i]
                if i < len(expert_names) and str(expert_names[i]).strip()
                else fallback_names[i]
                for i in range(precision_rows)
            ]

        precision_df = pd.DataFrame(
            precision_matrix,
            index=expert_names,
            columns=data["criteria"],
        )
        precision_df.index.name = "Decision maker"
        precision_df.to_excel(
            writer,
            sheet_name="Attribute_Precision",
        )

    out.seek(0)
    return out.getvalue()


def plot_weights(fahp, criteria):
    wf = np.asarray(
        fahp["w_fuzzy_individual"],
        dtype=float,
    )

    crisp = np.array([
        [
            defuzz_coa(wf[k, :, e])
            for e in range(wf.shape[2])
        ]
        for k in range(wf.shape[0])
    ])

    mean = crisp.mean(axis=1)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=criteria,
            y=mean,
            name="Mean weight",
        )
    )

    for e, name in enumerate(
        fahp.get(
            "expert_names",
            fahp.get(
                "sheets",
                [f"E{i + 1}" for i in range(crisp.shape[1])],
            ),
        )
    ):
        fig.add_trace(
            go.Scatter(
                x=criteria,
                y=crisp[:, e],
                mode="markers",
                name=name,
            )
        )

    fig.update_layout(
        title="Fuzzy criterion weights — FAHP-Express",
        xaxis_title="Criterion",
        yaxis_title="Weight",
        height=500,
    )

    return fig


def relation_matrix(result):
    plus = np.asarray(
        result["phi_plus"],
        dtype=float,
    )
    minus = np.asarray(
        result["phi_minus"],
        dtype=float,
    )

    n = len(plus)
    rel = np.empty((n, n), dtype=object)

    for i in range(n):
        for j in range(n):
            rel[i, j] = (
                "I" if i == j
                else promethee_i_relation(plus[i], minus[i], plus[j], minus[j])
            )

    return rel


def alternative_labels(data):
    """Return the alternative names used as chart labels.

    Calculations keep using the IDs internally. A valid description is used
    only as a visual label.
    """
    ids = [str(x) for x in data["ids"]]
    descriptions = data.get("descriptions", [])

    if descriptions is None or len(descriptions) != len(ids):
        return ids

    labels = []
    used = {}

    for alt_id, desc in zip(ids, descriptions):
        label = str(desc).strip() if desc is not None else ""
        if not label:
            label = alt_id

        # Avoid visually identical labels when two alternatives share
        # the same description.
        count = used.get(label, 0) + 1
        used[label] = count
        if count > 1:
            label = f"{label} ({alt_id})"

        labels.append(label)

    return labels


def plot_relation(data, result):
    rel = relation_matrix(result)

    mapping = {
        "P-": 0,
        "I": 1,
        "P": 2,
        "R": 3,
    }

    z = np.vectorize(
        mapping.get
    )(rel)

    labels = alternative_labels(data)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=rel,
            texttemplate="%{text}",
            zmin=0,
            zmax=3,
            colorscale=[
                [0.00, "#3333cc"],
                [0.249, "#3333cc"],
                [0.250, "#aaaaaa"],
                [0.499, "#aaaaaa"],
                [0.500, "#22aa22"],
                [0.749, "#22aa22"],
                [0.750, "#e8cc28"],
                [1.00, "#e8cc28"],
            ],
            colorbar=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["P-", "I", "P", "R"],
                title="Relation",
            ),
        )
    )

    fig.update_layout(
        title="PROMETHEE I — relations",
        height=max(750, len(labels) * 110),
        margin=dict(l=180, r=80, t=80, b=120),
    )

    return fig


def plot_rank_heatmap(data, result):
    labels = alternative_labels(data)

    df = pd.DataFrame(
        result["rank_experts"],
        index=labels,
        columns=result["expert_names"],
    )

    df["E-PROM"] = result["rank_e_prom"]

    df = df.sort_values(
        "E-PROM"
    )

    fig = go.Figure(
        go.Heatmap(
            z=df.values,
            x=list(df.columns),
            y=list(df.index),
            text=np.vectorize(
                lambda x: f"{x:g}"
            )(df.values),
            texttemplate="%{text}",
            colorscale="Blues_r",
            zmin=1,
            zmax=len(labels),
            xgap=1,
            ygap=1,
            colorbar=dict(
                title="Position"
            ),
        )
    )

    fig.update_layout(
        title="Positions by decision maker and E-PROM",
        height=max(600, len(labels) * 55),
        margin=dict(l=180, r=80, t=80, b=100),
    )

    return fig



def plot_mc_rank_acceptability(data, mc):
    labels = alternative_labels(data)
    order = np.argsort(mc["mean_rank"])
    M = mc["rank_acceptability"][order, :]

    fig = go.Figure(
        go.Heatmap(
            z=M,
            x=list(range(1, len(labels) + 1)),
            y=[labels[i] for i in order],
            text=np.vectorize(lambda x: f"{x:.1f}%")(M),
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=max(1, np.nanmax(M)),
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        title="Monte Carlo — rank acceptability",
        xaxis_title="Position",
        yaxis_title="Alternative",
        height=max(550, len(labels) * 55),
        margin=dict(l=180, r=80, t=80, b=100),
    )
    return fig


def plot_mc_outworking(data, mc):
    labels = alternative_labels(data)
    order = np.argsort(-mc["outranking_acceptability"].sum(axis=1))
    M = mc["outranking_acceptability"][np.ix_(order, order)]

    fig = go.Figure(
        go.Heatmap(
            z=M,
            x=[labels[i] for i in order],
            y=[labels[i] for i in order],
            colorscale="Blues",
            zmin=0,
            zmax=100,
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        title="Monte Carlo — outranking acceptability",
        xaxis_title="Outranked alternative",
        yaxis_title="Outranking alternative",
        height=max(700, len(labels) * 55),
        margin=dict(l=180, r=120, t=100, b=180),
    )
    return fig


def mc_summary_df(data, mc):
    return pd.DataFrame({
        "Alternative": data["ids"],
        "Mean position": mc["mean_rank"],
        "Position SD": mc["std_rank"],
        "1st-place frequency": 100 * mc["freq_top1"],
        "Mean Phi": mc["mean_phi"],
        "Phi SD": mc["std_phi"],
    }).sort_values(
        "Mean position"
    ).reset_index(drop=True)

# ================================================================
# HEADER
# ================================================================

st.title("E-PROM")
st.subheader(
    "Express Fuzzy Preference Ranking with PROMETHEE — v8.0.1"
)
st.caption(
    "FAHP-Express + Fuzzy PROMETHEE — generalized method."
)

# ================================================================
# MODELOS PARA DOWNLOAD
# ================================================================

TEMPLATES_DIR = ROOT / "templates"
EVAL_TEMPLATE = TEMPLATES_DIR / "E_PROM_Evaluations_Template.xlsx"
FAHP_TEMPLATE = TEMPLATES_DIR / "E_PROM_FAHP_Express_Template.xlsx"

st.markdown("### 📥 Download the template workbooks")
st.caption(
    "Use these files as a starting point. Enter the experts’ evaluations and, separately, the FAHP-Express references."
)

col_model_1, col_model_2 = st.columns(2)

with col_model_1:
    if EVAL_TEMPLATE.exists():
        with open(EVAL_TEMPLATE, "rb") as f:
            st.download_button(
                "⬇️ Evaluation Workbook",
                data=f.read(),
                file_name=EVAL_TEMPLATE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_eval_template",
            )
    else:
        st.warning("Evaluation template not found in the package.")

with col_model_2:
    if FAHP_TEMPLATE.exists():
        with open(FAHP_TEMPLATE, "rb") as f:
            st.download_button(
                "⬇️ FAHP-Express Workbook",
                data=f.read(),
                file_name=FAHP_TEMPLATE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_fahp_template",
            )
    else:
        st.warning("FAHP-Express template not found in the package.")

st.divider()

GUIDE_PATH = ROOT / "GUIDE.md"
GUIDE_TEXT = (
    GUIDE_PATH.read_text(encoding="utf-8")
    if GUIDE_PATH.exists()
    else "GUIDE.md was not found in the package."
)



# ================================================================
# SIDEBAR
# ================================================================

CFG_VERSION = st.session_state.get("cfg_version", 0)
LOADED = st.session_state.get(
    "loaded_cfg", {"criteria": {}, "dm_weights": None, "monte_carlo": {}}
)


def wkey(name, k=None):
    """Widget key that changes when a configuration file is loaded, so the
    widgets are re-created with the loaded defaults."""
    if k is None:
        return f"{name}_v{CFG_VERSION}"
    return f"{name}_{k}_v{CFG_VERSION}"


preview = None
fahp_preview = None
input_issues = []
types = dirs = prefs = qs = ps = scales = None
pref_names = []
config = None
expert_weights_ui = None
criteria = []
run = False

with st.sidebar:

    st.header("1. Inputs")

    req_file = st.file_uploader(
        "Evaluation workbook",
        type=["xlsx"],
    )

    fahp_file = st.file_uploader(
        "FAHP-Express workbook",
        type=["xlsx"],
    )

    if req_file is not None:

        try:

            req_path = save_uploaded(req_file)
            input_issues += validate_evaluation_workbook(req_path)

            if summarize(input_issues)["errors"] == 0:

                preview = read_expert_workbook(req_path)

                if fahp_file is not None:
                    fahp_path = save_uploaded(fahp_file)
                    fahp_issues = validate_fahp_workbook(
                        fahp_path,
                        preview["criteria"],
                        preview["expert_names"],
                    )
                    input_issues += fahp_issues

                    if summarize(fahp_issues)["errors"] == 0:
                        fahp_preview = read_fahp_workbook(
                            fahp_path,
                            preview["criteria"],
                            expert_names=preview["expert_names"],
                        )
            else:
                st.error(
                    "The evaluation workbook has errors. "
                    "See the validation panel in the main area."
                )

        except Exception as exc:
            preview = None
            st.error(f"Error reading the workbook: {exc}")

    if preview is not None:

        try:

            criteria = preview["criteria"]
            ncrit = len(criteria)

            st.success(
                f"{preview['n_experts']} decision makers · "
                f"{preview['n_requirements']} alternatives · "
                f"{ncrit} criteria"
            )

            if fahp_file is not None and fahp_preview is None:
                st.error(
                    "The FAHP-Express workbook has errors. "
                    "See the validation panel in the main area."
                )

            st.divider()
            st.header("2. Criterion configuration")

            with st.expander("💾 Save / load configuration"):

                cfg_up = st.file_uploader(
                    "Load configuration (.json)",
                    type=["json"],
                    key="cfg_upload",
                )

                if cfg_up is not None:
                    raw_cfg = cfg_up.getvalue()
                    digest = hashlib.md5(raw_cfg).hexdigest()

                    if st.session_state.get("cfg_digest") != digest:
                        try:
                            cfg = parse_config(raw_cfg)
                            defaults, warns = criteria_defaults(cfg, criteria)
                            st.session_state["loaded_cfg"] = {
                                "criteria": defaults,
                                "dm_weights": (
                                    expert_weight_defaults(
                                        cfg, preview["expert_names"]
                                    )
                                    if "decision_makers" in cfg
                                    else None
                                ),
                                "monte_carlo": cfg.get("monte_carlo", {}),
                            }
                            st.session_state["cfg_warnings"] = warns
                            st.session_state["cfg_digest"] = digest
                            st.session_state["cfg_version"] = CFG_VERSION + 1
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))

                    for message in st.session_state.get("cfg_warnings", []):
                        st.warning(message)

                cfg_slot = st.container()

            st.info(
                "Ordinal: Usual or Level. Continuous: U-shape, V-shape, or Linear."
            )

            dirs = []
            types = []
            prefs = []
            qs = []
            ps = []
            scales = []
            pref_names = []
            rows = []

            for k, criterion in enumerate(criteria):

                d = LOADED["criteria"].get(k, {})
                d_q = float(d.get("q") or 0.0)
                d_p = float(d.get("p") or 0.0)

                st.markdown(f"### {criterion}")

                type_options = [ORDINAL, CONTINUOUS]
                typ = st.selectbox(
                    "Data type",
                    type_options,
                    index=type_options.index(d.get("type", ORDINAL)),
                    key=wkey("type", k),
                )

                direction_options = ["Maximize", "Minimize"]
                direction_name = st.selectbox(
                    "Direction",
                    direction_options,
                    index=direction_options.index(d.get("direction", "Maximize")),
                    key=wkey("direction", k),
                )

                if typ == ORDINAL:

                    scale_options = [5, 7, 9]
                    scale = st.selectbox(
                        "Ordinal scale",
                        scale_options,
                        index=scale_options.index(d.get("scale") or 7),
                        key=wkey("scale", k),
                    )

                    pref_options = ["Usual", "Level"]
                    pref_name = st.selectbox(
                        "Preference function",
                        pref_options,
                        index=(
                            pref_options.index(d["preference"])
                            if d.get("preference") in pref_options
                            else 0
                        ),
                        key=wkey("pref", k),
                    )

                    if pref_name == "Usual":
                        q = 0.0
                        p = 0.0
                    else:

                        q = st.number_input(
                            "q — indifference",
                            min_value=0.0,
                            value=d_q,
                            step=1.0,
                            key=wkey("q", k),
                        )

                        p = st.number_input(
                            "p — preference",
                            min_value=float(q + 1e-9),
                            value=float(max(d_p, q + 1.0)),
                            step=1.0,
                            key=wkey("p", k),
                        )

                else:

                    scale = 0
                    st.caption(
                        "Continuous values are represented by a trapezoidal fuzzy "
                        "number in the original unit. Its width is set by the "
                        "attribute precision given in the FAHP-Express workbook. "
                        "Express q and p in the original unit."
                    )

                    pref_options = ["U-shape", "V-shape", "Linear"]
                    pref_name = st.selectbox(
                        "Preference function",
                        pref_options,
                        index=(
                            pref_options.index(d["preference"])
                            if d.get("preference") in pref_options
                            else 0
                        ),
                        key=wkey("pref", k),
                    )

                    if pref_name == "U-shape":

                        q = st.number_input(
                            "q — indifference threshold",
                            min_value=0.0,
                            value=d_q,
                            step=1.0,
                            key=wkey("q", k),
                            help=(
                                "Differences up to q are treated as indifference. "
                                "Use the criterion's original unit."
                            ),
                        )

                        p = 0.0

                    elif pref_name == "V-shape":

                        q = 0.0

                        p = st.number_input(
                            "p — preference threshold",
                            min_value=1e-9,
                            value=d_p if d_p > 0 else 1.0,
                            step=1.0,
                            key=wkey("p", k),
                            help=(
                                "Preference grows linearly up to p and is "
                                "complete from p on."
                            ),
                        )

                    else:  # Linear

                        q = st.number_input(
                            "q — indifference threshold",
                            min_value=0.0,
                            value=d_q,
                            step=1.0,
                            key=wkey("q", k),
                        )

                        p = st.number_input(
                            "p — preference threshold",
                            min_value=float(q + 1e-9),
                            value=float(max(d_p, q + 1.0)),
                            step=1.0,
                            key=wkey("p", k),
                        )

                dirs.append(1 if direction_name == "Maximize" else -1)
                types.append(typ)
                prefs.append(PREF_CODES[pref_name])
                pref_names.append(pref_name)
                qs.append(q)
                ps.append(p)
                scales.append(scale)

                rows.append({
                    "Criterion": criterion,
                    "Type": typ,
                    "Direction": direction_name,
                    "Preference": pref_name,
                    "q": q,
                    "p": p,
                    "Scale": scale if typ == ORDINAL else "Continuous",
                })

            config = pd.DataFrame(rows)

            with st.expander("⚖️ Decision-maker weights (optional)"):
                st.caption(
                    "Equal weights by default. Raise the weight of a decision "
                    "maker to give their judgments more influence on the group result."
                )
                saved_w = LOADED.get("dm_weights") or [1.0] * preview["n_experts"]
                expert_weights_ui = [
                    st.number_input(
                        str(name),
                        min_value=0.0,
                        value=float(saved_w[e]),
                        step=0.1,
                        key=wkey("dmw", e),
                    )
                    for e, name in enumerate(preview["expert_names"])
                ]

            st.divider()
            st.header("3. Execution")

            run = st.button(
                "▶ Run E-PROM",
                type="primary",
                use_container_width=True,
            )

            mc_saved = {
                "n_mc": st.session_state.get(wkey("mc_n")),
                "seed": st.session_state.get(wkey("mc_seed")),
                "weight_sampling": st.session_state.get(wkey("mc_weight_mode")),
            }
            cfg_slot.download_button(
                "⬇️ Save current configuration",
                data=dumps(
                    build_config(
                        criteria, types, dirs, pref_names, qs, ps, scales,
                        mc_variation=st.session_state.get("mc_variation"),
                        expert_weights=expert_weights_ui,
                        expert_names=preview["expert_names"],
                        monte_carlo={k: v for k, v in mc_saved.items() if v is not None},
                    )
                ),
                file_name="E_PROM_configuration.json",
                mime="application/json",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Error building the configuration: {exc}"
            )
            types = None
            run = False


# ================================================================
# INPUT VALIDATION + THRESHOLD ASSISTANT + GUIDE (main area)
# ================================================================

if req_file is not None:
    show_issues(input_issues)

if preview is not None and types is not None:

    with st.expander("🧭 Threshold assistant — suggested q and p", expanded=False):

        st.caption(
            "Starting points computed from your data and the attribute precision. "
            "Continuous criteria: q covers the fuzzy width of two equal values and p is the "
            "75th percentile of the pairwise differences. Ordinal criteria: Level function "
            "with q / p at the 25th / 75th percentiles of the pairwise differences. "
            "Review them before applying."
        )

        if fahp_preview is not None:
            precision_for_assistant = fahp_preview["precision_levels"]
        else:
            precision_for_assistant = [
                ["Medium"] * len(criteria) for _ in range(preview["n_experts"])
            ]
            st.caption(
                "FAHP-Express workbook not loaded: Medium precision is assumed."
            )

        try:
            suggestions = suggest_thresholds(
                preview["evaluations"], types, scales, precision_for_assistant
            )
        except Exception as exc:
            suggestions = None
            st.warning(f"Could not compute suggestions: {exc}")

        if suggestions:

            st.dataframe(
                pd.DataFrame({
                    "Criterion": criteria,
                    "Current": [
                        f"{pref_names[k]} (q={qs[k]:g}, p={ps[k]:g})"
                        for k in range(len(criteria))
                    ],
                    "Suggested": [
                        f"{s['preference']} (q={s['q']:g}, p={s['p']:g})"
                        for s in suggestions
                    ],
                    "Why": [s["rationale"] for s in suggestions],
                }),
                use_container_width=True,
                hide_index=True,
            )

            curve_name = st.selectbox(
                "Preview the preference function of",
                criteria,
                key="assistant_curve",
            )
            ck = criteria.index(curve_name)
            sug = suggestions[ck]
            d_max = 1.5 * max(ps[ck], sug["p"], qs[ck], sug["q"], 1.0)

            fig_curve = go.Figure()
            if pref_names[ck] in PREF_CODES and (
                pref_names[ck] != "Level" or ps[ck] > qs[ck]
            ):
                x_cur, y_cur = preference_curve(pref_names[ck], qs[ck], ps[ck], d_max)
                fig_curve.add_trace(go.Scatter(x=x_cur, y=y_cur, mode="lines", name="Current"))
            x_sug, y_sug = preference_curve(sug["preference"], sug["q"], sug["p"], d_max)
            fig_curve.add_trace(
                go.Scatter(x=x_sug, y=y_sug, mode="lines", name="Suggested", line={"dash": "dash"})
            )
            fig_curve.update_layout(
                height=280,
                xaxis_title="Difference d between two alternatives",
                yaxis_title="Preference P(d)",
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
            st.plotly_chart(fig_curve, use_container_width=True)

            chosen = st.multiselect(
                "Criteria to update",
                criteria,
                default=criteria,
                key="assistant_chosen",
            )

            if st.button("Apply suggestions to the selected criteria", key="assistant_apply"):

                new_entries = {}
                for k in range(len(criteria)):
                    entry = dict(LOADED["criteria"].get(k, {}))
                    entry.update(
                        type=types[k],
                        direction="Maximize" if dirs[k] == 1 else "Minimize",
                        scale=scales[k] if types[k] == ORDINAL else entry.get("scale", 7),
                        preference=pref_names[k],
                        q=qs[k],
                        p=ps[k],
                    )
                    if criteria[k] in chosen:
                        entry.update(
                            preference=suggestions[k]["preference"],
                            q=suggestions[k]["q"],
                            p=suggestions[k]["p"],
                        )
                    new_entries[k] = entry

                st.session_state["loaded_cfg"] = {**LOADED, "criteria": new_entries}
                st.session_state["cfg_version"] = CFG_VERSION + 1
                st.rerun()

# ================================================================
# EXECUTION
# ================================================================

ANALYSIS_KEYS = [
    "mc", "mc_params", "terms", "sens", "loo", "reversal",
    "flowsort", "portfolio", "report_pdf", "rob_done",
]

if run:

    if fahp_file is None:
        st.error(
            "Upload the FAHP-Express workbook."
        )
        st.stop()

    if fahp_preview is None:
        st.error(
            "The FAHP-Express workbook has errors. "
            "Fix them (see the validation panel) and run again."
        )
        st.stop()

    if sum(expert_weights_ui) <= 0:
        st.error("At least one decision maker needs a weight greater than zero.")
        st.stop()

    config_issues = validate_against_config(preview, types, scales)
    if summarize(config_issues)["errors"]:
        show_issues(config_issues, expanded=True)
        st.error(
            "Some values do not match the configured data types / scales. "
            "Fix the workbook or the configuration."
        )
        st.stop()
    show_issues(config_issues)

    try:

        with st.spinner(
            "Running FAHP-Express + Fuzzy PROMETHEE..."
        ):

            data = preview
            fahp = fahp_preview

            for message in fahp.get("warnings", []):
                st.warning(message)

            result = run_e_prom(
                data["evaluations"],
                fahp["w_fuzzy_individual"],
                types,
                dirs,
                prefs,
                qs,
                ps,
                scales,
                precision_levels=fahp["precision_levels"],
                expert_weights=expert_weights_ui,
            )

            ranking = result_df(
                data,
                result,
            )

            export_config = config.copy()
            xlsx = excel_bytes(
                data,
                result,
                export_config,
                fahp,
            )

        for key in ANALYSIS_KEYS:
            st.session_state.pop(key, None)

        st.session_state.update(
            data=data,
            fahp=fahp,
            run_params=dict(
                types=list(types),
                dirs=list(dirs),
                prefs=list(prefs),
                qs=list(qs),
                ps=list(ps),
                scales=list(scales),
                pref_names=list(pref_names),
                expert_weights=list(expert_weights_ui),
            ),
            result=result,
            config=config,
            ranking=ranking,
            xlsx=xlsx,
        )

        st.success(
            "E-PROM completed successfully."
        )

    except Exception as exc:
        st.exception(exc)
        st.stop()


if "result" not in st.session_state:
    with st.expander("📘 User guide", expanded=True):
        st.markdown(GUIDE_TEXT)


# ================================================================
# RESULTS
# ================================================================

if "result" in st.session_state:

    data = st.session_state["data"]
    fahp = st.session_state["fahp"]
    run_params = st.session_state["run_params"]
    result = st.session_state["result"]
    config = st.session_state["config"]
    ranking = st.session_state["ranking"]

    st.divider()

    a, b, c = st.columns(3)

    a.metric(
        "Decision makers",
        data["n_experts"],
    )

    b.metric(
        "Alternatives",
        data["n_requirements"],
    )

    c.metric(
        "Criteria",
        data["n_criteria"],
    )

    EW = run_params["expert_weights"]
    PL = fahp["precision_levels"]
    ARGS = (
        data["evaluations"],
        fahp["w_fuzzy_individual"],
        run_params["types"],
        run_params["dirs"],
        run_params["prefs"],
        run_params["qs"],
        run_params["ps"],
        run_params["scales"],
    )
    labels = alternative_labels(data)
    rank_order = np.argsort(
        np.asarray(result["rank_e_prom"], dtype=float), kind="mergesort"
    )

    if "terms" not in st.session_state:
        st.session_state["terms"] = flow_terms(
            *ARGS, precision_levels=PL, expert_weights=EW
        )
    terms = st.session_state["terms"]

    (
        tab1, tab2, tab3, tab_explain, tab_sens,
        tab_mc, tab_flow, tab_port, tab_report, tab_guide,
    ) = st.tabs(
        [
            "🏆 Ranking",
            "📈 PROMETHEE",
            "👥 Decision Makers",
            "🔍 Explain",
            "🧪 Sensitivity",
            "🎲 Monte Carlo",
            "🗂️ Classification",
            "📦 Portfolio",
            "📄 Report",
            "📘 Guide",
        ]
    )

    with tab1:

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download results as Excel",
            data=st.session_state["xlsx"],
            file_name="E_PROM_Results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    with tab2:

        st.subheader("Fuzzy criterion weights")
        st.caption(
            "Weights obtained by FAHP-Express. Fuzzy numbers are preserved during Fuzzy PROMETHEE and are not defuzzified at this stage."
        )
        fuzzy_weights_df = pd.DataFrame(
            np.asarray(fahp["w_fuzzy_group"], dtype=float),
            index=data["criteria"],
            columns=["a", "b", "c", "d"],
        )
        fuzzy_weights_df.index.name = "Criterion"
        st.dataframe(
            fuzzy_weights_df.style.format("{:.6f}"),
            use_container_width=True,
        )

        st.subheader("Attribute precision")
        precision_matrix = fahp.get(
            "precision_levels",
            [["High"] * len(data["criteria"]) for _ in fahp.get("expert_names", [])],
        )
        # Keep the exported precision table dimensionally consistent.
        precision_rows = len(precision_matrix)
        expert_names = list(fahp.get("expert_names", []))

        if len(expert_names) != precision_rows:
            fallback_names = [
                f"Decision maker {i + 1}"
                for i in range(precision_rows)
            ]
            expert_names = [
                expert_names[i]
                if i < len(expert_names) and str(expert_names[i]).strip()
                else fallback_names[i]
                for i in range(precision_rows)
            ]

        precision_df = pd.DataFrame(
            precision_matrix,
            index=expert_names,
            columns=data["criteria"],
        )
        precision_df.index.name = "Decision maker"
        st.dataframe(precision_df, use_container_width=True)
        st.caption(
            "Precision is defined individually for each decision maker and criterion. "
            "High = lower uncertainty, Medium = intermediate, and Low = higher "
            "uncertainty. For continuous criteria, the original unit is preserved; for "
            "ordinal criteria, fuzzy width is adjusted in scale units, "
            "respecting their limits."
        )

        st.plotly_chart(
            plot_weights(
                fahp,
                data["criteria"],
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_relation(
                data,
                result,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_rank_heatmap(
                data,
                result,
            ),
            use_container_width=True,
        )

    with tab3:

        phi_df = pd.DataFrame(
            result["phi_net_experts"],
            index=data["ids"],
            columns=result["expert_names"],
        )

        phi_df.index.name = "Alternative"

        st.subheader(
            "Net Phi by decision maker"
        )

        st.dataframe(
            phi_df,
            use_container_width=True,
        )

        if data["n_experts"] >= 2:

            ag = expert_agreement(
                result["phi_net_experts"], result["expert_names"]
            )

            st.subheader("Agreement among decision makers")

            m1, m2 = st.columns(2)
            m1.metric(
                "Kendall's W",
                "n/a" if np.isnan(ag["kendall_w"]) else f"{ag['kendall_w']:.3f}",
                ag["kendall_w_label"] + " agreement",
                delta_color="off",
            )
            m2.metric(
                "Lowest mean Spearman",
                (
                    "n/a"
                    if np.all(np.isnan(ag["mean_agreement"]))
                    else f"{np.nanmin(ag['mean_agreement']):.3f}"
                ),
            )

            if ag["flagged"]:
                st.warning(
                    "Low agreement with the other decision makers: "
                    + ", ".join(ag["flagged"])
                    + ". Consider discussing their judgments before aggregating."
                )

            fig_sp = go.Figure(
                go.Heatmap(
                    z=ag["spearman"],
                    x=ag["expert_names"],
                    y=ag["expert_names"],
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    text=np.round(ag["spearman"], 2),
                    texttemplate="%{text}",
                )
            )
            fig_sp.update_layout(
                height=320,
                title="Spearman correlation between individual rankings",
                margin={"l": 10, "r": 10, "t": 40, "b": 10},
            )
            st.plotly_chart(fig_sp, use_container_width=True)
            st.caption(
                "Rule of thumb for Kendall's W: below 0.3 weak, 0.3-0.6 moderate, "
                "above 0.6 strong agreement."
            )

        st.subheader(
            "Configuration"
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

    with tab_explain:

        st.subheader("Why this ranking? Contribution of each criterion")
        st.caption(
            "Each bar is the weighted contribution of a criterion to the net flow "
            "of an alternative (positive = helps, negative = hurts). The diamond is "
            "the net flow. Defuzzification is not exactly additive, so the bars can "
            "differ slightly from the diamond."
        )

        contrib = criterion_contributions(terms)

        fig_ct = go.Figure()
        for k, criterion in enumerate(data["criteria"]):
            fig_ct.add_trace(
                go.Bar(
                    name=criterion,
                    x=[labels[i] for i in rank_order],
                    y=contrib[rank_order, k],
                )
            )
        fig_ct.add_trace(
            go.Scatter(
                x=[labels[i] for i in rank_order],
                y=np.asarray(result["phi_net"])[rank_order],
                mode="markers",
                name="Net Phi",
                marker={"symbol": "diamond", "size": 11, "color": "black"},
            )
        )
        fig_ct.update_layout(
            barmode="relative",
            height=430,
            yaxis_title="Contribution to net Phi",
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig_ct, use_container_width=True)

        contrib_df = pd.DataFrame(
            contrib, index=data["ids"], columns=data["criteria"]
        )
        contrib_df["Sum of contributions"] = contrib.sum(axis=1)
        contrib_df["Net Phi"] = np.asarray(result["phi_net"])
        contrib_df.index.name = "Alternative"
        st.dataframe(
            contrib_df.style.format("{:.4f}"),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Pairwise comparison")
        st.caption(
            "Criterion by criterion, how much alternative A is preferred to B and "
            "B to A (fuzzy preference, weighted and defuzzified)."
        )

        pc1, pc2 = st.columns(2)
        a_label = pc1.selectbox(
            "Alternative A", labels, index=int(rank_order[0]), key="pw_a"
        )
        b_label = pc2.selectbox(
            "Alternative B",
            labels,
            index=int(rank_order[min(1, len(labels) - 1)]),
            key="pw_b",
        )
        ia, ib = labels.index(a_label), labels.index(b_label)

        if ia == ib:
            st.info("Choose two different alternatives.")
        else:
            pw = pairwise_comparison(
                *ARGS, ia, ib, precision_levels=PL, expert_weights=EW
            )
            mean_values = np.mean(
                [np.asarray(v, dtype=float) for v in data["evaluations"].values()],
                axis=0,
            )

            def _favours(x):
                if x > 1e-9:
                    return a_label
                if x < -1e-9:
                    return b_label
                return "tie"

            pw_df = pd.DataFrame({
                "Criterion": data["criteria"],
                "Weight": np.asarray(fahp["w_crisp_group"], dtype=float),
                f"Value {a_label}": mean_values[ia],
                f"Value {b_label}": mean_values[ib],
                f"Preference {a_label} over {b_label}": pw["pref_ij"],
                f"Preference {b_label} over {a_label}": pw["pref_ji"],
                "Net (A - B)": pw["net"],
                "Favours": [_favours(x) for x in pw["net"]],
            })

            t1, t2, t3 = st.columns(3)
            t1.metric(f"π({a_label}, {b_label})", f"{pw['total_ij']:.4f}")
            t2.metric(f"π({b_label}, {a_label})", f"{pw['total_ji']:.4f}")
            t3.metric("Net (A - B)", f"{pw['total_ij'] - pw['total_ji']:+.4f}")

            st.dataframe(
                pw_df.style.format(
                    {
                        "Weight": "{:.3f}",
                        f"Value {a_label}": "{:g}",
                        f"Value {b_label}": "{:g}",
                        f"Preference {a_label} over {b_label}": "{:.4f}",
                        f"Preference {b_label} over {a_label}": "{:.4f}",
                        "Net (A - B)": "{:+.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            fig_pw = go.Figure(
                go.Bar(
                    x=data["criteria"],
                    y=pw["net"],
                    marker_color=[
                        "#305496" if x >= 0 else "#C0504D" for x in pw["net"]
                    ],
                )
            )
            fig_pw.update_layout(
                height=300,
                yaxis_title=f"Net preference ({a_label} − {b_label})",
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
            st.plotly_chart(fig_pw, use_container_width=True)

    with tab_sens:

        st.subheader("Weight sensitivity")
        st.caption(
            "The importance of one criterion is raised or lowered while the other "
            "criteria keep their relative proportions. The table shows the range of "
            "that criterion's weight share inside which the result does not change; a "
            "narrow range means the ranking is sensitive to that weight."
        )

        if "sens" not in st.session_state:
            st.session_state["sens"] = weight_sensitivity(
                terms, fahp["w_fuzzy_individual"]
            )
        sens = st.session_state["sens"]

        def _range(lo, hi, changes_lo, changes_hi):
            lo_txt = f"{lo:.1%}" if changes_lo else "0%"
            hi_txt = f"{hi:.1%}" if changes_hi else "100%"
            return f"{lo_txt} – {hi_txt}"

        st.dataframe(
            pd.DataFrame({
                "Criterion": [data["criteria"][s["criterion"]] for s in sens],
                "Current share": [f"{s['share_now']:.1%}" for s in sens],
                "Full ranking unchanged for share in": [
                    _range(s["order_lower"], s["order_upper"], s["order_changes_down"], s["order_changes_up"])
                    for s in sens
                ],
                "First place unchanged for share in": [
                    _range(s["top_lower"], s["top_upper"], s["top_changes_down"], s["top_changes_up"])
                    for s in sens
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

        sens_name = st.selectbox(
            "Criterion to sweep", data["criteria"], key="sens_criterion"
        )
        sk = data["criteria"].index(sens_name)
        sw = sens[sk]

        sc1, sc2 = st.columns(2)

        fig_phi = go.Figure()
        fig_rank = go.Figure()
        for i in range(data["n_requirements"]):
            fig_phi.add_trace(
                go.Scatter(x=sw["sweep_share"], y=sw["sweep_phi"][i], mode="lines", name=labels[i])
            )
            fig_rank.add_trace(
                go.Scatter(
                    x=sw["sweep_share"], y=sw["sweep_rank"][i], mode="lines",
                    line_shape="hv", name=labels[i],
                )
            )
        for fig_s, title_s in ((fig_phi, "Net Phi"), (fig_rank, "Position")):
            fig_s.add_vline(x=sw["share_now"], line_dash="dash", line_color="grey")
            fig_s.update_layout(
                height=340,
                xaxis_title=f"Weight share of {sens_name}",
                yaxis_title=title_s,
                xaxis_tickformat=".0%",
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
        fig_rank.update_yaxes(autorange="reversed")
        sc1.plotly_chart(fig_phi, use_container_width=True)
        sc2.plotly_chart(fig_rank, use_container_width=True)

        st.divider()
        st.subheader("Robustness checks")
        st.caption(
            "Leave-one-decision-maker-out re-runs the analysis without each decision "
            "maker. The rank-reversal test removes each alternative and checks whether the "
            "relative order of the remaining ones changes."
        )

        if st.button("Run robustness checks", key="run_rob"):
            with st.spinner("Running..."):
                st.session_state["loo"] = leave_one_expert_out(
                    *ARGS, precision_levels=PL, expert_weights=EW,
                    base_rank=result["rank_e_prom"],
                )
                st.session_state["reversal"] = rank_reversal_test(
                    *ARGS, precision_levels=PL, expert_weights=EW,
                    base_rank=result["rank_e_prom"],
                )
                st.session_state["rob_done"] = True

        if st.session_state.get("rob_done"):

            loo = st.session_state.get("loo")
            if loo:
                st.markdown("**Leave one decision maker out**")
                loo_df = pd.DataFrame({
                    "Removed": [r["removed"] for r in loo],
                    "Same full ranking": ["yes" if r["same_order"] else "no" for r in loo],
                    "Same first place": ["yes" if r["same_first"] else "no" for r in loo],
                    "Spearman vs full": [r["spearman_vs_full"] for r in loo],
                })
                for i in range(data["n_requirements"]):
                    loo_df[f"Position {labels[i]}"] = [r["rank"][i] for r in loo]
                st.dataframe(loo_df, use_container_width=True, hide_index=True)
            else:
                st.info("Leave-one-out needs at least two decision makers.")

            reversal = st.session_state.get("reversal")
            if reversal:
                st.markdown("**Rank-reversal test**")
                st.dataframe(
                    pd.DataFrame({
                        "Alternative removed": [labels[r["removed"]] for r in reversal],
                        "Reversed pairs": [
                            "; ".join(f"{labels[a]} vs {labels[b]}" for a, b in r["reversals"]) or "none"
                            for r in reversal
                        ],
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("The rank-reversal test needs at least three alternatives.")

    with tab_mc:

        st.subheader("Monte Carlo — E-PROM stability")

        st.markdown(
            "For **ordinal criteria**, uncertainty is obtained empirically "
            "from decision makers’ responses. For **continuous criteria**, "
            "enter the desired percentage variation."
        )
        st.info(
            "In E-PROM, decision makers’ flows remain fuzzy during "
            "aggregation. Defuzzification occurs only after fuzzy aggregation "
            "in each simulation."
        )

        continuous_idx = [
            i for i, typ in enumerate(run_params["types"])
            if typ == CONTINUOUS
        ]

        variation = []
        for k, criterion in enumerate(data["criteria"]):
            if k in continuous_idx:
                default = float(
                    LOADED["criteria"].get(k, {}).get("mc_variation_pct") or 20.0
                )
                variation.append(
                    st.number_input(
                        f"Variation for {criterion} (%)",
                        min_value=0.0,
                        max_value=1000.0,
                        value=default,
                        step=5.0,
                        key=wkey("mc_var", k),
                        help="Applies ± percentage variation to each observed decision-maker value.",
                    )
                )
            else:
                variation.append(0.0)

        st.session_state["mc_variation"] = list(variation)

        mc_saved_cfg = LOADED.get("monte_carlo", {})
        weight_labels = [
            "Bootstrap decision makers (recommended)",
            "Independent per criterion (legacy)",
        ]
        weight_mode_label = st.selectbox(
            "Weight sampling",
            weight_labels,
            index=(
                weight_labels.index(mc_saved_cfg["weight_sampling"])
                if mc_saved_cfg.get("weight_sampling") in weight_labels
                else 0
            ),
            key=wkey("mc_weight_mode"),
            help=(
                "Recommended: each simulation resamples whole FAHP-Express "
                "rows of the decision makers, which stays consistent when they "
                "use different reference criteria. Legacy: every criterion "
                "draws its own value; only valid when all decision makers "
                "share the same reference criterion."
            ),
        )
        weight_mode = (
            "dm_bootstrap"
            if weight_mode_label.startswith("Bootstrap")
            else "independent"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            n_mc = st.number_input(
                "Number of simulations",
                min_value=100,
                max_value=500000,
                value=int(mc_saved_cfg.get("n_mc") or 2000),
                step=100,
                key=wkey("mc_n"),
            )
        with col2:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=int(mc_saved_cfg.get("seed") if mc_saved_cfg.get("seed") is not None else 42),
                step=1,
                key=wkey("mc_seed"),
            )
        with col3:
            st.metric(
                "Continuous criteria",
                len(continuous_idx),
            )

        if st.button(
            "▶ Run Monte Carlo",
            type="primary",
            key="run_mc",
        ):

            try:
                with st.spinner("Running Monte Carlo..."):
                    mc = run_e_prom_monte_carlo(
                        data["evaluations"],
                        fahp["reference_rows"],
                        run_params["types"],
                        run_params["dirs"],
                        run_params["prefs"],
                        run_params["qs"],
                        run_params["ps"],
                        run_params["scales"],
                        variation,
                        n_mc=int(n_mc),
                        seed=int(seed),
                        precision_levels=fahp["precision_levels"],
                        weight_sampling=weight_mode,
                        expert_weights=EW,
                    )
            except Exception as exc:
                st.error(f"Monte Carlo failed: {exc}")
                st.stop()

            st.session_state["mc"] = mc
            st.session_state["mc_params"] = {
                "variation_text": ", ".join(
                    f"{data['criteria'][k]} {variation[k]:g}%"
                    for k in continuous_idx
                ),
            }

        if "mc" in st.session_state:

            mc = st.session_state["mc"]

            st.success(
                f"Monte Carlo completed: {mc['n_mc']:,} simulations."
            )

            st.dataframe(
                mc_summary_df(data, mc),
                use_container_width=True,
                hide_index=True,
            )

            st.plotly_chart(
                plot_mc_rank_acceptability(data, mc),
                use_container_width=True,
            )

            st.plotly_chart(
                plot_mc_outworking(data, mc),
                use_container_width=True,
            )

            w_df = pd.DataFrame(
                mc["W_samples"],
                columns=data["criteria"],
            )

            st.subheader("Weight distribution")
            st.dataframe(
                w_df.describe().T[
                    ["mean", "std", "min", "max"]
                ],
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Download Monte Carlo results",
                data=pd.DataFrame({
                    "Alternative": data["ids"],
                    "Mean position": mc["mean_rank"],
                    "Position SD": mc["std_rank"],
                    "1st-place frequency (%)": 100 * mc["freq_top1"],
                    "Mean Phi": mc["mean_phi"],
                    "Phi SD": mc["std_phi"],
                }).to_csv(index=False).encode("utf-8-sig"),
                file_name="E_PROM_MonteCarlo.csv",
                mime="text/csv",
            )

    with tab_flow:

        st.subheader("Classification — FlowSort")
        st.markdown(
            "Assign each alternative to **ordered classes**. Classes are separated by "
            "**limiting profiles** (reference alternatives written in the same "
            "units as your data). Each alternative is compared with the profiles using "
            "the same fuzzy preferences, weights and thresholds as the ranking, and "
            "falls into the class whose boundaries enclose its net flow."
        )

        n_classes = int(
            st.number_input(
                "Number of classes",
                min_value=2,
                max_value=6,
                value=3,
                step=1,
                key="fs_n",
            )
        )

        default_names = [
            f"Class {i + 1}"
            + (" (best)" if i == 0 else " (worst)" if i == n_classes - 1 else "")
            for i in range(n_classes)
        ]
        name_cols = st.columns(n_classes)
        class_names = [
            name_cols[i].text_input(
                f"Class {i + 1}",
                value=default_names[i],
                key=f"fs_name_{n_classes}_{i}",
            )
            for i in range(n_classes)
        ]

        st.markdown(
            "**Limiting profiles** — one row per boundary, from the **worst** boundary "
            "(P1) to the **best** boundary. Ordinal values must be integers inside the scale. "
            "The defaults are quantiles of your data."
        )

        fs_reset = st.session_state.get("fs_reset", 0)
        suggested_profiles = suggest_profiles(
            data["evaluations"],
            run_params["types"],
            run_params["dirs"],
            run_params["scales"],
            n_classes,
        )
        m_prof = n_classes - 1
        profile_index = [
            f"P{h + 1}"
            + (" (worst boundary)" if h == 0 and m_prof > 1
               else " (best boundary)" if h == m_prof - 1 and m_prof > 1 else "")
            for h in range(m_prof)
        ]
        profiles_df = st.data_editor(
            pd.DataFrame(
                suggested_profiles,
                columns=data["criteria"],
                index=profile_index,
            ),
            key=f"fs_prof_{n_classes}_{fs_reset}",
            use_container_width=True,
        )

        if st.button("Reset profiles to the suggested values", key="fs_reset_btn"):
            st.session_state["fs_reset"] = fs_reset + 1
            st.rerun()

        with_prob = st.checkbox(
            "Also estimate class probabilities (Monte Carlo, uses the settings of the Monte Carlo tab)",
            value=True,
            key="fs_prob",
        )

        if st.button("▶ Classify", type="primary", key="run_fs"):

            try:
                with st.spinner("Running FlowSort..."):
                    profile_values = profiles_df.to_numpy(dtype=float)
                    fs = run_flowsort(
                        data["evaluations"],
                        fahp["w_fuzzy_individual"],
                        profile_values,
                        run_params["types"],
                        run_params["dirs"],
                        run_params["prefs"],
                        run_params["qs"],
                        run_params["ps"],
                        run_params["scales"],
                        precision_levels=PL,
                        expert_weights=EW,
                    )
                    fs["profiles"] = profile_values
                    fs["class_names"] = class_names

                    if with_prob:
                        mc_fs = run_e_prom_monte_carlo(
                            data["evaluations"],
                            fahp["reference_rows"],
                            run_params["types"],
                            run_params["dirs"],
                            run_params["prefs"],
                            run_params["qs"],
                            run_params["ps"],
                            run_params["scales"],
                            variation,
                            n_mc=int(n_mc),
                            seed=int(seed),
                            precision_levels=PL,
                            weight_sampling=weight_mode,
                            expert_weights=EW,
                            flowsort_profiles=profile_values,
                        )
                        fs["class_acceptability"] = mc_fs["class_acceptability"]

                    st.session_state["flowsort"] = fs
            except Exception as exc:
                st.error(f"Classification failed: {exc}")

        if "flowsort" in st.session_state:

            fs = st.session_state["flowsort"]
            fs_names = fs["class_names"]

            if not fs["profile_order_ok"]:
                st.warning(
                    "The profiles are not ordered consistently: their net flows do not "
                    "increase from P1 to the last profile. Each profile should be "
                    "better than the previous one on the weighted criteria, otherwise the "
                    "classes are ill-defined."
                )

            fs_table = pd.DataFrame({
                "Alternative": labels,
                "Class": [fs_names[c] for c in fs["class_index"]],
                "Net flow in R + {a}": fs["phi"],
            })
            for e, dm_name in enumerate(fs["expert_names"]):
                fs_table[f"Class by {dm_name}"] = [
                    fs_names[c] for c in fs["class_per_dm"][e]
                ]
            if "class_acceptability" in fs:
                for c, cname in enumerate(fs_names):
                    fs_table[f"% {cname}"] = fs["class_acceptability"][:, c]

            fs_table = fs_table.assign(_c=fs["class_index"]).sort_values(
                ["_c", "Net flow in R + {a}"], ascending=[True, False]
            ).drop(columns="_c")

            st.dataframe(
                fs_table.style.format(
                    {c: "{:.1f}" for c in fs_table.columns if c.startswith("% ")}
                    | {"Net flow in R + {a}": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            counts = np.bincount(fs["class_index"], minlength=len(fs_names))
            fig_cls = go.Figure(go.Bar(x=fs_names, y=counts, marker_color="#305496"))
            fig_cls.update_layout(
                height=280,
                yaxis_title="Alternatives",
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
            st.plotly_chart(fig_cls, use_container_width=True)

            st.download_button(
                "⬇️ Download classification (CSV)",
                data=fs_table.to_csv(index=False).encode("utf-8-sig"),
                file_name="E_PROM_FlowSort.csv",
                mime="text/csv",
            )

    with tab_port:

        st.subheader("Portfolio selection — PROMETHEE V")
        st.markdown(
            "Choose the **best set** of alternatives rather than a single one: "
            "maximize the total value of the selected alternatives subject to "
            "resource budgets, the number of alternatives and mandatory / excluded "
            "alternatives. The problem is solved exactly (integer programming)."
        )

        objective_label = st.radio(
            "Objective",
            list(OBJECTIVE_MODES.values()),
            index=0,
            key="pf_objective",
            help=(
                "Net flows of a set of alternatives always sum to zero, so with the "
                "classic objective only alternatives with a positive net flow add "
                "value and the optimum is often a very small portfolio. The shifted "
                "objective gives every alternative a non-negative value, so the "
                "budget and the number limits decide how many enter (it favours "
                "larger portfolios)."
            ),
        )
        objective_mode = next(
            key for key, label_o in OBJECTIVE_MODES.items() if label_o == objective_label
        )

        mean_eval = np.mean(
            [np.asarray(v, dtype=float) for v in data["evaluations"].values()],
            axis=0,
        )
        continuous_names = [
            data["criteria"][k]
            for k in range(len(data["criteria"]))
            if run_params["types"][k] == CONTINUOUS
        ]

        res_names = st.multiselect(
            "Resource criteria (budget constraints)",
            continuous_names,
            default=[],
            key="pf_resources",
            help=(
                "Continuous criteria whose values (averaged over the decision "
                "makers) are consumed by each selected alternative, e.g. cost."
            ),
        )

        budgets = []
        for name in res_names:
            col = data["criteria"].index(name)
            total = float(np.abs(mean_eval[:, col]).sum())
            budgets.append(
                st.number_input(
                    f"Budget — {name} (total of all alternatives: {total:,.2f})",
                    min_value=0.0,
                    value=float(round(total / 2, 2)),
                    key=f"pf_budget_{col}",
                )
            )

        n_alt = data["n_requirements"]
        pf1, pf2 = st.columns(2)
        pf_min = int(pf1.number_input("Minimum number of alternatives", min_value=0, max_value=n_alt, value=0, step=1, key="pf_min"))
        pf_max = int(pf2.number_input("Maximum number of alternatives", min_value=1, max_value=n_alt, value=n_alt, step=1, key="pf_max"))

        must_labels = st.multiselect("Mandatory alternatives", labels, key="pf_must_in")
        excl_labels = st.multiselect("Excluded alternatives", labels, key="pf_must_out")

        if st.button("▶ Find the best portfolio", type="primary", key="run_pf"):
            try:
                resources = [
                    np.abs(mean_eval[:, data["criteria"].index(name)])
                    for name in res_names
                ]
                kwargs = dict(
                    resources=resources if resources else None,
                    budgets=budgets,
                    min_count=pf_min,
                    max_count=pf_max,
                    must_include=[labels.index(x) for x in must_labels],
                    must_exclude=[labels.index(x) for x in excl_labels],
                )
                phi_vec = np.asarray(result["phi_net"], dtype=float)
                value_vec = portfolio_values(phi_vec, objective_mode)
                st.session_state["portfolio"] = {
                    "solution": solve_portfolio(value_vec, **kwargs),
                    "greedy": greedy_by_rank(value_vec, **kwargs),
                    "resource_names": list(res_names),
                    "mode": objective_mode,
                    "values": value_vec,
                    "kwargs_min_count": pf_min,
                }
            except Exception as exc:
                st.error(f"Portfolio selection failed: {exc}")

        if "portfolio" in st.session_state:

            pf = st.session_state["portfolio"]
            sol, base = pf["solution"], pf["greedy"]

            if not sol["feasible"]:
                st.error("No portfolio satisfies all the constraints. Relax a budget or the number limits.")
            else:
                shifted = pf.get("mode") == "shifted"
                values_used = np.asarray(pf.get("values", result["phi_net"]), dtype=float)

                g1, g2, g3 = st.columns(3)
                g1.metric("Selected alternatives", sol["count"])
                g2.metric(
                    "Total value (optimal)" if shifted else "Total net Phi (optimal)",
                    f"{sol['objective']:.4f}",
                )
                g3.metric(
                    "Walking down the ranking",
                    f"{base['objective']:.4f}",
                    f"{sol['objective'] - base['objective']:+.4f} with the optimum",
                    delta_color="off",
                )

                for label_r, used, bud in zip(pf["resource_names"], sol["usage"], sol["budgets"]):
                    st.progress(
                        min(1.0, float(used / bud)) if bud > 0 else 0.0,
                        text=f"{label_r}: {used:,.2f} used of {bud:,.2f}",
                    )

                pf_table = {
                    "Alternative": labels,
                    "Net Phi": np.asarray(result["phi_net"], dtype=float),
                }
                if shifted:
                    pf_table["Value used (Phi − lowest Phi)"] = values_used
                pf_table.update({
                    "Position": np.asarray(result["rank_e_prom"], dtype=float),
                    "In optimal portfolio": ["✔" if s else "" for s in sol["selected"]],
                    "In ranking-order baseline": ["✔" if s else "" for s in base["selected"]],
                })
                pf_df = pd.DataFrame(pf_table).sort_values("Position")
                st.dataframe(pf_df, use_container_width=True, hide_index=True)

                n_positive = int(np.sum(values_used > 1e-12))
                usage_full = [
                    (used / bud) if bud > 0 else 0.0
                    for used, bud in zip(sol["usage"], sol["budgets"])
                ]
                budget_binding = any(u > 0.999 for u in usage_full)
                if (
                    not shifted
                    and sol["count"] <= max(n_positive, int(pf.get("kwargs_min_count", 0)))
                    and not budget_binding
                    and sol["count"] < len(labels)
                ):
                    st.info(
                        "With the classic objective, net flows sum to zero, so only "
                        "alternatives with a positive net flow add value: the optimum "
                        f"keeps {sol['count']} of {len(labels)} even though the budget is "
                        "not exhausted. Use the **shifted** objective to let the budget "
                        "and the number limits decide the size of the portfolio."
                    )
                st.caption(
                    "The ranking-order baseline walks down the ranking and keeps every "
                    "alternative that still fits. The optimum can beat it because a "
                    "lower-ranked alternative may fit the budget better. Alternatives with "
                    "negative net flow are selected only when needed to reach the minimum number."
                )
                st.download_button(
                    "⬇️ Download portfolio (CSV)",
                    data=pf_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="E_PROM_Portfolio.csv",
                    mime="text/csv",
                )

    with tab_report:

        st.subheader("PDF report")
        st.markdown(
            "Generate a report with the configuration, weights, ranking and every "
            "analysis you have run in this session (agreement, sensitivity, Monte Carlo, "
            "classification, portfolio)."
        )

        included = {
            "Ranking, configuration and weights": True,
            "Contribution of each criterion": True,
            "Weight sensitivity": True,
            "Agreement among decision makers": data["n_experts"] >= 2,
            "Robustness checks (run them in the Sensitivity tab)": bool(st.session_state.get("rob_done")),
            "Monte Carlo (run it in the Monte Carlo tab)": "mc" in st.session_state,
            "Classification (run it in the Classification tab)": "flowsort" in st.session_state,
            "Portfolio (run it in the Portfolio tab)": "portfolio" in st.session_state,
        }
        for section, present in included.items():
            st.write(("✅ " if present else "⬜ ") + section)

        with_json = st.checkbox(
            "Append the configuration file (reproducibility)", value=True, key="rep_json"
        )

        if st.button("Generate report", type="primary", key="run_report"):
            try:
                with st.spinner("Building the PDF..."):
                    if "sens" not in st.session_state:
                        st.session_state["sens"] = weight_sensitivity(terms, fahp["w_fuzzy_individual"])

                    extras = {
                        "contributions": criterion_contributions(terms),
                        "sensitivity": st.session_state["sens"],
                    }
                    if data["n_experts"] >= 2:
                        extras["agreement"] = expert_agreement(
                            result["phi_net_experts"], result["expert_names"]
                        )
                    if st.session_state.get("rob_done"):
                        extras["loo"] = st.session_state.get("loo")
                        extras["reversal"] = st.session_state.get("reversal")
                    if "mc" in st.session_state:
                        extras["mc"] = st.session_state["mc"]
                        extras["mc_params"] = st.session_state.get("mc_params", {})
                    if "flowsort" in st.session_state:
                        extras["flowsort"] = st.session_state["flowsort"]
                    if "portfolio" in st.session_state:
                        extras["portfolio"] = st.session_state["portfolio"]

                    cfg_bytes = dumps(
                        build_config(
                            data["criteria"],
                            run_params["types"],
                            run_params["dirs"],
                            run_params["pref_names"],
                            run_params["qs"],
                            run_params["ps"],
                            run_params["scales"],
                            mc_variation=st.session_state.get("mc_variation"),
                            expert_weights=EW,
                            expert_names=data["expert_names"],
                        )
                    ) if with_json else None

                    st.session_state["report_pdf"] = build_report(
                        data, result, config, fahp, extras, cfg_bytes,
                        version=APP_VERSION,
                    )
            except Exception as exc:
                st.error(f"Could not build the report: {exc}")

        if "report_pdf" in st.session_state:
            st.success("Report ready.")
            st.download_button(
                "⬇️ Download PDF report",
                data=st.session_state["report_pdf"],
                file_name="E_PROM_Report.pdf",
                mime="application/pdf",
            )

    with tab_guide:
        st.markdown(GUIDE_TEXT)