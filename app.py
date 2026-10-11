from pathlib import Path
import io
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.data_loader import (
    read_expert_workbook,
    read_fahp_workbook,
)
from core.e_prom_core import run_e_prom
from core.fuzzy_numbers import defuzz_coa
from core.labels import CONTINUOUS, ORDINAL
from core.monte_carlo import run_e_prom_monte_carlo
from core.promethee import promethee_i_relation


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
    "Express Fuzzy Preference Ranking with PROMETHEE — v7.4"
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

# Show the guide on the start screen (before any result exists). After a run,
# it is available in the "Guide" tab.
if "result" not in st.session_state:
    with st.expander("📘 User guide", expanded=True):
        st.markdown(GUIDE_TEXT)



# ================================================================
# SIDEBAR
# ================================================================

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

    run = False

    if req_file is not None:

        try:

            preview = read_expert_workbook(save_uploaded(req_file))

            criteria = preview["criteria"]
            ncrit = len(criteria)

            st.success(
                f"{preview['n_experts']} decision makers · "
                f"{preview['n_requirements']} alternatives · "
                f"{ncrit} criteria"
            )

            st.divider()
            st.header("2. Criterion configuration")

            st.info(
                "Ordinal: Usual or Level. Continuous: U-shape, V-shape, or Linear."
            )

            dirs = []
            types = []
            prefs = []
            qs = []
            ps = []
            scales = []
            rows = []

            for k, criterion in enumerate(
                criteria
            ):

                st.markdown(
                    f"### {criterion}"
                )

                typ_display = st.selectbox(
                    "Data type",
                    [ORDINAL, CONTINUOUS],
                    key=f"type_{k}",
                )
                typ = typ_display

                direction_name = st.selectbox(
                    "Direction",
                    ["Maximize", "Minimize"],
                    key=f"direction_{k}",
                )

                if typ == ORDINAL:

                    scale = st.selectbox(
                        "Ordinal scale",
                        [5, 7, 9],
                        index=1,
                        key=f"scale_{k}",
                    )

                    pref_name = st.selectbox(
                        "Preference function",
                        ["Usual", "Level"],
                        key=f"pref_{k}",
                    )

                    if pref_name == "Usual":
                        q = 0.0
                        p = 0.0
                    else:

                        q = st.number_input(
                            "q — indifference",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
                        )

                        p = st.number_input(
                            "p — preference",
                            min_value=float(q + 1e-9),
                            value=float(q + 1),
                            step=1.0,
                            key=f"p_{k}",
                        )

                else:

                    scale = 0
                    st.caption(
                        "Continuous values are represented by a trapezoidal fuzzy "
                        "number in the original unit. Its width is set by the "
                        "attribute precision given in the FAHP-Express workbook. "
                        "Express q and p in the original unit."
                    )

                    pref_name = st.selectbox(
                        "Preference function",
                        [
                            "U-shape",
                            "V-shape",
                            "Linear",
                        ],
                        key=f"pref_{k}",
                    )

                    if pref_name == "U-shape":

                        q = st.number_input(
                            "q — indifference threshold",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
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
                            value=1.0,
                            step=1.0,
                            key=f"p_{k}",
                            help=(
                                "Preference grows linearly up to p and is "
                                "complete from p on."
                            ),
                        )

                    else:  # Linear

                        q = st.number_input(
                            "q — indifference threshold",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
                        )

                        p = st.number_input(
                            "p — preference threshold",
                            min_value=float(q + 1e-9),
                            value=float(q + 1),
                            step=1.0,
                            key=f"p_{k}",
                        )

                code = PREF_CODES[pref_name]

                dirs.append(
                    1
                    if direction_name == "Maximize"
                    else -1
                )

                types.append(typ)
                prefs.append(code)
                qs.append(q)
                ps.append(p)
                scales.append(scale)

                rows.append({
                    "Criterion": criterion,
                    "Type": typ_display,
                    "Direction": direction_name,
                    "Preference": pref_name,
                    "q": q,
                    "p": p,
                    "Scale": (
                        scale
                        if typ == ORDINAL
                        else "Continuous"
                    ),
                })

            config = pd.DataFrame(rows)

            st.divider()
            st.header("3. Execution")

            run = st.button(
                "▶ Run E-PROM",
                type="primary",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Error reading the workbook: {exc}"
            )


# ================================================================
# EXECUTION
# ================================================================

if run:

    if fahp_file is None:
        st.error(
            "Upload the FAHP-Express workbook."
        )
        st.stop()

    try:

        with st.spinner(
            "Running FAHP-Express + Fuzzy PROMETHEE..."
        ):

            data = preview

            fahp = read_fahp_workbook(
                save_uploaded(fahp_file),
                data["criteria"],
                expert_names=data["expert_names"],
            )

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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏆 Ranking",
            "📈 PROMETHEE",
            "👥 Decision Makers",
            "🎲 Monte Carlo",
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

        st.subheader(
            "Configuration"
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

    with tab4:

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
                default = 20.0
                variation.append(
                    st.number_input(
                        f"Variation for {criterion} (%)",
                        min_value=0.0,
                        max_value=1000.0,
                        value=default,
                        step=5.0,
                        key=f"mc_var_{k}",
                        help="Applies ± percentage variation to each observed decision-maker value.",
                    )
                )
            else:
                variation.append(0.0)

        weight_mode_label = st.selectbox(
            "Weight sampling",
            [
                "Bootstrap decision makers (recommended)",
                "Independent per criterion (legacy)",
            ],
            key="mc_weight_mode",
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
                value=2000,
                step=100,
                key="mc_n",
            )
        with col2:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=42,
                step=1,
                key="mc_seed",
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
                    )
            except Exception as exc:
                st.error(f"Monte Carlo failed: {exc}")
                st.stop()

            st.session_state["mc"] = mc

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

    with tab5:
        st.markdown(GUIDE_TEXT)