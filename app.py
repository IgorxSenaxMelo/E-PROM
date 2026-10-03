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
from core.cpp import cpp_normal


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
        "Alternativa": data["ids"],
        "Descricao": data["descriptions"],
        "Phi+": result["phi_plus"],
        "Phi-": result["phi_minus"],
        "Phi líquido": result["phi_net"],
    }).sort_values(
        ["Rank", "Alternativa"]
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
        phi.index.name = "Alternativa"

        phi.to_excel(
            writer,
            sheet_name="Phi_por_Especialista",
        )

        config.to_excel(
            writer,
            sheet_name="Configuracao",
            index=False,
        )

        fuzzy_w = pd.DataFrame(
            fahp["w_fuzzy_group"],
            index=data["criteria"],
            columns=["a", "b", "c", "d"],
        )
        fuzzy_w.index.name = "Criterio"

        fuzzy_w.to_excel(
            writer,
            sheet_name="Pesos_Fuzzy",
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
            name="Peso médio",
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
        title="Pesos fuzzy — FAHP-Express",
        xaxis_title="Critério",
        yaxis_title="Peso",
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
    rel = np.empty(
        (n, n),
        dtype=object,
    )

    eps = 1e-10

    for i in range(n):
        for j in range(n):

            if i == j:
                rel[i, j] = "I"
                continue

            a = (
                plus[i] > plus[j] + eps
                and minus[i] < minus[j] - eps
            )

            b = (
                plus[i] < plus[j] - eps
                and minus[i] > minus[j] + eps
            )

            eq = (
                abs(plus[i] - plus[j]) <= eps
                and abs(minus[i] - minus[j]) <= eps
            )

            if a:
                rel[i, j] = "P"
            elif b:
                rel[i, j] = "P-"
            elif eq:
                rel[i, j] = "I"
            else:
                rel[i, j] = "R"

    return rel


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

    labels = [
        str(x)
        for x in data["ids"]
    ]

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
                title="Relação",
            ),
        )
    )

    fig.update_layout(
        title="PROMETHEE I — relações",
        height=750,
    )

    return fig


def plot_rank_heatmap(data, result):
    labels = [
        str(x)
        for x in data["ids"]
    ]

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
                title="Posição"
            ),
        )
    )

    fig.update_layout(
        title="Posições por especialista e E-PROM",
        height=max(600, len(labels) * 30),
    )

    return fig


# ================================================================
# HEADER
# ================================================================

st.title("E-PROM")
st.subheader(
    "Express Fuzzy Preference Ranking with PROMETHEE"
)
st.caption(
    "FAHP-Express + Fuzzy PROMETHEE — método generalizado."
)


# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:

    st.header("1. Entradas")

    req_file = st.file_uploader(
        "Planilha das avaliações",
        type=["xlsx"],
    )

    fahp_file = st.file_uploader(
        "Planilha FAHP-Express",
        type=["xlsx"],
    )

    run = False

    if req_file is not None:

        try:

            preview = read_expert_workbook(
                save_uploaded(req_file)
            )

            criteria = preview["criteria"]
            ncrit = len(criteria)

            st.success(
                f"{preview['n_experts']} especialistas · "
                f"{preview['n_requirements']} alternativas · "
                f"{ncrit} critérios"
            )

            st.divider()
            st.header("2. Configuração dos critérios")

            st.info(
                "Ordinal: Usual ou Level. "
                "Contínuo: U-shape, V-shape ou Linear."
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

                typ = st.selectbox(
                    "Tipo de dado",
                    ["Ordinal", "Contínuo"],
                    key=f"type_{k}",
                )

                direction_name = st.selectbox(
                    "Direção",
                    ["Maximizar", "Minimizar"],
                    key=f"direction_{k}",
                )

                if typ == "Ordinal":

                    scale = st.selectbox(
                        "Escala ordinal",
                        [5, 7, 9],
                        index=1,
                        key=f"scale_{k}",
                    )

                    pref_name = st.selectbox(
                        "Função de preferência",
                        ["Usual", "Level"],
                        key=f"pref_{k}",
                    )

                    if pref_name == "Usual":
                        q = 0.0
                        p = 0.0
                    else:

                        q = st.number_input(
                            "q — indiferença",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
                        )

                        p = st.number_input(
                            "p — preferência",
                            min_value=float(q + 1e-9),
                            value=float(q + 1),
                            step=1.0,
                            key=f"p_{k}",
                        )

                else:

                    scale = 0

                    pref_name = st.selectbox(
                        "Função de preferência",
                        [
                            "U-shape",
                            "V-shape",
                            "Linear",
                        ],
                        key=f"pref_{k}",
                    )

                    if pref_name == "U-shape":

                        q = st.number_input(
                            "q — limiar de indiferença",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
                            help=(
                                "Diferença até q: indiferença. "
                                "Use a unidade original do critério."
                            ),
                        )

                        p = 0.0

                    elif pref_name == "V-shape":

                        q = 0.0

                        p = st.number_input(
                            "p — limiar de preferência",
                            min_value=1e-9,
                            value=1.0,
                            step=1.0,
                            key=f"p_{k}",
                            help=(
                                "Até p, a preferência cresce linearmente; "
                                "a partir de p, é completa."
                            ),
                        )

                    else:  # Linear

                        q = st.number_input(
                            "q — limiar de indiferença",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
                        )

                        p = st.number_input(
                            "p — limiar de preferência",
                            min_value=float(q + 1e-9),
                            value=float(q + 1),
                            step=1.0,
                            key=f"p_{k}",
                        )

                code = PREF_CODES[pref_name]

                dirs.append(
                    1
                    if direction_name == "Maximizar"
                    else -1
                )

                types.append(typ)
                prefs.append(code)
                qs.append(q)
                ps.append(p)
                scales.append(scale)

                rows.append({
                    "Criterio": criterion,
                    "Tipo": typ,
                    "Direção": direction_name,
                    "Função": pref_name,
                    "q": q,
                    "p": p,
                    "Escala": (
                        scale
                        if typ == "Ordinal"
                        else "Contínua"
                    ),
                })

            config = pd.DataFrame(rows)

            st.divider()
            st.header("3. Execução")

            run = st.button(
                "▶ Executar E-PROM",
                type="primary",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Erro ao ler a planilha: {exc}"
            )


# ================================================================
# EXECUTION
# ================================================================

if run:

    if fahp_file is None:
        st.error(
            "Envie a planilha FAHP-Express."
        )
        st.stop()

    try:

        with st.spinner(
            "Executando FAHP-Express + Fuzzy PROMETHEE..."
        ):

            data = preview

            fahp = read_fahp_workbook(
                save_uploaded(fahp_file),
                n_criteria=len(
                    data["criteria"]
                ),
            )

            result = run_e_prom(
                data["evaluations"],
                fahp["w_fuzzy_individual"],
                types,
                dirs,
                prefs,
                qs,
                ps,
                scales,
            )

            ranking = result_df(
                data,
                result,
            )

            xlsx = excel_bytes(
                data,
                result,
                config,
                fahp,
            )

        st.session_state.update(
            data=data,
            fahp=fahp,
            result=result,
            config=config,
            ranking=ranking,
            xlsx=xlsx,
        )

        st.success(
            "E-PROM executado com sucesso."
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
    result = st.session_state["result"]
    config = st.session_state["config"]
    ranking = st.session_state["ranking"]

    st.divider()

    a, b, c = st.columns(3)

    a.metric(
        "Especialistas",
        data["n_experts"],
    )

    b.metric(
        "Alternativas",
        data["n_requirements"],
    )

    c.metric(
        "Critérios",
        data["n_criteria"],
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏆 Ranking",
            "📈 PROMETHEE",
            "👥 Especialistas",
            "🎲 CPP",
        ]
    )

    with tab1:

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Baixar resultado em Excel",
            data=st.session_state["xlsx"],
            file_name="Resultado_E_PROM.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    with tab2:

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

        phi_df.index.name = "Alternativa"

        st.subheader(
            "Phi líquido por especialista"
        )

        st.dataframe(
            phi_df,
            use_container_width=True,
        )

        st.subheader(
            "Configuração"
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

    with tab4:

        st.info(
            "CPP é um módulo separado para critérios contínuos. "
            "Informe os desvios-padrão por alternativa e critério."
        )

        continuous_idx = [
            i
            for i, typ in enumerate(
                config["Tipo"]
            )
            if typ == "Contínuo"
        ]

        if not continuous_idx:

            st.warning(
                "Nenhum critério contínuo foi configurado."
            )

        else:

            continuous_criteria = [
                data["criteria"][i]
                for i in continuous_idx
            ]

            std_df = pd.DataFrame(
                0.0,
                index=data["ids"],
                columns=continuous_criteria,
            )

            std_input = st.data_editor(
                std_df,
                use_container_width=True,
                key="cpp_std",
            )

            nsim = st.number_input(
                "Simulações Monte Carlo",
                min_value=1000,
                max_value=500000,
                value=10000,
                step=1000,
            )

            seed = st.number_input(
                "Seed",
                min_value=0,
                value=0,
                step=1,
            )

            if st.button(
                "▶ Executar CPP",
                type="primary",
            ):

                expert_values = np.stack(
                    [
                        data["evaluations"][name]
                        for name in data["expert_names"]
                    ],
                    axis=0,
                )

                X_all = np.mean(
                    expert_values,
                    axis=0,
                )

                X = X_all[
                    :,
                    continuous_idx,
                ]

                S = std_input.to_numpy(
                    dtype=float
                )

                W = np.array([
                    defuzz_coa(
                        fahp["w_fuzzy_group"][i]
                    )
                    for i in continuous_idx
                ])

                cpp = cpp_normal(
                    X,
                    S,
                    np.asarray(
                        dirs
                    )[continuous_idx],
                    nsim=int(nsim),
                    seed=int(seed),
                    weights=W,
                )

                cpp_df = pd.DataFrame({
                    "Alternativa": data["ids"],
                    "Prob. melhor": cpp["score"],
                    "Rank CPP": cpp["rank"],
                }).sort_values(
                    "Rank CPP"
                )

                st.subheader(
                    "Resultado CPP"
                )

                st.dataframe(
                    cpp_df,
                    use_container_width=True,
                    hide_index=True,
                )
