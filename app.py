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

from core.data_loader import read_expert_workbook, read_fahp_workbook, CRITERIA
from core.e_prom_core import run_e_prom
from core.fuzzy_numbers import defuzz_coa
from core.cpp import cpp_normal


st.set_page_config(
    page_title="E-PROM — Fuzzy Preference Ranking",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def save_uploaded(uploaded_file):
    """Save a Streamlit UploadedFile to a temporary file."""
    suffix = Path(uploaded_file.name).suffix or ".xlsx"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.close()
    return Path(tmp.name)


def get_expert_names(data):
    """Accept both the current loader key ('sheets') and the newer key."""
    return list(data.get("expert_names", data.get("sheets", [])))


def get_criteria(data):
    """Return criteria from the loader when available, otherwise the 5 E-PROM criteria."""
    return list(data.get("criteria", CRITERIA))


def normalize_result(result):
    """
    Accept result dictionaries produced by different E-PROM revisions.
    The current expected names are rank_e_prom and *_experts.
    """
    r = dict(result)

    if "rank_e_prom" not in r and "rank_siprem" in r:
        r["rank_e_prom"] = r["rank_siprem"]

    if "phi_net_experts" not in r and "phi_net_expert" in r:
        r["phi_net_experts"] = r["phi_net_expert"]

    if "phi_plus_experts" not in r and "phi_plus_expert" in r:
        r["phi_plus_experts"] = r["phi_plus_expert"]

    if "phi_minus_experts" not in r and "phi_minus_expert" in r:
        r["phi_minus_experts"] = r["phi_minus_expert"]

    if "expert_names" not in r:
        r["expert_names"] = []

    return r


def result_df(data, result):
    """Build the main E-PROM ranking table."""
    result = normalize_result(result)

    rank = np.asarray(result["rank_e_prom"], dtype=int)
    phi_plus = np.asarray(result["phi_plus"], dtype=float)
    phi_minus = np.asarray(result["phi_minus"], dtype=float)
    phi_net = np.asarray(result["phi_net"], dtype=float)

    df = pd.DataFrame(
        {
            "Rank": rank,
            "Alternativa": data["ids"],
            "Descricao": data["descriptions"],
            "Phi+": phi_plus,
            "Phi-": phi_minus,
            "Phi líquido": phi_net,
        }
    )

    return (
        df.sort_values(["Rank", "Alternativa"])
        .reset_index(drop=True)
    )


def excel_bytes(data, result, config):
    """
    Export results to Excel.

    Important:
    sheet_name is passed explicitly to avoid the pandas/openpyxl
    positional-argument error seen in the Streamlit deployment.
    """
    result = normalize_result(result)
    out = io.BytesIO()
    ranking = result_df(data, result)

    expert_names = get_expert_names(data)
    if not expert_names:
        expert_names = result.get("expert_names", [])

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        ranking.to_excel(
            writer,
            sheet_name="Ranking",
            index=False,
        )

        phi_experts = np.asarray(
            result["phi_net_experts"],
            dtype=float,
        )

        phi_df = pd.DataFrame(
            phi_experts,
            index=data["ids"],
            columns=expert_names,
        )
        phi_df.index.name = "Alternativa"

        phi_df.to_excel(
            writer,
            sheet_name="Phi_por_Especialista",
        )

        pd.DataFrame(config).to_excel(
            writer,
            sheet_name="Configuracao",
            index=False,
        )

        # Also export fuzzy/group weights when available.
        if "w_fuzzy_group" in st.session_state.get("fahp", {}):
            wf = np.asarray(
                st.session_state["fahp"]["w_fuzzy_group"],
                dtype=float,
            )
            criteria = get_criteria(data)

            weights_df = pd.DataFrame(
                wf,
                index=criteria,
                columns=["a", "b", "c", "d"],
            )
            weights_df.index.name = "Criterio"

            weights_df.to_excel(
                writer,
                sheet_name="Pesos_Fuzzy",
            )

    out.seek(0)
    return out.getvalue()


def aggregate_weights(fahp):
    """Defuzzify individual fuzzy criterion weights for visualization."""
    wf = np.asarray(
        fahp["w_fuzzy_individual"],
        dtype=float,
    )

    crisp = np.array(
        [
            [
                defuzz_coa(wf[k, :, e])
                for e in range(wf.shape[2])
            ]
            for k in range(wf.shape[0])
        ]
    )

    return crisp.mean(axis=1), crisp


def relation_matrix(result, ids):
    """PROMETHEE I relation matrix: P, P-, I and R."""
    result = normalize_result(result)

    plus = np.asarray(result["phi_plus"], dtype=float)
    minus = np.asarray(result["phi_minus"], dtype=float)

    n = len(ids)
    rel = np.empty((n, n), dtype=object)
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


def plot_weights(fahp, criteria):
    """Plot FAHP-Express criterion weights."""
    mean_w, crisp = aggregate_weights(fahp)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=criteria,
            y=mean_w,
            name="Peso médio",
        )
    )

    names = fahp.get(
        "sheets",
        [f"Especialista {i + 1}" for i in range(crisp.shape[1])],
    )

    for e, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=criteria,
                y=crisp[:, e],
                mode="markers",
                name=name,
            )
        )

    fig.update_layout(
        title="Pesos dos critérios — FAHP-Express",
        xaxis_title="Critério",
        yaxis_title="Peso",
        height=500,
    )

    return fig


def plot_rank_heatmap(data, result):
    """Compare individual expert ranks with the E-PROM group rank."""
    result = normalize_result(result)

    labels = [str(x) for x in data["ids"]]
    expert_names = result.get(
        "expert_names",
        get_expert_names(data),
    )

    ranks = np.asarray(
        result["rank_experts"],
        dtype=float,
    )

    df = pd.DataFrame(
        ranks,
        index=labels,
        columns=expert_names,
    )

    df["E-PROM"] = np.asarray(
        result["rank_e_prom"],
        dtype=float,
    )

    df = df.sort_values("E-PROM")

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
            colorbar=dict(title="Posição"),
        )
    )

    fig.update_layout(
        title="Posições: especialistas vs E-PROM",
        height=max(600, len(labels) * 25),
    )

    return fig


def plot_relation(data, result):
    """Plot PROMETHEE I relations."""
    rel = relation_matrix(result, data["ids"])

    mapping = {
        "P-": 0,
        "I": 1,
        "P": 2,
        "R": 3,
    }

    z = np.vectorize(mapping.get)(rel)
    labels = [str(x) for x in data["ids"]]

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
        title="PROMETHEE I — relações de preferência",
        height=800,
    )

    return fig


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("E-PROM")
st.subheader(
    "Express Fuzzy Preference Ranking with PROMETHEE"
)
st.caption(
    "FAHP-Express + Fuzzy PROMETHEE — versão generalizada, "
    "sem classificação KEY."
)

with st.sidebar:

    st.header("1. Dados")

    req_file = st.file_uploader(
        "Planilha das avaliações",
        type=["xlsx"],
        help=(
            "Planilha com as abas dos especialistas. "
            "A última aba deve ser a aba de apoio/instruções."
        ),
    )

    fahp_file = st.file_uploader(
        "Planilha FAHP-Express",
        type=["xlsx"],
        help=(
            "Planilha com os julgamentos/referências do FAHP-Express."
        ),
    )

    if req_file is not None:

        tmp_req = save_uploaded(req_file)

        try:
            preview = read_expert_workbook(tmp_req)

            criteria = get_criteria(preview)
            ncrit = len(criteria)

            st.success(
                f"{ncrit} critérios detectados."
            )

            st.divider()
            st.header("2. Configuração dos critérios")

            st.info(
                "Para cada critério, escolha o tipo de dado, "
                "a direção, a função de preferência e, quando "
                "aplicável, os limiares q e p."
            )

            dirs = []
            types = []
            prefs = []
            qs = []
            ps = []
            scales = []
            rows = []

            for k, criterion in enumerate(criteria):

                st.markdown(
                    f"### {criterion}"
                )

                typ = st.selectbox(
                    "Tipo de dado",
                    ["Ordinal", "Contínuo"],
                    key=f"typ_{k}",
                    help=(
                        "Ordinal: escalas discretas, como Likert. "
                        "Contínuo: custo, tempo, medida física etc."
                    ),
                )

                direction = st.selectbox(
                    "Direção",
                    ["Maximizar", "Minimizar"],
                    key=f"dir_{k}",
                )

                if typ == "Ordinal":

                    scale = st.selectbox(
                        "Escala",
                        ["5", "7", "9"],
                        index=1,
                        key=f"scale_{k}",
                        help=(
                            "Escolha a escala utilizada para "
                            "representar a avaliação ordinal como "
                            "número fuzzy trapezoidal."
                        ),
                    )

                    pref_name = st.selectbox(
                        "Função de preferência",
                        ["Usual", "Level"],
                        key=f"pref_{k}",
                    )

                    if pref_name == "Level":

                        q = st.number_input(
                            "q — indiferença",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"q_{k}",
                            help=(
                                "Diferença até q: região de indiferença."
                            ),
                        )

                        p = st.number_input(
                            "p — preferência",
                            min_value=float(q + 1e-9),
                            value=max(float(q + 1), 1.0),
                            step=1.0,
                            key=f"p_{k}",
                            help=(
                                "A partir de p: preferência completa."
                            ),
                        )

                    else:
                        q = 0.0
                        p = 0.0

                else:

                    scale = 1

                    pref_name = st.selectbox(
                        "Função de preferência",
                        ["Level", "Linear"],
                        key=f"pref_{k}",
                        help=(
                            "Para dados contínuos, o E-PROM "
                            "utiliza funções com limiares q e p."
                        ),
                    )

                    q = st.number_input(
                        "q — limiar de indiferença",
                        min_value=0.0,
                        value=0.0,
                        step=0.1,
                        key=f"q_{k}",
                        help=(
                            "Informe q na unidade original do critério."
                        ),
                    )

                    p = st.number_input(
                        "p — limiar de preferência",
                        min_value=float(q + 1e-9),
                        value=max(float(q + 1), 1.0),
                        step=0.1,
                        key=f"p_{k}",
                        help=(
                            "Informe p na unidade original do critério. "
                            "Deve ser maior que q."
                        ),
                    )

                pref_code = {
                    "Usual": 1,
                    "Level": 4,
                    "Linear": 5,
                }[pref_name]

                dirs.append(
                    1 if direction == "Maximizar" else -1
                )
                types.append(typ)
                prefs.append(pref_code)
                qs.append(q)
                ps.append(p)
                scales.append(int(scale))

                rows.append(
                    {
                        "Criterio": criterion,
                        "Tipo": typ,
                        "Direção": direction,
                        "Função": pref_name,
                        "q": q,
                        "p": p,
                        "Escala": scale,
                    }
                )

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
                f"Erro ao ler a planilha de avaliações: {exc}"
            )
            run = False

    else:
        run = False


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

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
                save_uploaded(fahp_file)
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

            result = normalize_result(result)

            ranking = result_df(
                data,
                result,
            )

            # Store FAHP before generating Excel because the export
            # optionally includes the group fuzzy weights.
            st.session_state["fahp"] = fahp

            xlsx = excel_bytes(
                data,
                result,
                config,
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


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

if "result" in st.session_state:

    data = st.session_state["data"]
    fahp = st.session_state["fahp"]
    result = normalize_result(
        st.session_state["result"]
    )
    config = st.session_state["config"]
    ranking = st.session_state["ranking"]

    st.divider()

    a, b, c = st.columns(3)

    a.metric(
        "Especialistas",
        data["n_experts"],
    )

    a2 = data.get(
        "n_requirements",
        len(data["ids"]),
    )

    b.metric(
        "Alternativas",
        a2,
    )

    c.metric(
        "Critérios",
        len(get_criteria(data)),
    )

    t1, t2, t3, t4 = st.tabs(
        [
            "🏆 Ranking",
            "📈 PROMETHEE",
            "👥 Especialistas",
            "🎲 CPP",
        ]
    )

    # ---------------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------------

    with t1:

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

    # ---------------------------------------------------------------
    # PROMETHEE
    # ---------------------------------------------------------------

    with t2:

        st.plotly_chart(
            plot_weights(
                fahp,
                get_criteria(data),
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

        if "rank_experts" in result:
            st.plotly_chart(
                plot_rank_heatmap(
                    data,
                    result,
                ),
                use_container_width=True,
            )

    # ---------------------------------------------------------------
    # Experts
    # ---------------------------------------------------------------

    with t3:

        expert_names = result.get(
            "expert_names",
            get_expert_names(data),
        )

        if "phi_net_experts" in result:

            phi_df = pd.DataFrame(
                result["phi_net_experts"],
                index=data["ids"],
                columns=expert_names,
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
            "Configuração dos critérios"
        )

        st.dataframe(
            config,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------
    # CPP
    # ---------------------------------------------------------------

    with t4:

        st.info(
            "CPP — Composition of Probabilistic Preferences. "
            "Este módulo é aplicado aos critérios configurados "
            "como Contínuos."
        )

        cont_idx = [
            i
            for i, value in enumerate(config["Tipo"])
            if value == "Contínuo"
        ]

        if not cont_idx:

            st.warning(
                "Nenhum critério contínuo foi configurado."
            )

        else:

            cont_criteria = [
                get_criteria(data)[i]
                for i in cont_idx
            ]

            std_df = pd.DataFrame(
                0.0,
                index=data["ids"],
                columns=cont_criteria,
            )

            edited = st.data_editor(
                std_df,
                use_container_width=True,
                key="cpp_std",
                help=(
                    "Informe o desvio-padrão de cada alternativa "
                    "para cada critério contínuo."
                ),
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

                expert_names = get_expert_names(data)

                if not expert_names:
                    expert_names = result.get(
                        "expert_names",
                        [],
                    )

                # Mean decision matrix across experts.
                stacked = np.stack(
                    [
                        np.asarray(
                            data["evaluations"][expert],
                            dtype=float,
                        )
                        for expert in expert_names
                    ],
                    axis=0,
                )

                vals = np.mean(
                    stacked,
                    axis=0,
                )

                X = vals[:, cont_idx]
                S = edited.to_numpy(dtype=float)

                # Group fuzzy weights for CPP.
                W = np.array(
                    [
                        defuzz_coa(
                            fahp["w_fuzzy_group"][i]
                        )
                        for i in cont_idx
                    ],
                    dtype=float,
                )

                W_sum = W.sum()
                if W_sum > 0:
                    W = W / W_sum

                cpp = cpp_normal(
                    X,
                    S,
                    np.array(dirs)[cont_idx],
                    int(nsim),
                    int(seed),
                    weights=W,
                )

                cpp_df = pd.DataFrame(
                    {
                        "Alternativa": data["ids"],
                        "Score_CPP": cpp["score"],
                        "Rank_CPP": cpp["rank"],
                    }
                ).sort_values(
                    "Rank_CPP"
                )

                st.subheader(
                    "Ranking probabilístico — CPP"
                )

                st.dataframe(
                    cpp_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Probabilidade de ser a melhor"
                )

                prob_df = pd.DataFrame(
                    cpp["prob_best"],
                    index=data["ids"],
                    columns=cont_criteria,
                )

                st.dataframe(
                    prob_df,
                    use_container_width=True,
                )
