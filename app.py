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
from core.monte_carlo import run_e_prom_monte_carlo


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



def plot_mc_rank_acceptability(data, mc):
    labels = [str(x) for x in data["ids"]]
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
        title="CPP/Monte Carlo — aceitabilidade das posições",
        xaxis_title="Posição",
        yaxis_title="Alternativa",
        height=max(550, len(labels) * 30),
    )
    return fig


def plot_mc_outworking(data, mc):
    labels = [str(x) for x in data["ids"]]
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
        title="CPP/Monte Carlo — aceitabilidade de sobreclassificação",
        xaxis_title="Alternativa sobreclassificada",
        yaxis_title="Alternativa que sobreclassifica",
        height=700,
    )
    return fig


def mc_summary_df(data, mc):
    return pd.DataFrame({
        "Alternativa": data["ids"],
        "Posição média": mc["mean_rank"],
        "DP posição": mc["std_rank"],
        "Freq. 1º lugar": 100 * mc["freq_top1"],
        "Phi médio": mc["mean_phi"],
        "DP Phi": mc["std_phi"],
    }).sort_values(
        "Posição média"
    ).reset_index(drop=True)

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
# MODELOS PARA DOWNLOAD
# ================================================================

TEMPLATES_DIR = ROOT / "templates"
EVAL_TEMPLATE = TEMPLATES_DIR / "E_PROM_Avaliacoes_Template.xlsx"
FAHP_TEMPLATE = TEMPLATES_DIR / "E_PROM_FAHP_Express_Template.xlsx"

st.markdown("### 📥 Baixe as planilhas-modelo")
st.caption(
    "Use estes arquivos como ponto de partida. Preencha as avaliações dos "
    "especialistas e, separadamente, as referências do FAHP-Express."
)

col_model_1, col_model_2 = st.columns(2)

with col_model_1:
    if EVAL_TEMPLATE.exists():
        with open(EVAL_TEMPLATE, "rb") as f:
            st.download_button(
                "⬇️ Planilha de Avaliações",
                data=f.read(),
                file_name=EVAL_TEMPLATE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_eval_template",
            )
    else:
        st.warning("Modelo de avaliações não encontrado no pacote.")

with col_model_2:
    if FAHP_TEMPLATE.exists():
        with open(FAHP_TEMPLATE, "rb") as f:
            st.download_button(
                "⬇️ Planilha FAHP-Express",
                data=f.read(),
                file_name=FAHP_TEMPLATE.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="download_fahp_template",
            )
    else:
        st.warning("Modelo FAHP-Express não encontrado no pacote.")

st.divider()

GUIDE_TEXT = """# E-PROM — Guide

## 1. Objetivo

O E-PROM (Express Fuzzy Preference Ranking with PROMETHEE) combina:

**FAHP-Express → pesos fuzzy dos critérios → Fuzzy PROMETHEE**

O método mantém os pesos fuzzy durante o Fuzzy PROMETHEE e realiza a
defuzzificação dos fluxos no final.

---

## 2. Arquivo de avaliações

A planilha de avaliações deve possuir:

- uma aba por especialista;
- uma última aba chamada `Guide`, que não é processada;
- coluna A: ID da alternativa;
- coluna B: descrição;
- coluna C em diante: critérios.

Todos os especialistas devem usar os mesmos IDs, descrições e nomes de critérios.

### Exemplo

| ID | Descrição | Impacto | Custo | Prazo |
|---|---|---:|---:|---:|
| R1 | Alternativa 1 | 7 | 300000 | 12 |
| R2 | Alternativa 2 | 5 | 250000 | 18 |

---

## 3. Tipo do critério

### Ordinal

Use quando a avaliação é uma escala ordenada, por exemplo:

- 1–5;
- 1–7;
- 1–9.

Funções permitidas:

- **Usual**
- **Level**

O valor ordinal é convertido para um número fuzzy trapezoidal.

### Contínuo

Use para grandezas como:

- custo em R$;
- prazo em meses;
- distância;
- tempo;
- consumo;
- desempenho físico.

Funções permitidas:

- **U-shape**
- **V-shape**
- **Linear**

O valor permanece na unidade original. Não é convertido para 1–7.

---

## 4. Limiar q e p

### U-shape

`q` representa o limiar de indiferença.

### V-shape

`p` representa o limiar de preferência.

### Linear

Usa os dois:

- `q`: limiar de indiferença;
- `p`: limiar de preferência.

Sempre use a unidade original do critério.

Exemplo:

Custo em reais:

`q = 10000`

`p = 50000`

Prazo em meses:

`q = 1`

`p = 4`

---

## 5. FAHP-Express

Na planilha FAHP-Express, cada especialista informa uma linha de referência
a partir da célula B4.

A linha deve possuir exatamente o mesmo número de critérios da planilha
de avaliações.

O E-PROM reconstrói as comparações par a par e calcula os pesos fuzzy pelo
procedimento de Buckley.

---

## 6. Pesos fuzzy no Fuzzy PROMETHEE

O E-PROM não defuzzifica os pesos antes do PROMETHEE.

Para cada critério:

`peso fuzzy × preferência fuzzy`

Os termos são agregados para obter os fluxos fuzzy:

- Φ+;
- Φ−;
- Φ líquido.

A defuzzificação pelo centro de área é aplicada aos fluxos finais para
obter o ranking.

---

## 7. Monte Carlo / CPP

O módulo de Monte Carlo avalia a estabilidade do resultado, preservando
a informação fuzzy durante a agregação dos especialistas.

### Critérios ordinais

A distribuição é **empírica**, construída diretamente a partir das
respostas dos especialistas.

Para cada alternativa e critério, uma resposta observada entre os
especialistas é sorteada de acordo com sua frequência.

Assim, se 10 especialistas responderam:

`7, 7, 6, 5, 7, 6, 7, 4, 7, 6`

a distribuição usada na simulação preserva essas frequências.

### Critérios contínuos

O usuário informa uma variação percentual.

Exemplo:

`20%`

O valor observado `x` é simulado em:

`[0,80x ; 1,20x]`

por amostragem uniforme.

A variação é configurada individualmente para cada critério contínuo.

### Pesos

Em cada iteração, os valores de referência do FAHP-Express também são
reamostrados empiricamente a partir dos especialistas e um novo conjunto
de pesos fuzzy é calculado.

---

## 8. Principais resultados do Monte Carlo

O E-PROM calcula:

- posição média;
- desvio-padrão da posição;
- Φ líquido médio;
- desvio-padrão de Φ líquido;
- frequência de 1º lugar;
- matriz de aceitabilidade de ranking;
- matriz de aceitabilidade de sobreclassificação PROMETHEE I;
- distribuição dos pesos dos critérios.

O módulo não possui classificação `KEY` nem grupos específicos do domínio
militar.

---

## 9. Interpretação

Em cada simulação, os fluxos dos especialistas são agregados no domínio
fuzzy e somente o fluxo fuzzy agregado é defuzzificado para obter o ranking.

A análise probabilística deve ser usada como análise de estabilidade,
não como substituição automática do ranking determinístico.

Resultados úteis incluem:

- frequência com que uma alternativa ocupa cada posição;
- frequência de 1º lugar;
- dispersão de Φ;
- sensibilidade dos pesos;
- probabilidade de sobreclassificação entre pares.

---

## 10. Fluxo recomendado

1. Preparar as avaliações dos especialistas.
2. Preparar o FAHP-Express.
3. Configurar tipo e direção dos critérios.
4. Selecionar a função de preferência.
5. Definir q e/ou p.
6. Executar o E-PROM determinístico.
7. Verificar o ranking e as relações PROMETHEE I.
8. Executar o Monte Carlo/CPP.
9. Avaliar estabilidade e aceitabilidade das posições.
10. Exportar os resultados.
"""


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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🏆 Ranking",
            "📈 PROMETHEE",
            "👥 Especialistas",
            "🎲 CPP / Monte Carlo",
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

        st.subheader("CPP / Monte Carlo — estabilidade do E-PROM")

        st.markdown(
            "Para **critérios ordinais**, a incerteza é obtida empiricamente "
            "das respostas dos especialistas. Para **critérios contínuos**, "
            "informe a variação percentual ± desejada."
        )
        st.info(
            "No E-PROM, os fluxos dos especialistas permanecem fuzzy durante "
            "a agregação. A defuzzificação ocorre somente após a agregação "
            "fuzzy, em cada simulação."
        )

        continuous_idx = [
            i for i, typ in enumerate(config["Tipo"])
            if typ == "Contínuo"
        ]

        variation = []
        for k, criterion in enumerate(data["criteria"]):
            if k in continuous_idx:
                default = 20.0
                variation.append(
                    st.number_input(
                        f"Variação de {criterion} (%)",
                        min_value=0.0,
                        max_value=1000.0,
                        value=default,
                        step=5.0,
                        key=f"mc_var_{k}",
                        help="Aplica ± percentual ao valor observado de cada especialista.",
                    )
                )
            else:
                variation.append(0.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            n_mc = st.number_input(
                "Número de simulações",
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
                "Critérios contínuos",
                len(continuous_idx),
            )

        if st.button(
            "▶ Executar CPP / Monte Carlo",
            type="primary",
            key="run_mc",
        ):

            # Usa as linhas de referência individuais fornecidas pelos
            # especialistas no FAHP-Express. Mantém compatibilidade com
            # versões anteriores do data_loader que não armazenavam
            # explicitamente `reference_rows`.
            if "reference_rows" in fahp:
                reference_rows = np.asarray(
                    fahp["reference_rows"],
                    dtype=float,
                )
            else:
                # Fallback: A_individual preserva as razões v_j/v_i
                # usadas para reconstruir o FAHP-Express. Como uma
                # constante comum não altera as razões, a primeira linha
                # da matriz pode ser usada como uma representação
                # equivalente dos valores de referência para o MC.
                A_individual = fahp.get("A_individual")
                if A_individual is None:
                    st.error(
                        "Não foi possível recuperar as respostas de referência "
                        "do FAHP-Express para o CPP/Monte Carlo. Reexecute o "
                        "E-PROM com a planilha FAHP-Express atualizada."
                    )
                    st.stop()

                reference_rows = np.asarray(
                    [np.asarray(A, dtype=float)[0, :] for A in A_individual],
                    dtype=float,
                )

            mc = run_e_prom_monte_carlo(
                data["evaluations"],
                reference_rows,
                types,
                dirs,
                prefs,
                qs,
                ps,
                scales,
                variation,
                n_mc=int(n_mc),
                seed=int(seed),
            )

            st.session_state["mc"] = mc

        if "mc" in st.session_state:

            mc = st.session_state["mc"]

            st.success(
                f"Monte Carlo concluído: {mc['n_mc']:,} simulações."
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

            st.subheader("Distribuição dos pesos")
            st.dataframe(
                w_df.describe().T[
                    ["mean", "std", "min", "max"]
                ],
                use_container_width=True,
            )

            st.download_button(
                "⬇️ Baixar resultados do Monte Carlo",
                data=pd.DataFrame({
                    "Alternativa": data["ids"],
                    "Posição média": mc["mean_rank"],
                    "DP posição": mc["std_rank"],
                    "Freq. 1º lugar (%)": 100 * mc["freq_top1"],
                    "Phi médio": mc["mean_phi"],
                    "DP Phi": mc["std_phi"],
                }).to_csv(index=False).encode("utf-8-sig"),
                file_name="E_PROM_CPP_MonteCarlo.csv",
                mime="text/csv",
            )

    with tab5:
        st.markdown(GUIDE_TEXT)
