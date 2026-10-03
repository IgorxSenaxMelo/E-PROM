"""Núcleo matemático do E-PROM."""

from __future__ import annotations
import numpy as np

from .promethee import (
    run_fuzzy_promethee_trap,
    rank_from_net_flow,
)


# Ordinal fuzzy evaluation scales.
TFN_EVAL_5 = np.array([
    [1.0, 1.0, 1.5, 2.0],
    [1.5, 2.0, 2.0, 2.5],
    [2.0, 2.5, 3.0, 3.5],
    [3.0, 3.5, 4.0, 4.5],
    [4.0, 4.5, 5.0, 5.0],
])

TFN_EVAL_7 = np.array([
    [1.0, 1.0, 1.5, 2.0],
    [1.0, 1.5, 2.5, 3.0],
    [2.0, 2.5, 3.5, 4.0],
    [3.0, 3.5, 4.5, 5.0],
    [4.0, 4.5, 5.5, 6.0],
    [5.0, 5.5, 6.5, 7.0],
    [6.0, 6.5, 7.0, 7.0],
])

TFN_EVAL_9 = np.array([
    [1.0, 1.0, 1.5, 2.0],
    [1.0, 1.5, 2.5, 3.0],
    [2.0, 2.5, 3.5, 4.0],
    [3.0, 3.5, 4.5, 5.0],
    [4.0, 4.5, 5.0, 5.5],
    [5.0, 5.5, 6.5, 7.0],
    [6.0, 6.5, 7.5, 8.0],
    [7.0, 7.5, 8.5, 9.0],
    [8.0, 8.5, 9.0, 9.0],
])


def fuzzify_ordinal(value, scale_max, precision="Média"):
    """Converte uma avaliação ordinal em trapezoide fuzzy.

    A tabela linguística padrão representa a avaliação com precisão média.
    O nível de precisão atua como um operador de contração/dilatação em
    torno do valor ordinal observado, mantendo os limites da escala.

    Alta  -> menor incerteza fuzzy
    Média -> representação linguística padrão
    Baixa -> maior incerteza fuzzy
    """
    value = float(value)
    scale_max = int(scale_max)
    if abs(value - round(value)) > 1e-9:
        raise ValueError(f"Valor ordinal {value} não é inteiro.")
    v = int(round(value))
    if v < 1 or v > scale_max:
        raise ValueError(f"Valor ordinal {v} fora da escala 1–{scale_max}.")
    if scale_max == 5:
        base = TFN_EVAL_5[v - 1].copy()
    elif scale_max == 7:
        base = TFN_EVAL_7[v - 1].copy()
    elif scale_max == 9:
        base = TFN_EVAL_9[v - 1].copy()
    else:
        raise ValueError("Escala ordinal suportada: 5, 7 ou 9.")
    factors = {"Alta": 0.5, "Média": 1.0, "Baixa": 1.5}
    if precision not in factors:
        raise ValueError(f"Precisão ordinal não reconhecida: {precision!r}. Use Alta, Média ou Baixa.")
    factor = factors[precision]
    fuzzy = v + factor * (base - v)
    fuzzy = np.clip(fuzzy, 1.0, float(scale_max))
    return np.sort(fuzzy.astype(float))


PRECISION_DEFAULTS = {
    # (suporte externo %, núcleo %, relativos ao valor observado)
    # A escala é deliberadamente monotônica: maior precisão -> menor
    # dispersão fuzzy. Os valores são parâmetros operacionais padrão e
    # podem ser sobrescritos no arquivo FAHP-Express.
    "Alta": (5.0, 2.0),
    "Média": (10.0, 4.0),
    "Baixa": (100.0 / 6.0, 20.0 / 3.0),
}


def continuous_to_fuzzy(value, precision="Alta", support_pct=None, core_pct=None):
    """Converte um valor contínuo em trapezoide fuzzy relativo.

    x -> [x(1-r_s), x(1-r_c), x(1+r_c), x(1+r_s)]

    r_s = semi-amplitude externa (suporte)
    r_c = semi-amplitude interna (núcleo)

    A parametrização usa a unidade original do atributo e representa
    incerteza epistemológica sobre o valor observado.
    """
    x = float(value)
    if not np.isfinite(x):
        raise ValueError("Valor contínuo inválido.")

    if support_pct is None or core_pct is None:
        if precision not in PRECISION_DEFAULTS:
            raise ValueError(
                f"Precisão não reconhecida: {precision!r}. Use Alta, Média ou Baixa."
            )
        support_pct, core_pct = PRECISION_DEFAULTS[precision]

    rs = abs(float(support_pct)) / 100.0
    rc = abs(float(core_pct)) / 100.0

    if rc > rs:
        raise ValueError("A amplitude do núcleo não pode ser maior que a do suporte.")

    scale = abs(x)
    if scale == 0:
        return np.zeros(4, dtype=float)

    a = x - scale * rs
    b = x - scale * rc
    c = x + scale * rc
    d = x + scale * rs
    return np.sort(np.array([a, b, c, d], dtype=float))


def build_fuzzy_matrix(
    data_numeric,
    types,
    scales,
    precision_levels=None,
    precision_support_pct=None,
    precision_core_pct=None,
):
    """
    Ordinal -> trapezoide linguístico.
    Contínuo -> trapezoide fuzzy parametrizado pela precisão.

    A forma geral para um valor contínuo x é:
        [x(1-r_s), x(1-r_c), x(1+r_c), x(1+r_s)]

    mantendo o atributo na unidade original.
    """
    data_numeric = np.asarray(
        data_numeric,
        dtype=float,
    )

    n_req, n_crit = data_numeric.shape

    if len(types) != n_crit:
        raise ValueError(
            "Número de tipos de dado diferente do número de critérios."
        )

    F = np.zeros(
        (n_req, n_crit, 4),
        dtype=float,
    )

    if precision_levels is None:
        precision_levels = ["Alta"] * n_crit
    if len(precision_levels) != n_crit:
        raise ValueError("Número de níveis de precisão incompatível.")

    if precision_support_pct is None:
        precision_support_pct = [None] * n_crit
    if precision_core_pct is None:
        precision_core_pct = [None] * n_crit

    for i in range(n_req):

        for k in range(n_crit):

            value = data_numeric[i, k]

            if types[k] == "Ordinal":
                F[i, k] = fuzzify_ordinal(
                    value,
                    scales[k],
                    precision=str(precision_levels[k]),
                )

            elif types[k] == "Contínuo":
                if not np.isfinite(value):
                    raise ValueError(
                        f"Valor contínuo inválido na alternativa {i + 1}, "
                        f"critério {k + 1}."
                    )

                F[i, k] = continuous_to_fuzzy(
                    value,
                    precision=str(precision_levels[k]),
                    support_pct=precision_support_pct[k],
                    core_pct=precision_core_pct[k],
                )

            else:
                raise ValueError(
                    f"Tipo de dado não reconhecido: {types[k]!r}"
                )

    return F


def _validate_preferences(
    types,
    pref_types,
    q_vals,
    p_vals,
):
    for k, typ in enumerate(types):

        code = int(pref_types[k])
        q = float(q_vals[k])
        p = float(p_vals[k])

        if typ == "Ordinal":

            if code not in (1, 4):
                raise ValueError(
                    "Critérios ordinais aceitam apenas "
                    "Usual ou Level."
                )

            if code == 4 and p <= q:
                raise ValueError(
                    f"Critério {k + 1}: Level requer p > q."
                )

        elif typ == "Contínuo":

            if code not in (2, 3, 5):
                raise ValueError(
                    "Critérios contínuos aceitam apenas "
                    "U-shape, V-shape ou Linear."
                )

            if code == 2:
                if q < 0:
                    raise ValueError(
                        f"Critério {k + 1}: U-shape requer q >= 0."
                    )

            elif code == 3:
                if p <= 0:
                    raise ValueError(
                        f"Critério {k + 1}: V-shape requer p > 0."
                    )

            elif code == 5:
                if q < 0 or p <= q:
                    raise ValueError(
                        f"Critério {k + 1}: Linear requer 0 <= q < p."
                    )

        else:
            raise ValueError(
                f"Tipo de dado não reconhecido: {typ!r}"
            )


def run_e_prom(
    evaluations,
    weights_fuzzy_individual,
    types,
    directions,
    preference_types,
    q_vals,
    p_vals,
    scales,
    precision_levels=None,
    precision_support_pct=None,
    precision_core_pct=None,
):
    """
    Executa o E-PROM individualmente para cada especialista e
    agrega os fluxos fuzzy a posteriori.

    Os pesos fuzzy do FAHP-Express são mantidos fuzzy e enviados
    diretamente ao Fuzzy PROMETHEE. A defuzzificação ocorre
    somente nos fluxos individuais e no fluxo agregado.
    """
    expert_names = list(evaluations.keys())

    if not expert_names:
        raise ValueError(
            "Nenhum especialista encontrado."
        )

    n_exp = len(expert_names)
    n_req, n_crit = np.asarray(
        evaluations[expert_names[0]]
    ).shape

    if len(types) != n_crit:
        raise ValueError(
            "A configuração possui número de critérios diferente "
            "da matriz de decisão."
        )

    if len(directions) != n_crit:
        raise ValueError("Número de direções incompatível.")

    if len(preference_types) != n_crit:
        raise ValueError("Número de funções de preferência incompatível.")

    if len(q_vals) != n_crit or len(p_vals) != n_crit:
        raise ValueError("Número de limiares q/p incompatível.")

    if len(scales) != n_crit:
        raise ValueError("Número de escalas incompatível.")

    if precision_levels is None:
        precision_levels = ["Alta"] * n_crit
    if len(precision_levels) != n_crit:
        raise ValueError("Número de níveis de precisão incompatível.")

    weights_fuzzy_individual = np.asarray(
        weights_fuzzy_individual,
        dtype=float,
    )

    if weights_fuzzy_individual.shape != (
        n_crit,
        4,
        n_exp,
    ):
        raise ValueError(
            "Dimensão dos pesos fuzzy incompatível com "
            "número de critérios/especialistas: "
            f"{weights_fuzzy_individual.shape}; esperado "
            f"({n_crit}, 4, {n_exp})."
        )

    _validate_preferences(
        types,
        preference_types,
        q_vals,
        p_vals,
    )

    phi_plus_exp = np.zeros(
        (n_req, n_exp)
    )
    phi_minus_exp = np.zeros(
        (n_req, n_exp)
    )
    phi_net_exp = np.zeros(
        (n_req, n_exp)
    )

    phi_plus_fuzzy_exp = np.zeros(
        (n_req, 4, n_exp)
    )
    phi_minus_fuzzy_exp = np.zeros(
        (n_req, 4, n_exp)
    )
    phi_net_fuzzy_exp = np.zeros(
        (n_req, 4, n_exp)
    )

    for e, expert in enumerate(expert_names):

        data_e = np.asarray(
            evaluations[expert],
            dtype=float,
        )

        F_e = build_fuzzy_matrix(
            data_e,
            types,
            scales,
            precision_levels=precision_levels,
            precision_support_pct=precision_support_pct,
            precision_core_pct=precision_core_pct,
        )

        w_e = weights_fuzzy_individual[
            :, :, e
        ]

        output = run_fuzzy_promethee_trap(
            F_e,
            w_e,
            directions,
            preference_types,
            q_vals,
            p_vals,
        )

        (
            pp,
            pm,
            pn,
            ppf,
            pmf,
            pnf,
        ) = output

        phi_plus_exp[:, e] = pp
        phi_minus_exp[:, e] = pm
        phi_net_exp[:, e] = pn

        phi_plus_fuzzy_exp[:, :, e] = ppf
        phi_minus_fuzzy_exp[:, :, e] = pmf
        phi_net_fuzzy_exp[:, :, e] = pnf

    # A posteriori aggregation of fuzzy flows.
    plus_fuzzy = np.mean(
        phi_plus_fuzzy_exp,
        axis=2,
    )
    minus_fuzzy = np.mean(
        phi_minus_fuzzy_exp,
        axis=2,
    )
    net_fuzzy = np.mean(
        phi_net_fuzzy_exp,
        axis=2,
    )

    plus = np.array([
        __import__(
            "core.fuzzy_numbers",
            fromlist=["defuzz_coa"]
        ).defuzz_coa(x)
        for x in plus_fuzzy
    ])

    minus = np.array([
        __import__(
            "core.fuzzy_numbers",
            fromlist=["defuzz_coa"]
        ).defuzz_coa(x)
        for x in minus_fuzzy
    ])

    net = np.array([
        __import__(
            "core.fuzzy_numbers",
            fromlist=["defuzz_coa"]
        ).defuzz_coa(x)
        for x in net_fuzzy
    ])

    rank_experts = np.column_stack([
        rank_from_net_flow(
            phi_net_exp[:, e]
        )
        for e in range(n_exp)
    ])

    rank_e_prom = rank_from_net_flow(
        net
    )

    return {
        "expert_names": expert_names,
        "phi_plus_experts": phi_plus_exp,
        "phi_minus_experts": phi_minus_exp,
        "phi_net_experts": phi_net_exp,
        "phi_plus_fuzzy_experts": phi_plus_fuzzy_exp,
        "phi_minus_fuzzy_experts": phi_minus_fuzzy_exp,
        "phi_net_fuzzy_experts": phi_net_fuzzy_exp,
        "phi_plus_fuzzy": plus_fuzzy,
        "phi_minus_fuzzy": minus_fuzzy,
        "phi_net_fuzzy": net_fuzzy,
        "phi_plus": plus,
        "phi_minus": minus,
        "phi_net": net,
        "rank_experts": rank_experts,
        "rank_e_prom": rank_e_prom,
        "q_vals": np.asarray(q_vals, dtype=float),
        "p_vals": np.asarray(p_vals, dtype=float),
        "types": list(types),
        "directions": np.asarray(directions, dtype=int),
        "preference_types": np.asarray(preference_types, dtype=int),
        "scales": np.asarray(scales, dtype=int),
        "precision_levels": list(precision_levels),
        "precision_support_pct": precision_support_pct,
        "precision_core_pct": precision_core_pct,
    }
