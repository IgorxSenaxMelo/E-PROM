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


def fuzzify_ordinal(value, scale_max):
    value = float(value)
    scale_max = int(scale_max)

    if abs(value - round(value)) > 1e-9:
        raise ValueError(
            f"Valor ordinal {value} não é inteiro."
        )

    v = int(round(value))

    if v < 1 or v > scale_max:
        raise ValueError(
            f"Valor ordinal {v} fora da escala 1–{scale_max}."
        )

    if scale_max == 5:
        return TFN_EVAL_5[v - 1].copy()

    if scale_max == 7:
        return TFN_EVAL_7[v - 1].copy()

    if scale_max == 9:
        return TFN_EVAL_9[v - 1].copy()

    raise ValueError(
        "Escala ordinal suportada: 5, 7 ou 9."
    )


def build_fuzzy_matrix(
    data_numeric,
    types,
    scales,
):
    """
    Ordinal -> trapezoide linguístico.
    Contínuo -> número fuzzy degenerado [x,x,x,x].

    Assim, valores contínuos permanecem na unidade original.
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

    for i in range(n_req):

        for k in range(n_crit):

            value = data_numeric[i, k]

            if types[k] == "Ordinal":
                F[i, k] = fuzzify_ordinal(
                    value,
                    scales[k],
                )

            elif types[k] == "Contínuo":
                if not np.isfinite(value):
                    raise ValueError(
                        f"Valor contínuo inválido na alternativa {i + 1}, "
                        f"critério {k + 1}."
                    )

                F[i, k] = np.array(
                    [value, value, value, value],
                    dtype=float,
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
    }
