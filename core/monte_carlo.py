"""Monte Carlo do E-PROM.

Ordinal:
    amostragem empírica das respostas dos especialistas, como no MC-SIPREM.

Contínuo:
    perturbação uniforme do valor observado em cada especialista:
    x_MC ~ U[x(1-v), x(1+v)], onde v é informado em porcentagem.

Os pesos são reamostrados empiricamente a partir das respostas FAHP-Express
dos especialistas em cada iteração, mantendo a lógica do MC-SIPREM.
"""

from __future__ import annotations
import numpy as np

from .e_prom_core import build_fuzzy_matrix
from .fuzzy_numbers import defuzz_coa
from .fahp import crisp_to_fuzzy_trap_interp, calc_fahp_buckley
from .promethee import run_fuzzy_promethee_trap, rank_from_net_flow


def _empirical_sample(values, rng):
    values = np.asarray(values)
    unique, counts = np.unique(values, return_counts=True)
    prob = counts / counts.sum()
    return unique[rng.choice(len(unique), p=prob)]


def _sample_fahp_weights(reference_rows, rng):
    rows = np.asarray(reference_rows, dtype=float)
    n_exp, n_crit = rows.shape

    sampled = np.array([
        _empirical_sample(rows[:, k], rng)
        for k in range(n_crit)
    ])

    A = sampled[None, :] / sampled[:, None]

    M = np.zeros((n_crit, n_crit, 4))
    for i in range(n_crit):
        for j in range(n_crit):
            M[i, j] = crisp_to_fuzzy_trap_interp(A[i, j])

    w_crisp, w_fuzzy = calc_fahp_buckley(M)
    return w_crisp, w_fuzzy


def run_e_prom_monte_carlo(
    evaluations,
    reference_rows,
    types,
    directions,
    preference_types,
    q_vals,
    p_vals,
    scales,
    continuous_variation_pct,
    n_mc=1000,
    seed=42,
    precision_levels=None,
    precision_support_pct=None,
    precision_core_pct=None,
):
    """
    Retorna uma análise de estabilidade do E-PROM.

    Para cada iteração:
      1. Reamostra os pesos FAHP-Express empiricamente;
      2. Para cada especialista:
         - ordinal: sorteia uma resposta observada entre os especialistas;
         - contínuo: perturba o valor observado em ±v%;
      3. Executa Fuzzy PROMETHEE individual;
      4. Agrega os fluxos dos especialistas a posteriori;
      5. Registra ranking, phi e matriz de aceitabilidade.

    Não há classificação KEY nem K-means: o E-PROM permanece genérico.
    """
    rng = np.random.default_rng(seed)

    experts = list(evaluations.keys())
    n_exp = len(experts)
    n_req, n_crit = np.asarray(evaluations[experts[0]]).shape

    variation = np.asarray(
        continuous_variation_pct,
        dtype=float,
    )
    if len(variation) != n_crit:
        raise ValueError("Variações contínuas incompatíveis com os critérios.")
    if np.any(variation < 0):
        raise ValueError("A variação percentual não pode ser negativa.")

    reference_rows = np.asarray(reference_rows, dtype=float)
    if reference_rows.shape != (n_exp, n_crit):
        raise ValueError(
            "reference_rows deve possuir dimensão "
            f"({n_exp},{n_crit})."
        )

    if precision_levels is None:
        precision_matrix = [["Alta"] * n_crit for _ in range(n_exp)]
    else:
        arr = np.asarray(precision_levels, dtype=object)
        if arr.ndim == 1:
            if len(arr) != n_crit:
                raise ValueError("Número de níveis de precisão incompatível.")
            precision_matrix = [list(arr) for _ in range(n_exp)]
        elif arr.ndim == 2:
            if arr.shape != (n_exp, n_crit):
                raise ValueError(
                    "A matriz de precisão deve possuir dimensão "
                    f"({n_exp}, {n_crit})."
                )
            precision_matrix = arr.tolist()
        else:
            raise ValueError("Formato de precisão inválido.")

    phi_mc = np.zeros((n_req, n_mc))
    phi_plus_mc = np.zeros((n_req, n_mc))
    phi_minus_mc = np.zeros((n_req, n_mc))
    rank_mc = np.zeros((n_req, n_mc))
    top1_count = np.zeros(n_req, dtype=int)

    W_samples = np.zeros((n_mc, n_crit))

    for mc in range(n_mc):

        # 1) FAHP-Express: amostragem empírica das respostas dos especialistas.
        w_crisp_mc, w_fuzzy_group = _sample_fahp_weights(
            reference_rows,
            rng,
        )
        W_samples[mc] = w_crisp_mc

        # Os fluxos são mantidos fuzzy durante toda a agregação.
        phi_plus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
        phi_minus_fuzzy_exp = np.zeros((n_req, 4, n_exp))
        phi_net_fuzzy_exp = np.zeros((n_req, 4, n_exp))

        # 2) Atributos por especialista.
        for e, expert in enumerate(experts):

            data_e = np.asarray(
                evaluations[expert],
                dtype=float,
            )

            data_mc = data_e.copy()

            for i in range(n_req):
                for k in range(n_crit):

                    if types[k] == "Ordinal":
                        # CPP empírico: distribuição observada entre especialistas.
                        vals = np.array([
                            evaluations[name][i, k]
                            for name in experts
                        ])
                        data_mc[i, k] = _empirical_sample(
                            vals,
                            rng,
                        )

                    elif types[k] == "Contínuo":
                        # Incerteza epistemológica/paramétrica simples:
                        # ±v% do valor observado.
                        v = variation[k] / 100.0
                        x = data_e[i, k]

                        if x == 0:
                            # Em torno de zero, a variação relativa é degenerada.
                            data_mc[i, k] = 0.0
                        else:
                            lo = x * (1.0 - v)
                            hi = x * (1.0 + v)

                            # Mantém a faixa ordenada para valores negativos.
                            lo, hi = min(lo, hi), max(lo, hi)
                            data_mc[i, k] = rng.uniform(lo, hi)

                    else:
                        raise ValueError(
                            f"Tipo de critério desconhecido: {types[k]!r}"
                        )

            F_e = build_fuzzy_matrix(
                data_mc,
                types,
                scales,
                precision_levels=precision_matrix[e],
                precision_support_pct=precision_support_pct,
                precision_core_pct=precision_core_pct,
            )

            # Para manter a lógica de grupo do MC-SIPREM, o mesmo vetor
            # de pesos da iteração é aplicado a cada especialista.
            output = run_fuzzy_promethee_trap(
                F_e,
                w_fuzzy_group,
                directions,
                preference_types,
                q_vals,
                p_vals,
                defuzzify=False,
            )

            (
                _,
                _,
                _,
                pp_f,
                pm_f,
                pn_f,
            ) = output

            phi_plus_fuzzy_exp[:, :, e] = pp_f
            phi_minus_fuzzy_exp[:, :, e] = pm_f
            phi_net_fuzzy_exp[:, :, e] = pn_f

        # 3) Agregação a posteriori NO DOMÍNIO FUZZY.
        # A defuzzificação ocorre somente após a agregação dos especialistas.
        phi_plus_fuzzy_now = np.mean(phi_plus_fuzzy_exp, axis=2)
        phi_minus_fuzzy_now = np.mean(phi_minus_fuzzy_exp, axis=2)
        phi_fuzzy_now = np.mean(phi_net_fuzzy_exp, axis=2)

        phi_plus_now = np.array([
            defuzz_coa(x) for x in phi_plus_fuzzy_now
        ])
        phi_minus_now = np.array([
            defuzz_coa(x) for x in phi_minus_fuzzy_now
        ])
        phi_now = np.array([
            defuzz_coa(x) for x in phi_fuzzy_now
        ])

        rank_now = rank_from_net_flow(phi_now)

        phi_plus_mc[:, mc] = phi_plus_now
        phi_minus_mc[:, mc] = phi_minus_now
        phi_mc[:, mc] = phi_now
        rank_mc[:, mc] = rank_now

        top1_count[
            np.where(rank_now == 1)[0]
        ] += 1

    # Matriz de aceitabilidade de ranking.
    rank_acceptability = np.zeros((n_req, n_req))
    for i in range(n_req):
        for r in range(1, n_req + 1):
            rank_acceptability[i, r - 1] = (
                100.0 * np.mean(rank_mc[i] == r)
            )

    # Matriz de probabilidade de sobreclassificação PROMETHEE I.
    outranking = np.zeros((n_req, n_req))
    tol = 1e-10

    for mc in range(n_mc):
        pp = phi_plus_mc[:, mc]
        pm = phi_minus_mc[:, mc]

        for i in range(n_req):
            for j in range(n_req):
                if i == j:
                    continue

                not_worse_plus = pp[i] >= pp[j] - tol
                not_worse_minus = pm[i] <= pm[j] + tol
                strictly_better = (
                    pp[i] > pp[j] + tol
                    or pm[i] < pm[j] - tol
                )

                if (
                    not_worse_plus
                    and not_worse_minus
                    and strictly_better
                ):
                    outranking[i, j] += 1

    outranking = 100.0 * outranking / n_mc

    return {
        "phi_mc": phi_mc,
        "phi_plus_mc": phi_plus_mc,
        "phi_minus_mc": phi_minus_mc,
        "rank_mc": rank_mc,
        "mean_rank": np.mean(rank_mc, axis=1),
        "std_rank": np.std(rank_mc, axis=1, ddof=1),
        "mean_phi": np.mean(phi_mc, axis=1),
        "std_phi": np.std(phi_mc, axis=1, ddof=1),
        "freq_top1": top1_count / n_mc,
        "rank_acceptability": rank_acceptability,
        "outranking_acceptability": outranking,
        "W_samples": W_samples,
        "n_mc": n_mc,
        "seed": seed,
        "continuous_variation_pct": variation,
        "precision_levels": precision_matrix,
    }
