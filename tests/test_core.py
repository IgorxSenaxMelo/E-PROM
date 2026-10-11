import sys
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_loader import read_expert_workbook, read_fahp_workbook
from core.e_prom_core import (
    TFN_EVAL_5, TFN_EVAL_7, TFN_EVAL_9, build_fuzzy_matrix, continuous_to_fuzzy,
    fuzzify_ordinal, run_e_prom,
)
from core.fuzzy_numbers import defuzz_coa
from core.monte_carlo import run_e_prom_monte_carlo
from core.promethee import promethee_i_relation, rankdata_average

T = ROOT / "templates"
EVAL = T / "E_PROM_Evaluations_Template.xlsx"
FAHP = T / "E_PROM_FAHP_Express_Template.xlsx"

TYPES = ["Ordinal"] * 7 + ["Continuous"]
DIRS = [1] * 7 + [-1]
PREFS = [1] * 7 + [3]
QS = [0.0] * 8
PS = [0.0] * 7 + [10000.0]
SCALES = [9] * 7 + [0]


def load(eval_path=EVAL, fahp_path=FAHP):
    data = read_expert_workbook(eval_path)
    fahp = read_fahp_workbook(fahp_path, data["criteria"], data["expert_names"])
    return data, fahp


def test_templates_load_and_run():
    data, fahp = load()
    res = run_e_prom(data["evaluations"], fahp["w_fuzzy_individual"], TYPES, DIRS,
                     PREFS, QS, PS, SCALES, precision_levels=fahp["precision_levels"])
    assert list(res["rank_e_prom"]) == [2, 1, 3, 4]


def test_ordinal_scales_are_consistent():
    # Medium-precision trapezoid of every interior level is centred on its value.
    for table in (TFN_EVAL_7, TFN_EVAL_9):
        for v in range(2, len(table)):
            assert defuzz_coa(table[v - 1]) == pytest.approx(v, abs=1e-9)


def test_vectorized_fuzzification_matches_scalar():
    rng = np.random.default_rng(0)
    ordv = rng.integers(1, 10, 20).astype(float)
    cont = rng.uniform(1, 1e5, 20)
    for prec in ("High", "Medium", "Low", "Alta", "Média", "Baixa"):
        data = np.column_stack([ordv, cont])
        F = build_fuzzy_matrix(data, ["Ordinal", "Continuous"], [9, 0], [prec, prec])
        for i in range(20):
            assert np.allclose(F[i, 0], fuzzify_ordinal(ordv[i], 9, prec))
            assert np.allclose(F[i, 1], continuous_to_fuzzy(cont[i], prec))


def test_blank_precision_defaults_to_high(tmp_path):
    wb = load_workbook(FAHP)
    for ws in wb.worksheets:
        if ws.title in ("IGOR", "LUMA"):
            for c in range(2, 10):
                ws.cell(5, c).value = None
    p = tmp_path / "blank.xlsx"
    wb.save(p)
    data, fahp = load(fahp_path=p)
    assert all(x == "High" for row in fahp["precision_levels"] for x in row)
    run_e_prom(data["evaluations"], fahp["w_fuzzy_individual"], TYPES, DIRS, PREFS,
               QS, PS, SCALES, precision_levels=fahp["precision_levels"])


def test_promethee_i_weak_strict_rule():
    assert promethee_i_relation(0.5, 0.2, 0.3, 0.2) == "P"
    assert promethee_i_relation(0.3, 0.2, 0.5, 0.2) == "P-"
    assert promethee_i_relation(0.5, 0.1, 0.3, 0.2) == "P"
    assert promethee_i_relation(0.5, 0.3, 0.3, 0.2) == "R"
    assert promethee_i_relation(0.4, 0.2, 0.4, 0.2) == "I"


def test_ties_use_tolerance():
    assert list(rankdata_average([1.0, 1.0 + 1e-15, 0.5])) == [2.5, 2.5, 1.0]


def test_monte_carlo_precision_is_not_transposed_for_square_case():
    """n experts == n criteria used to silently transpose the precision."""
    rng = np.random.default_rng(0)
    n = 3
    ev = {f"E{e}": rng.integers(1, 10, (4, n)).astype(float) for e in range(n)}
    ref = np.array([[1, 2, 3], [1, 3, 2], [1, 2, 2]], float)
    prec = [["High", "Medium", "Low"], ["High", "High", "High"], ["Low", "Low", "Low"]]
    kw = dict(types=["Ordinal"] * n, directions=[1] * n, preference_types=[1] * n,
              q_vals=[0] * n, p_vals=[0] * n, scales=[9] * n,
              continuous_variation_pct=[0] * n, n_mc=50, seed=3)
    a = run_e_prom_monte_carlo(ev, ref, precision_levels=prec, **kw)
    b = run_e_prom_monte_carlo(ev, ref, precision_levels=np.array(prec, dtype=object).T.tolist(), **kw)
    assert not np.allclose(a["phi_mc"], b["phi_mc"])
    with pytest.raises(ValueError):
        run_e_prom_monte_carlo(
            {k: v[:, :2] for k, v in ev.items()}, ref[:, :2], types=["Ordinal"] * 2,
            directions=[1] * 2, preference_types=[1] * 2, q_vals=[0] * 2, p_vals=[0] * 2,
            scales=[9] * 2, continuous_variation_pct=[0] * 2, n_mc=5,
            precision_levels=np.array(prec, dtype=object)[:2, :2].T.tolist())


def test_monte_carlo_outputs_are_consistent():
    data, fahp = load()
    for mode in ("dm_bootstrap", "independent"):
        mc = run_e_prom_monte_carlo(
            data["evaluations"], fahp["reference_rows"], TYPES, DIRS, PREFS, QS, PS,
            SCALES, [0] * 7 + [20.0], n_mc=200, seed=1,
            precision_levels=fahp["precision_levels"], weight_sampling=mode)
        assert np.allclose(mc["rank_acceptability"].sum(axis=1), 100)
        assert mc["freq_top1"].sum() == pytest.approx(1.0)
        assert np.allclose(mc["W_samples"].sum(axis=1), 1)


def test_portuguese_workbooks_are_accepted(tmp_path):
    ev = load_workbook(EVAL)
    for ws in ev.worksheets:
        if ws.title in ("IGOR", "LUMA"):
            ws["B1"].value = "Descrição"
    ev.save(tmp_path / "ev_pt.xlsx")

    pt_terms = {
        "equal importance": "Igual importância",
        "weakly more important": "Fracamente mais importante",
        "moderately more important": "Moderadamente mais importante",
        "moderately to strongly more important": "Moderadamente a fortemente mais importante",
        "strongly more important": "Fortemente mais importante",
        "strongly to very strongly more important": "Fortemente a muito fortemente mais importante",
        "very strongly more important": "Muito fortemente mais importante",
        "very strongly to extremely more important": "Muito fortemente a extremamente mais importante",
        "extremely more important": "Extremamente mais importante",
    }
    pt_prec = {"High": "Alta", "Medium": "Média", "Low": "Baixa"}
    fh = load_workbook(FAHP)
    for ws in fh.worksheets:
        if ws.title in ("IGOR", "LUMA"):
            ws["A3"].value = "Critérios"
            ws["A4"].value = "Comparação com referência"
            ws["A5"].value = "Precisão do atributo"
            for c in range(2, 10):
                v = ws.cell(4, c).value
                ws.cell(4, c).value = pt_terms[v.lower()]
                ws.cell(5, c).value = pt_prec[ws.cell(5, c).value]
    fh.save(tmp_path / "fahp_pt.xlsx")

    d_en, f_en = load()
    d_pt, f_pt = load(tmp_path / "ev_pt.xlsx", tmp_path / "fahp_pt.xlsx")
    assert np.allclose(f_en["w_fuzzy_individual"], f_pt["w_fuzzy_individual"])
    assert f_en["precision_levels"] == f_pt["precision_levels"]


def test_order_of_experts_and_columns_does_not_matter(tmp_path):
    """FAHP sheets in a different order are paired by name."""
    fh = load_workbook(FAHP)
    fh.move_sheet("LUMA", offset=-1)
    fh.save(tmp_path / "reordered.xlsx")
    d, f = load(fahp_path=tmp_path / "reordered.xlsx")
    d0, f0 = load()
    assert f["expert_names"] == d["expert_names"]
    assert np.allclose(f["w_fuzzy_individual"], f0["w_fuzzy_individual"])
    assert f["warnings"] == []


def test_missing_guide_sheet_does_not_drop_a_decision_maker(tmp_path):
    ev = load_workbook(EVAL)
    del ev["Guide"]
    ev.save(tmp_path / "noguide.xlsx")
    d = read_expert_workbook(tmp_path / "noguide.xlsx")
    assert d["n_experts"] == 2


# ---------------------------------------------------------------- v8 features
from core.analysis import (
    criterion_contributions, expert_agreement, flow_terms, kendall_w,
    leave_one_expert_out, pairwise_comparison, rank_reversal_test, weight_sensitivity,
)
from core.config_io import build_config, criteria_defaults, dumps, parse_config
from core.e_prom_core import normalize_expert_weights
from core.flowsort import run_flowsort, suggest_profiles
from core.portfolio import greedy_by_rank, solve_portfolio
from core.report import build_report
from core.thresholds import nice_number, suggest_thresholds
from core.validation import (
    summarize, validate_against_config, validate_evaluation_workbook,
    validate_fahp_workbook,
)


def _run():
    data, fahp = load()
    args = (data["evaluations"], fahp["w_fuzzy_individual"], TYPES, DIRS, PREFS, QS, PS, SCALES)
    res = run_e_prom(*args, precision_levels=fahp["precision_levels"])
    return data, fahp, args, res


def test_contributions_and_pairwise_are_consistent_with_flows():
    data, fahp, args, res = _run()
    terms = flow_terms(*args, precision_levels=fahp["precision_levels"])
    contrib = criterion_contributions(terms)
    assert np.allclose(contrib.sum(axis=1), res["phi_net"], atol=0.01)
    pw = pairwise_comparison(*args, 0, 1, precision_levels=fahp["precision_levels"])
    assert pw["total_ij"] == pytest.approx(pw["pref_ij"].sum(), abs=0.01)


def test_weight_sensitivity_boundaries_change_the_ranking():
    data, fahp, args, res = _run()
    terms = flow_terms(*args, precision_levels=fahp["precision_levels"])
    sens = weight_sensitivity(terms, fahp["w_fuzzy_individual"])
    base = res["rank_e_prom"]
    for s in sens:
        g = int(np.argmin(np.abs(s["sweep_share"] - s["share_now"])))
        assert np.allclose(s["sweep_rank"][:, g], base)           # a = 1 is the current analysis
        assert s["order_lower"] <= s["share_now"] <= s["order_upper"]


def test_sensitivity_matches_a_full_rerun():
    """Scaling criterion 0 by the boundary factor must flip the ranking just outside it."""
    data, fahp, args, res = _run()
    terms = flow_terms(*args, precision_levels=fahp["precision_levels"])
    s = weight_sensitivity(terms, fahp["w_fuzzy_individual"])[0]
    assert s["order_changes_up"]
    w = fahp["w_fuzzy_individual"]
    shares = w.mean(axis=1) / w.mean(axis=1).sum(axis=0)
    c = shares[0]

    def group_share(a):
        return float(np.mean(a * c / (a * c + 1 - c)))

    lo, hi = 1e-4, 1e4
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if group_share(mid) < s["order_upper"] else (lo, mid)
    a = mid

    def rerun(factor):
        ww = w.copy()
        ww[0] *= factor
        norm = factor * c + (1 - c)
        return run_e_prom(args[0], ww / norm[None, None, :], *args[2:],
                          precision_levels=fahp["precision_levels"])["rank_e_prom"]

    assert np.allclose(rerun(a * 0.98), res["rank_e_prom"])
    assert not np.allclose(rerun(a * 1.02), res["rank_e_prom"])


def test_agreement_and_robustness():
    data, fahp, args, res = _run()
    ag = expert_agreement(res["phi_net_experts"], res["expert_names"])
    assert 0 <= ag["kendall_w"] <= 1
    assert ag["spearman"].shape == (2, 2)
    loo = leave_one_expert_out(*args, precision_levels=fahp["precision_levels"])
    assert len(loo) == 2
    rev = rank_reversal_test(*args, precision_levels=fahp["precision_levels"])
    assert len(rev) == 4
    assert kendall_w(np.array([[1, 1], [2, 2], [3, 3]], float)) == pytest.approx(1.0)


def test_expert_weights_change_aggregation():
    data, fahp, args, res = _run()
    heavy = run_e_prom(*args, precision_levels=fahp["precision_levels"], expert_weights=[1, 0])
    alone = run_e_prom({"IGOR": data["evaluations"]["IGOR"]}, fahp["w_fuzzy_individual"][:, :, :1],
                       *args[2:], precision_levels=fahp["precision_levels"][:1])
    assert np.allclose(heavy["phi_net"], alone["phi_net"])
    assert np.allclose(normalize_expert_weights([2, 1, 1], 3), [0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        normalize_expert_weights([0, 0], 2)


def test_flowsort_is_monotone_and_probabilities_sum_to_100():
    data, fahp, args, res = _run()
    prof = suggest_profiles(data["evaluations"], TYPES, DIRS, SCALES, 3)
    fs = run_flowsort(data["evaluations"], fahp["w_fuzzy_individual"], prof, TYPES, DIRS,
                      PREFS, QS, PS, SCALES, precision_levels=fahp["precision_levels"])
    assert fs["profile_order_ok"]
    order = np.argsort(-fs["phi"])
    assert np.all(np.diff(fs["class_index"][order]) >= 0)     # better flow -> better (lower) class
    mc = run_e_prom_monte_carlo(data["evaluations"], fahp["reference_rows"], TYPES, DIRS, PREFS,
                                QS, PS, SCALES, [0] * 7 + [20.0], n_mc=60, seed=2,
                                precision_levels=fahp["precision_levels"], flowsort_profiles=prof)
    assert np.allclose(mc["class_acceptability"].sum(axis=1), 100)


def test_portfolio_beats_or_matches_the_greedy_baseline():
    phi = np.array([0.30, 0.25, 0.20, -0.05, -0.20])
    cost = np.array([60.0, 50.0, 45.0, 10.0, 5.0])
    best = solve_portfolio(phi, [cost], [100.0])
    base = greedy_by_rank(phi, [cost], [100.0])
    assert best["objective"] >= base["objective"] - 1e-9
    assert best["usage"][0] <= 100.0 + 1e-9
    assert list(best["selected"][:3]) == [0, 0, 0] or best["objective"] == pytest.approx(0.45)
    forced = solve_portfolio(phi, [cost], [100.0], must_include=[4])
    assert forced["selected"][4] == 1
    assert not solve_portfolio(phi, [cost], [1.0], min_count=1)["feasible"]


def test_validation_reports_cells(tmp_path):
    assert summarize(validate_evaluation_workbook(EVAL))["errors"] == 0
    wb = load_workbook(EVAL)
    wb["IGOR"]["C2"].value = None
    wb["IGOR"]["D3"].value = "abc"
    p = tmp_path / "bad.xlsx"
    wb.save(p)
    issues = validate_evaluation_workbook(p)
    assert {i["cell"] for i in issues if i["severity"] == "error"} == {"C2", "D3"}

    fh = load_workbook(FAHP)
    fh["IGOR"]["C4"].value = "Super important"
    q = tmp_path / "badf.xlsx"
    fh.save(q)
    data = read_expert_workbook(EVAL)
    bad = validate_fahp_workbook(q, data["criteria"], data["expert_names"])
    assert any(i["cell"] == "C4" for i in bad)

    data = read_expert_workbook(EVAL)
    cfg_issues = validate_against_config(data, TYPES, [5] * 7 + [0])      # values up to 9 on a 5-scale
    assert summarize(cfg_issues)["errors"] > 0


def test_config_round_trip_and_name_matching():
    data = read_expert_workbook(EVAL)
    cfg = build_config(data["criteria"], TYPES, DIRS, ["Usual"] * 7 + ["V-shape"], QS, PS, SCALES)
    parsed = parse_config(dumps(cfg))
    defaults, warns = criteria_defaults(parsed, list(reversed(data["criteria"])))   # different order
    assert defaults[0]["preference"] == "V-shape" and defaults[0]["p"] == 10000.0
    assert warns == []
    with pytest.raises(ValueError):
        parse_config(b"{not json")


def test_threshold_suggestions_are_valid():
    data, fahp = load()
    for s, t in zip(suggest_thresholds(data["evaluations"], TYPES, SCALES, fahp["precision_levels"]), TYPES):
        assert s["p"] > s["q"] >= 0
        assert s["preference"] == ("Level" if t == "Ordinal" else "Linear")
    assert nice_number(4300, "up") == 5000 and nice_number(0.8, "down") == 0.5


def test_report_pdf_is_generated():
    import pandas as pd
    data, fahp, args, res = _run()
    terms = flow_terms(*args, precision_levels=fahp["precision_levels"])
    cfg = pd.DataFrame({"Criterion": data["criteria"], "Type": TYPES})
    pdf = build_report(data, res, cfg, fahp, {
        "contributions": criterion_contributions(terms),
        "sensitivity": weight_sensitivity(terms, fahp["w_fuzzy_individual"]),
        "agreement": expert_agreement(res["phi_net_experts"], res["expert_names"]),
    })
    assert pdf[:4] == b"%PDF" and len(pdf) > 5000
