# E-PROM

**Express Fuzzy Preference Ranking with PROMETHEE**

E-PROM integrates FAHP-Express (fuzzy criteria weighting) with Fuzzy PROMETHEE (multicriteria ranking) and adds a toolkit to understand, test and report the result.

## Features

- Input validation with cell-level error report; save / load configuration (JSON)
- Threshold assistant (suggested `q` / `p`) and preference-function preview
- Ranking, PROMETHEE I relations, flows per decision maker, optional decision-maker weights
- **Explain**: contribution of each criterion to the net flow; pairwise comparison
- **Sensitivity**: weight-stability ranges, sweeps, leave-one-decision-maker-out, rank-reversal test
- **Agreement** among decision makers (Kendall's W, Spearman)
- **Monte Carlo** stability analysis (rank acceptability, outranking acceptability, weight spread)
- **Classification** with FlowSort (and Monte Carlo class probabilities)
- **Portfolio** selection with PROMETHEE V (budgets, number limits, mandatory / excluded)
- **PDF report** with everything run in the session

## Included files

- `app.py` — Streamlit application
- `core/` — core modules: `data_loader`, `validation`, `labels`, `fahp`, `e_prom_core`, `promethee`, `analysis`, `monte_carlo`, `flowsort`, `portfolio`, `thresholds`, `config_io`, `report`, `fuzzy_numbers` (`cpp` is an unused experimental utility)
- `templates/` — evaluation and FAHP-Express workbook templates
- `GUIDE.md` — user guide (shown inside the app) and `E_PROM_Guide.pdf` (same content)
- `tests/test_core.py` — regression tests (`pytest`)
- `requirements.txt` — Python dependencies

## Input languages

Sheet labels, precision levels (High/Medium/Low) and linguistic importance terms are accepted in English and Portuguese (Alta/Média/Baixa, "Igual importância", ...).

## Deployment

On Streamlit Community Cloud, set `app.py` as the main file. The `core/`, `templates/` folders and `GUIDE.md` must sit next to `app.py`. Upload the whole project: partial updates of `core/` break the app.
