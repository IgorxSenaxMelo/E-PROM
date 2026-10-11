# E-PROM

**Express Fuzzy Preference Ranking with PROMETHEE**

E-PROM integrates FAHP-Express (fuzzy criteria weighting) with Fuzzy PROMETHEE (multicriteria ranking), plus a Monte Carlo stability analysis.

## Included files

- `app.py` — Streamlit application
- `core/` — core modules (`data_loader`, `labels`, `fahp`, `e_prom_core`, `promethee`, `monte_carlo`, `fuzzy_numbers`; `cpp` is an unused experimental utility)
- `templates/E_PROM_Evaluations_Template.xlsx` — evaluation workbook template
- `templates/E_PROM_FAHP_Express_Template.xlsx` — FAHP-Express workbook template
- `GUIDE.md` — user guide (also shown inside the app) and `E_PROM_Guide.pdf` (same content)
- `tests/test_core.py` — regression tests (`pytest`)
- `requirements.txt` — Python dependencies

## Input languages

Sheet labels, precision levels (High/Medium/Low) and linguistic importance terms are accepted in English and Portuguese (Alta/Média/Baixa, "Igual importância", ...).

## Deployment

On Streamlit Community Cloud, set `app.py` as the main file. The `core/`, `templates/` folders and `GUIDE.md` must sit next to `app.py`.
