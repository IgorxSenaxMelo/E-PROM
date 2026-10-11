# E-PROM — User Guide

**Express Fuzzy Preference Ranking with PROMETHEE**

E-PROM combines three building blocks:

```
FAHP-Express  →  fuzzy criterion weights  →  Fuzzy PROMETHEE  →  ranking
                                                     │
        Explain · Sensitivity · Agreement · Monte Carlo · FlowSort · Portfolio · Report
```

Each decision maker (DM) evaluates the alternatives and supplies a short set of
importance judgments. E-PROM runs the whole method **once per DM**, keeps every
quantity fuzzy, aggregates the DMs' flows, and defuzzifies only at the very end.

---

## 1. Quick start

1. Fill the **evaluation workbook** (one sheet per DM) and the **FAHP-Express workbook** (one sheet per DM).
2. Upload both files in the sidebar. The **input validation** panel lists every problem with its sheet and cell.
3. For each criterion choose: data type, direction, preference function, and `q` / `p`. The **threshold assistant** suggests starting values for `q` and `p`; you can **save / load** the whole configuration as a JSON file.
4. Click **Run E-PROM** → ranking, flows, PROMETHEE I relations.
5. Use the analysis tabs: **Explain** (why this ranking), **Sensitivity** (how fragile it is), **Monte Carlo** (stability under uncertainty), **Classification** (ordered classes), **Portfolio** (best set under a budget).
6. Download the Excel / CSV results or a complete **PDF report**.

---

## 2. Input files

### 2.1 Evaluation workbook

| Sheet | Content |
|---|---|
| One sheet per DM (sheet name = DM name) | Column A: alternative ID · Column B: description · Column C onward: criteria |
| Sheets named `Guide`, `_Lists` or `Instructions` | Support sheets, ignored by the app. Every other sheet is read as a decision maker. |

Rules:

- All DMs must use the same alternative IDs and the same criterion names. The *order* of rows and criteria may differ between sheets (they are aligned by ID and by name; names ignore case and accents).
- Ordinal criteria: integers inside the scale you select (**5, 7 or 9** — a 1–10 scale is not supported).
- Continuous criteria: any real number, in the original unit (cost in R$, schedule or time in months, distance, consumption, physical performance, ...). Values are never converted to a 1–7 scale.
- No empty cells.

| ID | Description | Impact (ordinal 1–9) | Cost (continuous, R$) | Deadline (continuous, months) |
|---|---|---:|---:|---:|
| R1 | Alternative 1 | 7 | 300000 | 12 |
| R2 | Alternative 2 | 5 | 250000 | 18 |

### 2.2 FAHP-Express workbook

One sheet per DM, plus the ignored `Guide` / `_Lists` sheets. Each DM sheet has:

| Row label | What to fill |
|---|---|
| `Criteria` | The criterion names (same names as the evaluation workbook). **The first criterion is that DM's reference criterion.** |
| `Comparison with reference` | For every criterion, how many times more important the reference is than that criterion (linguistic term, see §5). The reference itself is *Equal importance*. |
| `Attribute precision` | `High`, `Medium` or `Low` for every criterion (see §4). A blank cell is treated as `High`. |

The sheets must have the same names as in the evaluation workbook: decision makers are paired **by sheet name** (if the names differ, they are paired by position and the app shows a warning). The criteria must be the same as in the evaluation workbook. Comparison values can be linguistic terms (English or Portuguese) or the numbers 1–9; precision labels also accept the Portuguese forms `Alta`, `Média`, `Baixa`. The numeric fuzzy parameters of each precision level are fixed internally — you only choose the label.

Each DM may choose a **different reference criterion**. The reference should be the criterion that DM considers most important, so every other factor is ≥ 1.

> `Attribute precision` is stored in the FAHP-Express sheet for convenience, but it describes the **evaluations** of that DM on that criterion. It does **not** change the weights.

---

## 3. Fuzzy numbers — what they are and why E-PROM uses them

### 3.1 Trapezoidal fuzzy number

A judgment such as "about 7" is not a point. E-PROM represents it as a trapezoid
`[a, b, c, d]`:

```
membership
   1 |        ____________
     |       /|          |\
     |      / |   core   | \
   0 |_____/__|__________|__\_____ value
          a   b          c   d
          └── support ───────┘
```

- **Core** `[b, c]` — values that are fully compatible with the judgment (membership 1).
- **Support** `[a, d]` — values that are possible at all (membership > 0).
- A crisp number is the special case `a = b = c = d`.
- The wider the trapezoid, the more uncertain the judgment.

### 3.2 Arithmetic used

For `A = [a₁, b₁, c₁, d₁]` and `B = [a₂, b₂, c₂, d₂]`:

| Operation | Rule |
|---|---|
| Addition | `[a₁+a₂, b₁+b₂, c₁+c₂, d₁+d₂]` |
| Subtraction (interval arithmetic) | `[a₁−d₂, b₁−c₂, c₁−b₂, d₁−a₂]` |
| Division by a scalar `s` | `[a₁/s, b₁/s, c₁/s, d₁/s]` |
| Multiplication (weight × preference) | Dubois–Prade / Geldermann LR approximation: modal values multiply (`b₁b₂`, `c₁c₂`); spreads follow `left = b₁α₂ + b₂α₁ − α₁α₂`, `right = c₁β₂ + c₂β₁ + β₁β₂`, where α = b−a and β = d−c (negative spreads are set to 0) |
| Defuzzification (Center of Area) | `x̄ = [(c+d)² − cd − (a+b)² + ab] / [3(c+d−a−b)]` |

Subtraction widens the result (the width of `A − B` is the sum of both widths). This is
intentional: the difference between two uncertain values is itself uncertain.

### 3.3 Where fuzziness enters E-PROM

| Stage | Fuzzy object | Defuzzified? |
|---|---|---|
| Evaluations (ordinal and continuous) | trapezoid built from the observed value and its precision (§4) | No |
| Criterion weights | trapezoid from FAHP-Express (§5) | No |
| Pairwise preference | `weight ⊗ preference` (§6) | No |
| Flows Φ⁺, Φ⁻, Φ (per DM) | trapezoids | No |
| Group flows (mean over DMs) | trapezoids | **Yes — once, by COA**, to rank |

---

## 4. Precision — how uncertain is each evaluation?

Precision answers: *"how sure is this DM about this evaluation on this criterion?"*
It is set **per DM × criterion** (`Attribute precision` row) and uses three labels:

| Label | Meaning | Fuzzy width |
|---|---|---|
| **High** | The DM is confident / the data is exact | narrow |
| **Medium** | Default linguistic representation | intermediate |
| **Low** | The DM is unsure / the data is rough | wide |

Precision is monotone: **High ⇒ narrower trapezoid ⇒ less uncertainty.**
The three labels act on different scales depending on the criterion type.

### 4.1 Ordinal criteria (scales 5, 7, 9) — widths in scale units

Each ordinal value `v` has a base trapezoid (the **Medium** representation):

**9-point scale**

| v | [a, b, c, d] | v | [a, b, c, d] |
|:-:|---|:-:|---|
| 1 | [1, 1, 1.5, 2] | 6 | [5, 5.5, 6.5, 7] |
| 2 | [1, 1.5, 2.5, 3] | 7 | [6, 6.5, 7.5, 8] |
| 3 | [2, 2.5, 3.5, 4] | 8 | [7, 7.5, 8.5, 9] |
| 4 | [3, 3.5, 4.5, 5] | 9 | [8, 8.5, 9, 9] |
| 5 | [4, 4.5, 5.5, 6] | | |

**7-point scale**: 1 = [1, 1, 1.5, 2] · 2 = [1, 1.5, 2.5, 3] · 3 = [2, 2.5, 3.5, 4] · 4 = [3, 3.5, 4.5, 5] · 5 = [4, 4.5, 5.5, 6] · 6 = [5, 5.5, 6.5, 7] · 7 = [6, 6.5, 7, 7]

**5-point scale**: 1 = [1, 1, 1.5, 2] · 2 = [1.5, 2, 2, 2.5] · 3 = [2, 2.5, 3, 3.5] · 4 = [3, 3.5, 4, 4.5] · 5 = [4, 4.5, 5, 5]

Precision then **contracts or dilates the base trapezoid around `v`**:

```
fuzzy = v + factor × (base − v),   then clipped to [1, scale max]
factor:  High = 0.5   Medium = 1.0   Low = 1.5
```

Example — `v = 7` on the 9-point scale:

| Precision | Trapezoid |
|---|---|
| High | [6.5, 6.75, 7.25, 7.5] |
| Medium | [6, 6.5, 7.5, 8] |
| Low | [5.5, 6.25, 7.75, 8.5] |

Values at the ends of the scale are truncated by the clipping, so their trapezoids are one-sided.
**The same number means different things on different scales** (a 5 on a 5-point scale is the top; a 5 on a 9-point scale is the middle) — always select the scale the DMs actually used.

### 4.2 Continuous criteria — relative width in %

The value stays in its **original unit**. A value `x` becomes

```
[ x(1 − r_s),  x(1 − r_c),  x(1 + r_c),  x(1 + r_s) ]
 r_s = support half-width (outer)     r_c = core half-width (inner)
```

| Precision | Support r_s | Core r_c |
|---|---:|---:|
| High | ±5 % | ±2 % |
| Medium | ±10 % | ±4 % |
| Low | ±16.67 % | ±6.67 % |

Example — a cost of R$ 30,000:

| Precision | Trapezoid (R$) |
|---|---|
| High | [28,500, 29,400, 30,600, 31,500] |
| Medium | [27,000, 28,800, 31,200, 33,000] |
| Low | [25,000, 28,000, 32,000, 35,000] |

Notes:

- The width is **proportional to the value** — expensive alternatives carry a wider absolute band than cheap ones.
- A value of exactly `0` has no width (it stays `[0, 0, 0, 0]`).
- Use **High** for contractual/measured data, **Medium** for engineering estimates, **Low** for rough or expert-guess figures.

### 4.3 How precision interacts with `q` and `p`

The preference function is applied to the **fuzzy difference** of two alternatives, not to the
difference of the crisp values. For two alternatives with the same precision on a continuous criterion,
the fuzzy difference spans roughly `± 2·r_s·x` around the crisp difference.

Practical consequences:

- If `q` is **smaller** than that spread (≈ `2·r_s·x`), two nearly equal alternatives will still get a small, non-zero preference in *both* directions. It cancels out in the net flow Φ, but it inflates Φ⁺ and Φ⁻ (and can affect PROMETHEE I relations).
- Rule of thumb: choose `q` **at least as large as** `2·r_s·x` for typical values `x` if you want near-ties to be true indifference.
- With the **Usual** function on ordinal criteria, identical ratings produce a preference of about 0.5 in each direction for the same reason. This is expected behavior of fuzzy PROMETHEE.

---

## 5. FAHP-Express — fuzzy criterion weights

### 5.1 The "Express" idea

Classical AHP asks `n(n−1)/2` pairwise comparisons. FAHP-Express asks **only `n−1` per DM**:
every criterion is compared with a single reference criterion.

For a DM with values `v₁ … vₙ` (reference has `v = 1`):

```
A_ij = v_j / v_i        (A_1j = v_j,   A_j1 = 1 / v_j)
```

So the complete comparison matrix is rebuilt from one row. It is **perfectly consistent by construction** (consistency ratio = 0), so there is no consistency check to perform.

`v_j` reads: *"the reference criterion is `v_j` times more important than criterion j."*

### 5.2 Linguistic scale and its fuzzy form

| v | Linguistic term | Trapezoid |
|:-:|---|---|
| 1 | Equal importance | [1, 1, 1, 1] |
| 2 | Weakly more important | [1, 1.5, 2.5, 3] |
| 3 | Moderately more important | [2, 2.5, 3.5, 4] |
| 4 | Moderately to strongly more important | [3, 3.5, 4.5, 5] |
| 5 | Strongly more important | [4, 4.5, 5.5, 6] |
| 6 | Strongly to very strongly more important | [5, 5.5, 6.5, 7] |
| 7 | Very strongly more important | [6, 6.5, 7.5, 8] |
| 8 | Very strongly to extremely more important | [7, 7.5, 8.5, 9] |
| 9 | Extremely more important | [8, 8.5, 9, 9] |

- Ratios `A_ij = v_j / v_i` are generally **not integers** (e.g. 6/4 = 1.5). They are fuzzified by **linear interpolation** between the two neighbouring rows of the table.
- Ratios below 1 use the reciprocal: `[1/d, 1/c, 1/b, 1/a]`.
- Any ratio outside `[1/9, 9]` is rejected.

### 5.3 Weights (Buckley)

For each criterion `i`:

1. `r_i` = geometric mean of row `i` of the fuzzy matrix (computed component-wise).
2. `w_i = r_i ⊗ (Σ r)⁻¹`, where `(Σ r)⁻¹ = [1/Σd, 1/Σc, 1/Σb, 1/Σa]`.

The result is a **fuzzy weight** (trapezoid) per criterion and per DM. A crisp weight (mean of the four
numbers, normalized to 1) is computed only for display and for the Monte Carlo weight summary.
**Fuzzy weights are what actually enter PROMETHEE.**

---

## 6. Fuzzy PROMETHEE

### 6.1 Criterion configuration

| Type | Allowed functions | Notes |
|---|---|---|
| **Ordinal** | Usual · Level | `q`, `p` in *scale units* |
| **Continuous** | U-shape · V-shape · Linear | `q`, `p` in the *original unit* |

**Direction:** *Maximize* (more is better) or *Minimize* (less is better, e.g. cost).

### 6.2 Preference functions (for a difference `d ≥ 0`)

| Function | `P(d)` | Parameters |
|---|---|---|
| **Usual** | `0` if `d ≤ 0`, else `1` | none |
| **U-shape** | `0` if `d ≤ q`, else `1` | `q ≥ 0` |
| **V-shape** | `d/p` if `d < p`, else `1` | `p > 0` |
| **Level** | `0` if `d ≤ q`; `0.5` if `q < d ≤ p`; `1` if `d > p` | `0 ≤ q < p` |
| **Linear** | `0` if `d ≤ q`; `(d−q)/(p−q)` if `q < d < p`; `1` if `d ≥ p` | `0 ≤ q < p` |

- `q` — **indifference threshold**: differences up to `q` are ignored.
- `p` — **preference threshold**: differences from `p` on are a strict preference.
- Always express `q` and `p` in the criterion's original unit. Example: cost in R$: `q = 10000`, `p = 50000`; deadline in months: `q = 1`, `p = 4`.

### 6.3 Calculation (per DM)

For every ordered pair of alternatives `(i, j)`:

1. **Fuzzy difference** on criterion `k`: `D = F_ik ⊖ F_jk` (for *Minimize*: `F_jk ⊖ F_ik`).
2. **Fuzzy preference**: the preference function is applied to the four corners of `D` (the function is non-decreasing, so the result is again a trapezoid `P_k`).
3. **Weighted preference**: `π(i,j) = Σₖ  w_k ⊗ P_k`.
4. **Flows**:
   - `Φ⁺(i) = (1/(n−1)) Σⱼ π(i,j)` — how much `i` outranks the others;
   - `Φ⁻(i) = (1/(n−1)) Σⱼ π(j,i)` — how much `i` is outranked;
   - `Φ(i) = Φ⁺(i) ⊖ Φ⁻(i)` — net flow.

### 6.4 Group aggregation

The DMs' fuzzy flows are **averaged in the fuzzy domain** — a simple mean, or a weighted mean if you set decision-maker weights (a-posteriori aggregation); the group
`Φ⁺`, `Φ⁻` and `Φ` are then defuzzified by COA. The ranking sorts alternatives by the defuzzified net flow Φ
(rank 1 = highest Φ; ties share the average rank).

### 6.5 PROMETHEE I relations

Using the defuzzified `Φ⁺` and `Φ⁻`:

| Relation | Condition |
|---|---|
| **P** (i outranks j) | `Φ⁺(i) ≥ Φ⁺(j)` and `Φ⁻(i) ≤ Φ⁻(j)`, with at least one strict |
| **I** (indifferent) | both flows equal |
| **R** (incomparable) | `i` is better on one flow and worse on the other |

---

## 7. Monte Carlo (CPP-style) stability analysis

A single deterministic ranking hides how much it depends on uncertain inputs. The Monte Carlo module
repeats E-PROM many times, each time drawing the uncertain inputs from probability distributions,
and reports **probabilities** instead of a single answer. It is a *probabilistic preference* analysis
in the spirit of CPP (Composition of Probabilistic Preferences): the input is a distribution, the
output is the probability of each ranking outcome.

### 7.1 What is simulated in each iteration

| Input | How it is drawn |
|---|---|
| **Ordinal evaluation** (per alternative × criterion × DM) | **Empirical distribution** — one of the answers given by the DMs for that alternative and criterion, drawn with its observed frequency. Example: answers `7, 7, 6, 5, 7, 6, 7, 4, 7, 6` → P(7)=0.5, P(6)=0.3, P(5)=0.1, P(4)=0.1. |
| **Continuous evaluation** | **Uniform** in `[x(1−v), x(1+v)]`, where `v` is the % variation you set per criterion (e.g. 20 % → `[0.8x, 1.2x]`). |
| **Criterion weights** | Each iteration **bootstraps the decision makers**: it draws, with replacement, as many DMs as there are in the panel, takes their complete FAHP-Express rows, and averages their fuzzy weights. Whole rows are resampled, so it stays consistent when DMs use different reference criteria. The same weight vector is used for all DMs in that iteration. With very few DMs (e.g. 2) there are only a few distinct weight combinations, so the weight variability is limited. A legacy mode (each criterion draws its own value) is available; use it only if every DM has the same reference criterion. |
| **Precision** | Unchanged; each DM keeps their own precision for the fuzzification. |

Then, in each iteration: fuzzy PROMETHEE per DM → average of the fuzzy flows → COA → ranking.

> **Two kinds of uncertainty.** *Precision* (§4) is the DM's imprecision about a judgment and lives *inside* the fuzzy number. The *Monte Carlo variation* is a scenario analysis *around* it. For continuous criteria they act together, so a large variation on top of Low precision is very conservative.

### 7.2 Outputs

| Output | How to read it |
|---|---|
| **Mean position / SD of position** | Average rank over the simulations and its dispersion. Small SD = stable position. |
| **First-place frequency** | % of simulations in which the alternative is ranked 1st (ties for 1st share the credit). |
| **Mean Φ / SD of Φ** | Average net flow and its dispersion. |
| **Rank acceptability matrix** | % of simulations in which alternative *i* ends in position *r*. Each row sums to 100 %. Alternatives tied in a simulation share the tied positions equally. |
| **Outranking acceptability (PROMETHEE I)** | % of simulations in which *i* outranks *j* (P relation). |
| **Weight distribution** | Mean, SD, min and max of each criterion's crisp weight across simulations. |

### 7.3 Number of simulations

The Monte Carlo error of a probability `p` is about `√(p(1−p)/N)`. With `N = 2000` it is ≤ 1.1 percentage points.
Use ≥ 1000 for exploration and ≥ 5000 for results you will report. The seed makes runs reproducible.

### 7.4 How to interpret

- Use the analysis as a **stability check**, not as a replacement for the deterministic ranking.
- A deterministic leader with a high first-place frequency (and a narrow rank distribution) is a **robust** choice.
- Two alternatives with similar first-place frequencies and a low outranking acceptability in both directions are **not distinguishable** under the stated uncertainty — report them as a tie group.
- If the weight distribution is very wide for a criterion, the DMs disagree about its importance; consider a consensus round.

---

E-PROM is domain-generic: it has no `KEY` classification and no military-domain groups (those belong to SIPREM). It produces a ranking and PROMETHEE I relations.

Templates: `E_PROM_Evaluations_Template.xlsx` and `E_PROM_FAHP_Express_Template.xlsx` (downloadable inside the app). English and Portuguese labels are both accepted in the workbooks.

---

## 8. Analysis tools

### 8.1 Input validation

As soon as a workbook is uploaded the app checks it and shows **every** problem at once, each with its sheet and cell: empty or non-numeric cells, duplicated IDs, criteria or IDs that differ between decision makers, unknown linguistic terms or precision labels, a reference criterion that is not `Equal importance`, values below 1 in the reference row, ratios above 9, and mismatched sheet names. Errors block the run; warnings and notes do not. Before running, the values are also checked against the data types you configured (e.g. an ordinal value of 9 on a 7-point scale).

### 8.2 Save / load configuration

The sidebar can save the criteria setup (type, direction, preference function, `q`, `p`, scale), the decision-maker weights and the Monte Carlo settings to a small JSON file, and load it again later. Criteria are matched by **name**, so the file keeps working if the column order changes. The same file is appended to the PDF report for reproducibility.

### 8.3 Threshold assistant (suggested q and p)

For every criterion the assistant proposes a preference function and thresholds from your data and the attribute precision (see §4.3):

- **Continuous:** Linear; `q` = fuzzy width of two equal values (about `2 · r_s · typical value`, rounded to a "nice" number); `p` = 75th percentile of the pairwise differences between alternatives, at least `2q`.
- **Ordinal:** Level; `q` / `p` at the 25th / 75th percentiles of the pairwise differences, in scale units.

These are **starting points**, not recommendations: the thresholds express your preferences, so review them. A plot shows the current and suggested preference functions, and you can apply the suggestions to selected criteria.

### 8.4 Explain: contribution and pairwise comparison

- **Contribution by criterion.** Each stacked bar splits the net flow of an alternative into the weighted contribution of each criterion (`COA(Φ⁺_k) − COA(Φ⁻_k)`). Positive parts help the alternative, negative parts hurt it. Defuzzification is not exactly additive, so the sum can differ slightly from the net flow.
- **Pairwise comparison.** Pick two alternatives A and B to see, criterion by criterion, how much A is preferred to B and B to A, the criterion weights, the (average) values and which one each criterion favours.

### 8.5 Sensitivity to the weights

For each criterion the weight of every decision maker is multiplied by a factor and the weights are renormalized (the other criteria keep their relative proportions). The tool reports the **range of the weight share, around the current value, inside which the full ranking — and the first place — do not change**, and plots the net flows and positions along the sweep. Narrow ranges mean the result is fragile with respect to that weight. Ranges ending at 0 % or 100 % mean no change was found down to zero / up to the maximum explored.

### 8.6 Agreement among decision makers and robustness

- **Kendall's W** (concordance of the individual rankings, with tie correction): below 0.3 weak, 0.3–0.6 moderate, above 0.6 strong agreement. The **Spearman matrix** shows the agreement between each pair of DMs; a DM whose mean Spearman is below 0.5 (with three or more DMs) is flagged.
- **Leave-one-decision-maker-out** re-runs the analysis without each DM in turn and reports whether the ranking or the first place changes.
- **Rank-reversal test** removes each alternative in turn and lists the pairs whose relative order flips. PROMETHEE II is known to allow reversals; a result with none is more reassuring.
- **Decision-maker weights** (sidebar, optional) give some DMs more influence: the fuzzy flows are combined with a weighted mean instead of a simple mean (equal weights by default). They also apply to the Monte Carlo.

### 8.7 Classification — FlowSort

Instead of ranking, FlowSort assigns each alternative to **ordered classes** separated by **limiting profiles**:

1. Choose the number of classes (2–6) and their names (class 1 is the best).
2. Edit the limiting profiles — one row per boundary, from the **worst** boundary (P1) to the **best** one, written in the same units as your data (ordinal values must be integers in the scale). The defaults are quantiles of your data.
3. Each alternative is compared with the profiles inside the reference set `R ∪ {a}` using the same fuzzy preferences, weights and thresholds as the ranking; it is assigned to the class whose bounding profiles enclose its net flow: `Φ(r_h) ≤ Φ(a) < Φ(r_{h+1})`. DM flows are aggregated in the fuzzy domain before defuzzification.
4. The table also shows the class proposed by each DM and, optionally, the **probability of each class** from the Monte Carlo (same settings as the Monte Carlo tab).

The profiles must get strictly better from P1 to the last one on the weighted criteria; the app warns when their net flows are not increasing.

### 8.8 Portfolio selection — PROMETHEE V

When you can pick several alternatives, ranking is not enough. PROMETHEE V maximizes the **total net flow of the selected set** subject to:

- **resource budgets** on continuous criteria (e.g. cost; values are averaged over the DMs);
- a minimum and maximum **number** of alternatives;
- **mandatory** and **excluded** alternatives.

The problem is solved exactly as a 0-1 integer program. The result is compared with the *ranking-order baseline* (walk down the ranking keeping what still fits): the optimum can be better because a lower-ranked alternative may use the budget more efficiently. Alternatives with a negative net flow are chosen only when needed to reach the minimum number.

### 8.9 PDF report

The **Report** tab builds a PDF with the ranking, PROMETHEE I relations, configuration, precision, weights and every analysis you ran in the session (contribution, sensitivity, agreement, robustness checks, Monte Carlo, classification, portfolio), plus the configuration file for reproducibility.

---

## 9. Recommended workflow

1. Define alternatives and criteria; decide for each criterion its type (ordinal / continuous) and direction.
2. Collect each DM's evaluations (same IDs, same criterion names).
3. Collect each DM's FAHP-Express row (reference + comparisons) and the **attribute precision**.
4. Upload both workbooks and configure the criteria (function, `q`, `p`, scale).
5. Fix any problem listed by the validation panel; use the threshold assistant for `q` and `p` and check them against §4.3.
6. Run the deterministic E-PROM; inspect ranking, Φ per DM and PROMETHEE I relations.
7. Use **Explain** to understand the ranking and **Decision Makers** to check agreement — large disagreement is a signal to discuss before aggregating.
8. Test fragility: **Sensitivity** (weights), robustness checks, and the **Monte Carlo** (first-place frequency, rank acceptability, weight spread).
9. If the decision is a classification or a selection of several alternatives, use **Classification** or **Portfolio**.
10. Save the configuration, export the results and generate the PDF report (the ranking **together with** its stability).

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **TFN / trapezoid** | Fuzzy number `[a, b, c, d]` with support `[a, d]` and core `[b, c]`. |
| **COA** | Center of Area — defuzzification that returns the x-coordinate of the trapezoid's centroid. |
| **Φ⁺, Φ⁻, Φ** | Positive (leaving), negative (entering) and net PROMETHEE flows. |
| **Weight share** | Crisp weight of a criterion divided by the sum of all weights. |
| **Reference criterion** | The criterion each DM compares all others against in FAHP-Express (first column). |
| **Attribute precision** | High / Medium / Low confidence of a DM in the evaluations of one criterion. |
| **q / p** | Indifference / preference thresholds of the preference function. |
| **Rank acceptability** | Probability that an alternative occupies a given rank. |
| **FlowSort** | PROMETHEE-based sorting of alternatives into ordered classes delimited by limiting profiles. |
| **PROMETHEE V** | Selection of the best set of alternatives under constraints (budget, number, mandatory / excluded). |
| **Kendall's W** | Coefficient of concordance among the decision makers' rankings (0 = none, 1 = perfect). |
| **CPP** | Composition of Probabilistic Preferences: probabilistic treatment of preferences when inputs are distributions. |
