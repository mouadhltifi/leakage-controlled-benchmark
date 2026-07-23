# Census Double-Coding Adjudication — 10 Blind Codings vs CENSUS.md

**Date:** 2026-07-24. **Adjudicator protocol:** 10 independent blind codings (rubric-based, quotes attached, coders had no access to the repo/CENSUS.md/the KDD submission) compared cell-by-cell against `CENSUS.md` in this directory (frozen 2026-07-15). For every disagreement I re-fetched the primary paper text myself (arXiv/ar5iv HTML, verbatim-quote extraction) and resolved from the evidence. Scores: R = reported, P = partial, N = not reported.

**Blind-coding quality:** all 10 coders identified the correct papers, including the two traps the census itself documents — the Hu coder correctly identified arXiv 1712.02136 as *Listening to Chaotic Whispers* (not StockNet), and the BERTweet coder ran an explicit disambiguation sweep (SLOT/ECON/ALERTA-Net/MEANT/TEANet rejected) landing on Albada & Sonola 2503.10957. The census's citation note is independently validated.

---

## 1. Expected-disagreement classes (declared before counting)

- **Adv-ALSTM single-source C2 caveat:** the census scores C2 = R with the footnote that the text-availability half is **not applicable** (price-only system); the blind coder independently judged C2 "on the (explicit, dated, temporal) split alone" with the text half n/a. **Identical handling, identical score — this expected class produced zero disagreement.**
- **n/a handling generally:** no other n/a cells exist in either coding; no n/a-related divergence materialized anywhere.
- **Hedged agreements (frontier cells recognized as close on both sides):** StockNet C5 — blind agrees R but notes "if the census requires the justification to concern difficulty specifically, C5 could be argued down to P"; HoT C5 — blind agrees N but flags "borderline N vs P." Both match the census score; both confirm C5's boundary is the soft spot (see §5, rule B).

---

## 2. Full comparison table — 8 core systems × 5 dimensions (40 cells)

Format: census / blind. Disagreements **bold**.

| # | System | C1 | C2 | C3 | C4 | C5 | Agree |
|---|--------|----|----|----|----|----|:-:|
| 1 | StockNet | P / P | R / R | R / R | N / N | R / R | 5/5 |
| 2 | Adv-ALSTM | **P / R** | R¹ / R¹ | R / R | N / N | **R / P** | 3/5 |
| 3 | MSGCA | P / P | P / P | **R / P** | N / N | **R / P** | 3/5 |
| 4 | CAMEF | P / P | **N / P** | R / R | N / N | **R / P** | 3/5 |
| 5 | MSMF | P / P | N / N | N / N | N / N | N / N | 5/5 |
| 6 | CMTF | P / P | P / P | **R / P** | N / N | P / P | 4/5 |
| 7 | STONK | P / P | **P / R** | N / N | N / N | P / P | 4/5 |
| 8 | HoT | P / P | P / P | R / R | N / N | N / N | 5/5 |

**Agreement: 32 / 40 cells = 80%.** Column-wise: C1 7/8, C2 6/8, C3 6/8, C4 **8/8**, C5 5/8. C4 is unanimous across all coders and all systems (every blind note says "no test of any kind," matching the census); C5 carries the most interpretive spread.

### Bonus systems (2 × 5 = 10 cells, counted separately)

| # | System | C1 | C2 | C3 | C4 | C5 | Agree |
|---|--------|----|----|----|----|----|:-:|
| B1 | Hu et al. (WSDM 2018) | N / N | **P / R** | R / R | N / N | **N / P** | 3/5 |
| B2 | Albada & Sonola (BERTweet) | **N / P** | **P / R** | **R / P** | N / N | P / P | 2/5 |

Bonus agreement: 5/10. All five bonus disagreements adjudicated in §4.

---

## 3. Core disagreements — quotes, resolution, resolved score

### D1. Adv-ALSTM C1 — census P vs blind R → **resolved P** (census stands; closest call in the census)

- **Census:** "'inherits the optimal settings from ALSTM … selected via grid-search'; StockNet numbers 'directly copied' — no baseline-tuning parity."
- **Blind (R):** "Adv-ALSTM inherits the optimal settings from ALSTM, which are selected via grid-search" + baseline bullets "We tune three hyper-parameters U, T, lambda" (LSTM) / "we also tune U, T, and lambda" (ALSTM) — "explicit comparable tuning, hence R rather than P."
- **Primary text re-checked (ar5iv 1810.09936):** LSTM bullet: "We tune three hyper-parameters, number of hidden units (U), lag size (T), and weight of regularization term (λ)." ALSTM bullet: "Similar as LSTM, we also tune U, T, and λ." Parameter settings: "We search the optimal hyper-parameters of Adv-ALSTM on the validation set. For U, T, and λ, Adv-ALSTM inherits the optimal settings from ALSTM, which are selected via grid-search within the ranges of [4, 8, 16, 32], [2, 3, 4, 5, 10, 15], and [0.001, 0.01, 0.1, 1] … We further tune β and ϵ within [0.001, …, 1] and [0.001, …, 0.1]."
- **Resolution:** P. The paper reports genuine grid-search tuning of its price-only baselines — the strongest baseline-tuning reporting among the eight — but not the *comparable-tuning* property: (i) no parity statement, and budgets are unequal by construction (ALSTM's 4×6×4 = 96-point grid vs the challenger's inherited optimum plus an exclusive fresh 7×5 = 35-point search over its method knobs at comparison time); (ii) the split on which LSTM/ALSTM were tuned is never named (the "on the validation set" clause attaches to Adv-ALSTM's search) — "stated but incompletely" is the P definition. Under the paper's own operationalization of C1 (same calibration grid, same folds/seeds/budgets), inherit-then-extend is not comparable tuning and is not claimed to be.
- **Census change:** score stands; **evidence line must be rewritten** — the current line under-credits the LSTM/ALSTM grid search and leans on the "StockNet directly copied" fragment, which is irrelevant to C1 (StockNet is not price-only). Add a REBUTTAL-ARSENAL entry: this is the cell a hostile reviewer will push on for "none of the eight reports tuning its price-only baseline comparably."

### D2. Adv-ALSTM C5 — census R vs blind P → **resolved R** (census stands)

- **Census:** "'88 high-trade-volume-stocks in NASDAQ and NYSE markets' (ACL18)."
- **Blind (P):** same quote, but "'high-trade-volume' is a liquidity attribute inherited from the Xu & Cohen benchmark and merely descriptive here; no rationale … KDD17 described only as '50 stocks in U.S. markets'."
- **Resolution:** R under the rubric's first disjunct as the census consistently applies it (see rule B, §5): an explicitly stated liquidity attribute of the universe satisfies "deliberately liquid/efficient universe choice"; a justification is only required by the *second* disjunct (universe difficulty), otherwise the disjunction collapses. The blind coder's stricter justification-required reading would leave StockNet the only R and contradicts the census's own 4-R headline.
- **Census change:** none required; **optionally** note in the evidence line that KDD17 is uncharacterized ("50 stocks in U.S. markets") — a fair P-argument the census should pre-empt rather than hide.

### D3. MSGCA C3 — census R vs blind P → **resolved P — CENSUS MUST CHANGE**

- **Census (R):** "§5.1: 'chronologically partitioned … into training, **validation**, and testing subsets.'"
- **Blind (P):** "a validation subset is declared to exist but the paper never states it was used for hyperparameter or epoch selection; epochs fixed per dataset (200/250) with no rule; §V-G hyperparameter curves … do not name the evaluation split, so selection could be test-based — ambiguous."
- **Primary text re-checked (ar5iv 2406.06594):** the partition sentence exists; the extraction confirms "**no explicit sentences** about validation set usage or model checkpoint selection during training" anywhere; hyperparameters: "All hyperparameters of the baseline methods are set as the numbers reported in the original papers"; "The number of training epochs is 200 for BigData22 and CIKM18 and 250 for InnoStock and ACL18"; sensitivity: "With a large number of experiments, we set the learning rate lr=1e-4 as the best in MSGCA" — no split named.
- **Resolution:** P. The rubric reads "model selection insulated from the test set (**explicit validation set for selection/early stopping**)." Declaring that a validation subset *exists* does not state that selection used it. Every other census R on C3 rests on an explicit **use** statement (StockNet "tweaked on the development set"; Adv-ALSTM "search … on the validation set"; CAMEF "used for convergence checking and early stopping"; HoT "best model based on validation F1"; Hu "optimize the hyper-parameters and choose the best epoch"; Albada early stopping on validation loss). MSGCA was the census's only use-less R besides CMTF (D6). The demotion also *harmonizes* with §sec:msgca in the paper: the released code has no validation set at all and selects on test — "declares a partition, never states the selection protocol, code selects on test" is a cleaner and stronger story than "reports insulated selection."
- **Census change:** matrix cell R†→P†, per-paper line, Fully-reported 2→1, C3 headline (see §6).

### D4. MSGCA C5 — census R vs blind P → **resolved R** (census stands)

- **Census:** "the established datasets are '**high-trade-volume stocks** in US stock markets.'"
- **Blind (P):** "universe described … no justification of universe difficulty/efficiency anywhere (intro's 'very difficult task' is generic)"; also notes InnoStock is chosen for news-sparsity, not difficulty.
- **Resolution:** R — same rule as D2 (explicit liquidity attribute stated ⇒ first disjunct met). Optional evidence-line caveat: InnoStock (1 of 4 datasets) is a deliberately news-sparse, newly-listed-firm universe; the liquid characterization covers the three established benchmarks.

### D5. CAMEF C2 — census N vs blind P → **resolved N** (census stands; rubric line must be clarified)

- **Census (N):** "'divided into training, validation, and testing sets in a **6:2:2 ratio**' — no 'chronological/temporal' wording; no event-release-time / look-ahead statement."
- **Blind (P):** chronological half absent, but "the availability-timing half is stated by construction — event script used at release time i with pre-event segment X_{i−τ:i} to forecast X_{i+1:i+τ} (§3 Problem Formulation, §6.1.3 Test Settings). One half stated → P."
- **Primary text re-checked (arXiv 2502.04592v2):** §3: "the model uses both the event text ℰ_i and the historical time-series segment 𝒳_{i−τ:i}, which spans τ steps before the event's release at time i, to forecast the future time steps 𝒳_{i+1:i+τ}." §6.1.1: "divided … in a 6:2:2 ratio" with no ordering statement (three independent reads — census v2, blind v3, my fetch — concur).
- **Resolution:** N. The release-anchored clause is the *definition of event-driven forecasting* (there is no other way to pose the task), carries no leakage/availability framing, and appears nowhere near the evaluation-protocol text. Crediting task-formulation windows makes the availability half vacuous: MSGCA ("published time stamp t … predict t+1"), HoT (5-day lag window), Hu (t−N..t−1), Albada ([d−Δd, d−1]) would all be credited too — the blind coders for MSGCA and HoT themselves *refused* that credit, and accepting it would flip MSGCA's C2 to R outright. The consistent line (rule A, §5): the availability half requires an explicit availability/leakage-motivated lag or timestamp-alignment statement, which CAMEF does not have. What CAMEF *does* state is availability-consistency within a sample by construction; the leak our audit priced is across samples — the split — which is exactly the half CAMEF fails to report.
- **Census change:** score stands; **add one rubric-clarification sentence** (rule A) and optionally acknowledge the §3 clause in CAMEF's evidence line so the N is visibly reasoned, not an oversight.

### D6. CAMEF C5 — census R vs blind P → **resolved R** (census stands)

- **Census:** "'largest market impacts … in **major U.S. stock indexes and Treasury bonds** … we focused on … key stock indexes.'"
- **Blind (P):** "deliberately selected with a cited justification on event-impact grounds; liquidity/efficiency/difficulty of the universe never addressed."
- **Primary text re-checked:** "Studies … have demonstrated that the largest market impacts are typically observed in major U.S. stock indexes and Treasury bonds. Therefore, we focused on collecting high-frequency trading time-series data at 5 minute interval for key stock indexes, including the S&P 500 (SPX), Dow Industrial (INDU), NASDAQ (NDX), as well as U.S. Treasury Bond at 1-Month … and … 5-Year."
- **Resolution:** R. This is the strongest deliberate-choice statement among the four census R's: an explicit selection sentence ("Therefore, we focused on"), cited grounds, and a universe of major indexes/Treasuries — the canonical liquid/efficient instruments, named as such ("major"). First disjunct met; the blind coder's demand that the *grounds* be liquidity/difficulty again merges the two disjuncts (rule B).

### D7. CMTF C3 — census R vs blind P → **resolved P — CENSUS MUST CHANGE**

- **Census (R):** "'268 **validation days** (08/2022–09/2023)' distinct from test."
- **Blind (P):** "a named 268-day validation split exists and Optuna tuning is described, but the paper never states the tuning objective or epoch selection was evaluated on the validation set (no early-stopping/selection-criterion sentence) — ambiguous per rubric."
- **Primary text re-checked (arXiv 2504.13522v2):** split sentence: "the data is partitioned chronologically into distinct splits: 804 training days …, 268 validation days …, and 268 test days …". Optuna: "To optimize the Transformer model, we employ Optuna for efficient hyperparameter tuning, covering both the model architecture and the training parameters" — **no sentence names the split the search objective was evaluated on**; no early-stopping sentence; no other validation-use sentence exists.
- **Resolution:** P — identical defect class to D3: a declared validation split with no stated use for selection. Same rule, same outcome.
- **Census change:** matrix cell R→P, per-paper line, Fully-reported 1→0, C3 headline (see §6).

### D8. STONK C2 — census P vs blind R → **resolved R — CENSUS MUST CHANGE (headline-relevant)**

- **Census (P):** "§V-A: 'ordered chronologically, earliest 80% … training …' AND §III-A: 'introduced a one day time lag for all numerical columns except Open'; but news-publication-time lag not explicit → chronological + numeric-leakage guard, text-availability half not stated."
- **Blind (R):** "5-Fold TimeSeriesSplit primary; one-day lag ensures 'only historically available information was used'" — and, decisively, "text fused as **previous day's embedding** (Sec III-C1); LLM prompt uses 'news article from yesterday'."
- **Primary text re-checked (arXiv 2508.13327v1):** §III-A: "To prevent target leakage in our modeling process, we introduced a one day time lag for all numerical columns except Open, ensuring that only historically available information was used for prediction." §III-C1: "Simple concatenation is used to directly merge scaled numerical features with **the previous day's aggregated textual embedding**, producing a combined market vector for the day." LLM prompt: "Based on the numerical data and the news article from yesterday, determine if today's stock will move 'up' or 'down'…" §V-A: chronological 80/20 and "partitioned into five non-overlapping folds along the time axis"; random split explicitly flagged as leaky.
- **Resolution:** R — the census cell is wrong. The chronological half is undisputed. On the availability half, STONK does not merely define a lookback window: it *introduces* a one-day lag as a named leakage control with an availability rationale ("only historically available information"), and it states — in three separate places — that the **text** input for day t is day t−1's (fusion design, prompt, and the lag discipline the pipeline is built around). That is the same day-granularity availability discipline StockNet's trading-day alignment was credited R for; a rule that credits StockNet's [d_{t−1}, d_t) corpus interval but not STONK's stated previous-day text lag cannot be articulated without special pleading. Under rule A, STONK passes where CAMEF/HoT/Hu/Albada fail because the lag is a deliberate, leakage-motivated control with a stated rationale, not a task-formulation window (the Open exemption is availability-consistent for an at-open prediction and concerns a price feature, not the text/macro half the rubric names). The census keyed on §III-A's sentence scope ("numerical columns") and treated §III-C1 as architecture description; the blind coder read the pipeline as a whole and is right.
- **Census change:** matrix cell P→R, per-paper line, Fully-reported 0→1, C2 headline: availability-timing becomes **2/8 (StockNet, STONK)**; full-C2 becomes 2/8. **This kills the "availability-timing = StockNet only" headline** (§6, §7).

---

## 4. Bonus-system disagreements

### D9. Hu C2 — census P vs blind R → **resolved P** (census stands)

- **Census (P):** dated chronological split; "no news-availability-timing statement."
- **Blind (R):** "input news from t−N to t−1 … predicting yt on dt … news strictly precedes day-t open-to-open target" → both halves by construction.
- **Primary text re-checked (ar5iv 1712.02136):** "the goal is to use the news corpus sequence from time t−N to t−1 … to predict the class of Rise_Percent(t)" with Rise_Percent(t) = (Open(t+1)−Open(t))/Open(t); split "training set (66.7%) from September 2014 to May 2016 … test set (33.3%) from May 2016 to March 2017"; **no** alignment/availability sentence beyond the window.
- **Resolution:** P — rule A: a problem-statement lookback window is not an availability statement (same reasoning that keeps CAMEF at N and MSGCA/HoT/Albada at P; the blind coders for MSGCA and HoT applied exactly this rule). Census stands.

### D10. Hu C5 — census N vs blind P → **resolved N** (census stands)

- **Census (N):** "'2,527 stocks … vast majority of Chinese stocks' — no liquidity/selection justification."
- **Blind (P):** "universe described, no justification" — P on description alone.
- **Primary text re-checked:** "There are totally 2527 stocks, covering the vast majority of Chinese stocks." No filtering criteria (liquidity, volume, ST status, suspension) anywhere.
- **Resolution:** N. The dimension is a liquid/efficient-choice-or-difficulty-justification dimension, not a dataset-description dimension. An unfiltered whole-market universe is the *opposite* of a deliberately liquid selection, and nothing engages difficulty. Census P-class (CMTF, STONK, Albada) all name implicitly liquid/large-cap universes; "vast majority of Chinese stocks" does not.

### D11. Albada C1 — census N vs blind P → **resolved P — census (bonus row) should change**

- **Census (N):** "No explicit price-only baseline reported; no tuning-parity statement."
- **Blind (P):** "TECHNICALANALYST 54.96/0.016456 in 'Accuracy and MCC from selected papers on the Stocknet dataset' (Sec 6, Table 2)" — literature-copied price-only comparator on the identical benchmark.
- **Primary text re-checked (arXiv 2503.10957v1):** Table 2 caption "Accuracy and MCC from selected papers on the Stocknet dataset"; rows include TechnicalAnalyst (54.96 / 0.016456) among RAND, TSLDA, HAN, the Xu & Cohen analyst variants, HATS, GCN, Adversarial LSTM, MAN-SF. The authors trained no price-only model of their own.
- **Resolution:** P with an explicit "copied-row-only" note. The paper *does* place its headline number against a price-only comparator on the same dataset and split — the presence half of C1 — while running nothing price-only itself and stating nothing about tuning. Consistency forces this: HoT's census P rests on an ARIMA row whose values are likewise carried from the StockNet literature (the blind HoT coder noted the copied '-' cells). If HoT's copied ARIMA row is presence, Albada's copied TechnicalAnalyst row is presence. Bonus row only; "zero comparably tuned" is unaffected (P ≠ R), and B2's fully-reported count stays 1.
- **Census change:** B2 C1 N→P with "literature-copied comparator only; no self-run price-only baseline."

### D12. Albada C2 — census P vs blind R → **resolved P** (census stands)

- **Blind (R)** credited "tweets and prices in window [d−Δd, d−1]" as the availability half. Re-checked: "…using the market information M comprising relevant social media tweets and historical prices in the window [d−Δd, d−1]" — a formulation window inherited from StockNet, restated without any alignment/availability machinery of the paper's own. Rule A → P stands (identical to D9).

### D13. Albada C3 — census R vs blind P → **resolved R** (census stands; evidence line should be upgraded)

- **Census (R):** "validation set (8/1/2015–10/1/2015) distinct from test" (existence-only evidence).
- **Blind (P):** found the *stronger* evidence — "Early stopping was used to end trials early if validation loss did not decline" — but demoted for an audit-style suspicion: "the criterion for choosing among grid-search configs is unstated and the headline MCC 0.1114 is the best row of Table 1 … pattern consistent with test-set selection."
- **Primary text re-checked:** "Early stopping was used to end trials early if validation loss did not decline for four epochs." Config-level selection criterion indeed unstated.
- **Resolution:** R. The rubric's parenthetical is disjunctive — "explicit validation set for selection **/ early stopping**" — and validation-loss early stopping is explicitly stated; that is a stated use, unlike MSGCA/CMTF (D3/D7), where *no* use of any kind is stated. The blind coder's best-row concern is practice-inference, and this census counts reporting only. **Census change:** none to the score; replace the existence-only evidence quote with the early-stopping sentence (it is the actual basis for R and pre-empts the D3/D7 consistency question).

---

## 5. Adjudication rules distilled (add to CENSUS.md as rubric clarifications)

- **Rule A — C2 availability half.** Credited only for an explicit availability- or leakage-motivated lag/alignment statement (StockNet's trading-day alignment; STONK's one-day lag "ensuring that only historically available information was used" plus previous-day text fusion). Task-formulation lookback windows ("news from t−N to t−1," "window [d−Δd, d−1]," "event released at time i," "published time stamp t … predict t+1") do **not** count: every forecasting formulation places inputs before targets, so crediting them makes the half vacuous — and would flip MSGCA's C2 to R outright, which even MSGCA's own blind coder rejected.
- **Rule B — C5 first disjunct.** An explicitly stated liquidity/efficiency attribute of the universe ("high-trade-volume stocks," "major U.S. stock indexes and Treasury bonds" with a deliberate-selection sentence) satisfies "deliberately liquid/efficient universe choice"; a rationale is required only by the second disjunct (difficulty justification). Index membership alone (FTSE 100, S&P 500, "top 10 by size") = P; no liquidity-relevant characterization (bare counts, whole-market coverage) = N. Three blind coders independently applied a justification-required reading — evidence that the paper's Table-note gloss "C5 universe justification" invites the stricter reading the census does not use (see §8, paper edit 4).
- **Rule C — C3.** R requires a stated **use** of the validation set (selection and/or early stopping), not the declared existence of a partition. This is the rule already implicit in six of the census's eight R-grade evidence quotes; MSGCA and CMTF were the two cells graded R without it, and both demote to P.

---

## 6. Resolved census state (after applying the three core changes + one bonus change)

| # | System | C1 | C2 | C3 | C4 | C5 | Fully-reported |
|---|--------|----|----|----|----|----|:-:|
| 1 | StockNet | P | R | R | N | R | 3 |
| 2 | Adv-ALSTM | P | R¹ | R | N | R | 3¹ |
| 3 | MSGCA | P | P | **P**† | N | R | **1** |
| 4 | CAMEF | P | N | R | N | R | 2 |
| 5 | MSMF | P | N | N | N | N | 0 |
| 6 | CMTF | P | P | **P** | N | P | **0** |
| 7 | STONK | P | **R** | N | N | P | **1** |
| 8 | HoT | P | P | R | N | N | 1 |

Resolved headline block: C1 **0 R / 8 P** (unchanged); C2 full **2/8 (StockNet, STONK)**, chronological 6/8 (unchanged), availability half **2/8 (StockNet, STONK)**; C3 **4 R (StockNet, Adv-ALSTM, CAMEF, HoT), 2 P (MSGCA, CMTF), 2 N (MSMF, STONK)**; C4 0/8 (unchanged, unanimous); C5 4 R / 2 P / 2 N (unchanged). **Conjunction m = 0/8 unchanged; best reporters unchanged (StockNet, Adv-ALSTM at 3); no paper reaches 4.**

---

## 7. Do the paper's headline counts survive?

| Headline claim | Verdict |
|---|---|
| **0/8 report all five dimensions** (m = 0; "the strongest reach three") | **SURVIVES.** C4 = N is unanimous across 10/10 blind codings; after all resolutions the maximum fully-reported count is still 3 (StockNet, Adv-ALSTM). No resolution combination reaches 4. |
| **C4: 0/8 report ANY significance test** | **SURVIVES.** 8/8 core (and 2/2 bonus) blind codings score N with "no test of any kind" notes; MSGCA's ±0.0037 seed spread is classified a spread, not a test, by both sides. The single strongest-agreement dimension in the exercise. |
| **C1 comparable-tuning: 0/8** | **SURVIVES, with a documented near-miss.** Adv-ALSTM adjudicated P (D1) — but it is the one cell a competent reviewer can argue to R (its blind coder did). Evidence line must be upgraded and a rebuttal prepared; the claim's wording ("none reports tuning its price-only baseline comparably") remains defensible because no parity of budget/grid is stated and the challenger keeps an exclusive method-knob search. |
| **Availability-timing: StockNet only (1/8)** | **DOES NOT SURVIVE.** STONK C2 resolves to R (D8): a stated one-day lag as a named leakage control with an availability rationale, plus previous-day text fusion stated in three places. Count becomes **2/8 (StockNet, STONK)**. Main.tex and the census must be edited (§8). Bonus rows stay non-availability-timed, so "neither availability-timed" for B1/B2 survives, but "StockNet remains the census's only availability-timed system" does not. |

Derived paper-text casualty (not one of the four registered headlines, but printed): "A chronological split and an insulated validation set are commonly reported, **six of eight each**" (main.tex:221–222) — chronological stays 6/8; insulated validation drops to **4/8** under rule C (D3, D7).

---

## 8. Action list

### Census changes needed (kdd-artifact/audits/census/CENSUS.md — author's call, artifact is frozen-but-unshipped until Jul 26)

1. **MSGCA C3: R → P.** Matrix row 3, per-paper line ("declares a chronological train/validation/test partition; never states the validation set's use for selection or early stopping"), Fully-reported 2→1. Keep/extend the reported-vs-released dagger.
2. **CMTF C3: R → P.** Matrix row 6, per-paper line ("268 validation days declared; Optuna objective split and any selection/early-stopping rule unstated"), Fully-reported 1→0.
3. **STONK C2: P → R.** Matrix row 7, per-paper line quoting §III-A ("one day time lag … ensuring that only historically available information") **and** §III-C1 ("previous day's aggregated textual embedding"), Fully-reported 0→1.
4. **Headline block:** C2 line → full 2/8 and availability half 2/8 (StockNet, STONK); C3 line → 4 R / 2 P / 2 N; "How to read the columns" → availability half "only StockNet and STONK"; C3 removed from the commonly-reported pair or requalified.
5. **Rubric clarifications:** add rules A, B, C from §5 as three sentences under the dimension definitions.
6. **Evidence-line upgrades (no score change):** Adv-ALSTM C1 rewritten per D1 (acknowledge LSTM/ALSTM grid search; drop the StockNet-copied fragment); Albada C3 → quote the early-stopping sentence; optional caveats: Adv-ALSTM C5 (KDD17 uncharacterized), MSGCA C5 (InnoStock news-sparse), CAMEF C2 (acknowledge the §3 release-anchored clause and why it is not credited).
7. **Bonus B2 C1: N → P** (copied TechnicalAnalyst row; consistency with HoT's copied ARIMA row).

### Paper edits implied (thesis/kdd/main.tex)

1. **:221–222** "six of eight each" → chronological six of eight, insulated validation **four** of eight.
2. **:223–227** "only one reports availability-timing of its text features" → "only two report availability-timing of their text features" (and the three-links sentence adjusts: the availability link now breaks for six of eight rather than seven).
3. **:230–232** "(StockNet remains the census's only availability-timed system)" → reword; among all ten censused systems the availability-timed set is {StockNet, STONK}.
4. **Table `tab:census`:** MSGCA C3 R†→P†; CMTF C3 R→P; STONK C2 P→R. Table note: "only StockNet's R includes it" → "only StockNet's and STONK's R include it"; consider re-glossing "C5 universe justification" → "C5 universe liquidity stated or difficulty justified" (three blind coders misread the current gloss — a reviewer will too).
5. **† footnote** for MSGCA: adjust to "a train/validation/test partition is declared (its use for selection is not stated); the released code splits randomly and selection reads the test set" — strictly stronger for the paper's reported-vs-practice point.
6. No change needed in §sec:msgca (it describes code behavior only) or to m = 0 / "strongest reach three" / C1 / C4 sentences.

---

## 9. Bottom line

32/40 core-cell agreement (80%) between the frozen census and 10 independent blind coders; C4 unanimous. Of 8 core disagreements, 5 resolve for the census (including the two headline-critical defenses: Adv-ALSTM C1 stays P, CAMEF C2 stays N) and 3 against it (MSGCA C3 → P, CMTF C3 → P, STONK C2 → R). Three of the four registered headline counts survive outright; the fourth — availability-timing "StockNet only" — becomes "StockNet and STONK (2/8)" and must be edited in both CENSUS.md and main.tex before the Jul 26 gate. The census's core claim — zero of eight report all five dimensions, binding constraint C4 plus the comparable-tuning half of C1 — is confirmed and, after the C3 demotions, slightly *strengthened* (the modal reporter now fully reports fewer dimensions, and the fully-reported column drops for two systems while rising for one).
