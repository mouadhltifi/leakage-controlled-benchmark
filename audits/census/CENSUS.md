# Reporting Census — Five Reporting Dimensions in Multi-Source Stock Prediction

**What this is:** A census of what each PAPER REPORTS in its own text (Methods/Experiments/Setup), verified against the primary source. It is **not** a census of what the code does. Our own audit findings (e.g. the MSGCA validation-selection inflation, the CAMEF epoch mismatch) are **excluded** — this counts only reporting.

**The five reporting dimensions** (in the benchmark's protocol, C1–C4 are
the four enforced safeguards and C5, universe justification, is a scope
condition; the census scores all five as reporting dimensions)
- **C1** — a price-only baseline *tuned comparably* to the proposed model.
- **C2** — chronological train/test split **AND** availability-timed features (text/macro lagged to availability).
- **C3** — model selection insulated from the test set (explicit validation set for selection/early stopping).
- **C4** — multiplicity-corrected significance (any correction across configurations), at any unit.
- **C5** — a deliberately liquid/efficient universe choice (or any justification of universe difficulty).

**Scoring key:** **R** = REPORTED (clear statement of the full control) · **P** = PARTIAL (one half present / stated but incompletely) · **N** = NOT REPORTED (absent from the text). Evidence phrases are ≤15-word quotes or paraphrases from the paper.

> **Citation note:** arXiv **1712.02136 is NOT StockNet.** It is Hu et al., *"Listening to Chaotic Whispers"* (WSDM 2018), a Chinese-news system (2,527 stocks, News-RNN baselines). The real StockNet (Xu & Cohen, ACL 2018) has no canonical arXiv id; it is ACL Anthology **P18-1183**. This census uses the genuine StockNet paper. This census cites the genuine StockNet paper.

---

## Status matrix (the 8 requested systems)

| # | System (paper) | Multi-source? | C1 | C2 | C3 | C4 | C5 | Fully-reported |
|---|----------------|:---:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | **StockNet** — Xu & Cohen, ACL 2018 | yes (tweets+price) | P | **R** | **R** | N | **R** | 3 |
| 2 | **Adv-ALSTM** — Feng et al., IJCAI 2019 | no (price only)¹ | P | **R**¹ | **R** | N | **R** | 3¹ |
| 3 | **MSGCA** — Zong & Zhou 2024 | yes (indicators+text+…) | P | P | **R** | N | **R** | 2 |
| 4 | **CAMEF** — Zhang et al., KDD 2025 | yes (macro text+TS) | P | **N** | **R** | N | **R** | 2 |
| 5 | **MSMF** — Qin 2024 | yes (TS+image+text) | P | **N** | **N** | N | **N** | 0 |
| 6 | **CMTF** — Pei et al. 2025 | yes (price+news+reports) | P | P | **R** | N | P | 1 |
| 7 | **STONK** — Khanna et al. 2025 | yes (numeric+sentiment) | P | P | **N** | N | P | 0 |
| 8 | **Higher-Order-Transformers** — Omranpour et al. 2024² | yes (tweets+price) | P | P | **R** | N | **N** | 1 |

¹ Adv-ALSTM is **single-source** (price + adversarial training; no text/macro). Its temporal split is explicit, so C2's *chronological* half is met and the *text-availability* half is not applicable (no text to leak). Counted R with that caveat.
² Substitute for TEANet — see "Unverified" below.

**How to read the columns:** C3 (validation set) and the *chronological* half of C2 are commonly reported; the missing links that break the conjunction are **C4** (nobody), the *comparable-baseline-tuning* half of **C1** (nobody), and the *text/macro availability-timing* half of **C2** (only StockNet).

---

## Headline counts (N = 8 verified)

- **C1 (price-only baseline tuned comparably): 0 REPORTED.** A price-only/technical-only baseline is *present* in **8/8**, but **0/8** state that baselines were tuned comparably to the proposed model → all 8 scored **P**. MSGCA states the opposite explicitly ("baseline hyperparameters set as the numbers reported in the original papers").
- **C2 (chronological split AND availability-timed features): 1 REPORTED in full** (StockNet). Breaking the conjunction apart: a **chronological split is reported by 6/8** (all but CAMEF and MSMF); the **text/macro availability-timing half is reported by only 1/8** (StockNet's trading-day alignment). Adv-ALSTM reports the split but is single-source.
- **C3 (validation set insulated from test): 6 REPORTED** (StockNet, Adv-ALSTM, MSGCA, CAMEF, CMTF, HoT); **2 NOT** (MSMF, STONK).
- **C4 (multiplicity-corrected significance): 0 REPORTED.** In fact **0/8 report any significance test at all** (no t-test, McNemar, CI, or bootstrap), so multiplicity correction is moot. The strongest quantification of run-to-run variability among the eight is MSGCA's, which reports a per-seed spread only — "0.1112 ± 0.0037" over five seeds (Table IV), a standard deviation, not a significance test. No other system reports even that.
- **C5 (liquid/efficient universe justification): 4 REPORTED** (StockNet, Adv-ALSTM, MSGCA, CAMEF), **2 PARTIAL** (CMTF, STONK), **2 NOT** (MSMF, HoT).

### The conjunction

**Papers reporting all five jointly: m = 0 / 8.** The binding constraint is C4 (0/8 report significance testing) and the comparable-tuning half of C1 (0/8). No relaxation of the other controls changes m.

**Best reporters: StockNet and Adv-ALSTM at 3 of 5 fully reported** (C2, C3, C5) + partial C1, both missing C4. **No paper reports 4 or 5 controls fully.** → The census supports the claim; nothing found weakens it.

---

## Per-paper evidence

### 1. StockNet — Xu & Cohen, *Stock Movement Prediction from Tweets and Historical Prices*, ACL 2018
Dataset: StockNet (tweets + Yahoo prices), **88 stocks**, 01/01/2014–01/01/2016.
- **C1 — P.** §6.3: "ARIMA … an advanced technical analysis method **using only price signals**" (price-only baseline present); no statement baselines were tuned comparably.
- **C2 — R.** §3: "We split them **temporally** … 20,339 … 01/01/2014–01/08/2015 for training … 3,720 … 01/10/2015–01/01/2016 for test." §4 **trading-day alignment**: corpus in [d_{t-1},d_t), prices on d_{t-1}, to predict d_t. Both halves met.
- **C3 — R.** §3/§6.1: "2,555 movements … 01/08/2015–01/10/2015 are for development" and "hyper-parameters are **tweaked on the development set**."
- **C4 — N.** §6.2: metrics are accuracy and MCC only; no significance test.
- **C5 — R.** §3: "Since **high-trade-volume-stocks tend to be discussed more on Twitter**, we select … 88 stocks … top 10 in capital size."

### 2. Adv-ALSTM — Feng et al., *Enhancing Stock Movement Prediction with Adversarial Training*, IJCAI 2019
Datasets: ACL18 (88 stocks, 2014–2016), KDD17 (50 stocks, 2007–2017). **Single-source (price only).**
- **C1 — P.** Baselines MOM, MR, LSTM, ALSTM, StockNet; "inherits the optimal settings from ALSTM … selected via grid-search"; StockNet numbers "directly copied" — no baseline-tuning parity.
- **C2 — R (single-source caveat).** "We **temporally split** … training (Jan-01-2014 to Aug-01-2015) … testing (Oct-01-2015 to Jan-01-2016)." No text/macro modality to time; price features are lagged MAs.
- **C3 — R.** "We search the optimal hyper-parameters … on the **validation set**" (validation distinct from test in the temporal split).
- **C4 — N.** No t-test / Bonferroni / Holm / FDR reported.
- **C5 — R.** "88 **high-trade-volume-stocks** in NASDAQ and NYSE markets" (ACL18).

### 3. MSGCA — Zong & Zhou 2024, *Stock Movement Prediction with Multimodal Stable Fusion via Gated Cross-Attention* (arXiv 2406.06594; Springer Complex & Intelligent Systems 2025)
Datasets: InnoStock (369), BigData22 (50), ACL18 (87), CIKM18 (38).
- **C1 — P.** §5.2: "**Indicator-only methods** … LSTM and ALSTM" (price-only present); §5.1: "All hyperparameters of the baseline methods are set as the numbers reported in the original papers" — explicitly **not** comparable tuning.
- **C2 — P.** §5.1: "We **chronologically partitioned** each dataset into training, validation, and testing subsets" (chronological half met); **no** statement lagging text/news to availability.
- **C3 — R.** §5.1: "chronologically partitioned … into training, **validation**, and testing subsets."
- **C4 — N.** No multiplicity correction / significance test.
- **C5 — R.** §5.1: the established datasets are "**high-trade-volume stocks** in US stock markets."

### 4. CAMEF — Zhang et al., KDD 2025 (arXiv 2502.04592)
Dataset: 6 macro-release types 2008–Apr 2024; instruments S&P 500, Dow, NASDAQ, US Treasury 1M/5Y.
- **C1 — P.** Baselines ARIMA, DLinear, AutoFormer, FEDformer, iTransformer, PatchTST, TEST, GPT4MTS (TS-only present); "All models were trained for 10 epochs, batch size 32" (uniform protocol, not per-model tuning parity).
- **C2 — N.** Verbatim: "The dataset was divided into training, validation, and testing sets in a **6:2:2 ratio**" — **no** "chronological/temporal" wording; no event-release-time / look-ahead statement.
- **C3 — R.** "The **validation set** was used for convergence checking and **early stopping** to prevent overfitting."
- **C4 — N.** No correction across the ~30 model×asset×horizon configurations.
- **C5 — R.** "largest market impacts … in **major U.S. stock indexes and Treasury bonds** … we focused on … key stock indexes."

### 5. MSMF — Qin 2024, *Multi-Scale Multi-Modal Fusion for Enhanced Stock Market Prediction* (arXiv 2409.07855)
**Read in full from the PDF (15 pp).** §4 "Result and Analysis" jumps from the loss function straight to result Tables 1–7.
- **C1 — P.** Baselines LSTM/GRU/Transformer (return) and SVM/RF/XGBoost (movement) — technical-only present; **no** tuning statement and **no** description of the data they ran on.
- **C2 — N.** **No split described anywhere** — no dates, no ratio, no chronological/temporal/random statement.
- **C3 — N.** No validation set described (only ablation "experiments to validate modules").
- **C4 — N.** "significant" appears only rhetorically in the abstract/conclusion; no test.
- **C5 — N.** **No dataset named, no stock count, no date range, no universe** anywhere in the paper.
- *Note:* the weakest reporter in the census — near-total absence of experimental methodology.

### 6. CMTF — Pei et al. 2025, *Cross-Modal Temporal Fusion for Financial Market Forecasting* (arXiv 2504.13522; IOS Press FAIA)
Dataset: **5 FTSE 100 stocks** (Shell, Unilever, BAT, BP, Diageo), 02/04/2019–05/22/2024 (1360 days).
- **C1 — P.** Baselines Zero-Change, Linear Regression, ARIMA, RF, SVR, LSTM, Transformer (price-only present); per-baseline configs given but no parity with CMTF's "auto-training pipeline."
- **C2 — P.** "partitioned **chronologically** … 804 training days (02/2019–07/2022) … 268 test days (10/2023–05/2024)" (chronological half met); **no** news/sentiment/report availability-lag statement.
- **C3 — R.** "268 **validation days** (08/2022–09/2023)" distinct from test.
- **C4 — N.** No Bonferroni/Holm/FDR across ablations or baselines.
- **C5 — P.** "five **representative** UK-headquartered multinational corporations listed in the **FTSE 100**" — liquid blue-chips named, but no explicit liquidity/efficiency/difficulty justification.

### 7. STONK — Khanna et al. 2025, *Towards Unified Multimodal Financial Forecasting* (arXiv 2508.13327)
Data: FinSen (+ Financial PhraseBank, FiQA); "S&P 500 financial articles 2007–2023"; benchmarks 5 encoders (FinBERT, ModernBERT, Electra, DeBERTa, MiniLM).
- **C1 — P.** §V-B1: "Logistic Regression … on the **numerical features** without … sentiment" (numeric-only present); no tuning-parity statement.
- **C2 — P.** §V-A: "ordered **chronologically**, earliest 80% … training … recent 20% … testing" **AND** §III-A: "introduced a **one day time lag** for all numerical columns except Open"; but news-publication-time lag not explicit → chronological + numeric-leakage guard, text-availability half not stated.
- **C3 — N.** No separate validation set for selection/early stopping mentioned.
- **C4 — N.** No significance test / correction.
- **C5 — P.** "**S&P 500** financial articles from 2007 to 2023" — liquid index named, but no deliberate stock-universe/liquidity justification.

### 8. Higher-Order-Transformers — Omranpour, Rabusseau & Rabbany 2024, *Higher Order Transformers: Enhancing Stock Movement Prediction on Multimodal Time-Series Data* (arXiv 2412.10540)
Dataset: StockNet (88 stocks, 2014–2016). *(Readable StockNet-descendant with tweets, used in place of TEANet.)*
- **C1 — P.** "ARIMA … an advanced technical analysis method **using only price signals**" (price-only present); "grid search" reported only for their own model.
- **C2 — P.** "**temporally divided** in a 70:10:20 ratio … 01/01/2014 … training … 01/10/2015 … testing" (chronological half met); tweet-availability lag not stated.
- **C3 — R.** "70:10:20 ratio for train, **validation**, and test splits."
- **C4 — N.** No p-values, CIs, or significance tests anywhere.
- **C5 — N.** "88 stocks extracted from Yahoo Finance" — no selection/liquidity justification (the underlying StockNet universe is high-trade-volume, but the paper does not restate it).

---

## Additional StockNet-family rows (beyond the requested 8 — reinforce the pattern)

### B1. Hu et al. — *Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction*, WSDM 2018 (arXiv **1712.02136**)
Data: Chinese stocks, **2,527 stocks**, 425,250 news articles, 09/2014–03/2017.
- **C1 — N.** Baselines are news-oriented RNNs + RandomForest/MLP; **no price-only baseline**; validation used only for its own hyperparameters.
- **C2 — P.** "training (66.7%) … 09/2014–05/2016 … test (33.3%) … 05/2016–03/2017" (chronological half met); no news-availability-timing statement.
- **C3 — R.** "randomly sample a **validation set** from the training set (10%) … optimize hyper-parameters and choose the best epoch."
- **C4 — N.** No significance test.
- **C5 — N.** "2,527 stocks … vast majority of Chinese stocks" — no liquidity/selection justification.
- Fully reported: **1** (C3).

### B2. Albada & Sonola — *Predicting Stock Movement with BERTweet and Transformers*, 2025 (arXiv 2503.10957)
Data: StockNet, 78 stocks (8 conglomerate + top-10 by size in 8 sectors), 2014–2016.
- **C1 — N.** No explicit price-only baseline reported; no tuning-parity statement.
- **C2 — P.** "training … 1/1/2014–8/1/2015 … validation 8/1/2015–10/1/2015 … test 10/1/2015–1/1/2016" (chronological half met); tweet-availability lag not stated.
- **C3 — R.** validation set (8/1/2015–10/1/2015) distinct from test.
- **C4 — N.** No t-test / Bonferroni / Holm / FDR.
- **C5 — P.** "top 10 stocks in size" (implicitly large-cap) — no formal liquidity/difficulty justification.
- Fully reported: **1** (C3).

*Both bonus rows reinforce m = 0: neither exceeds 1 fully-reported control.*

---

## Unverified / substituted

- **TEANet** (Zhang et al. 2022, *Transformer-based attention network for stock movement prediction*, Expert Systems with Applications, DOI 10.1016/j.eswa.2022.117239). **Methodology not accessible** — full text is paywalled (ScienceDirect / Wiley mirror), no arXiv preprint, abstracts/ResearchGate do not expose the experimental setup. **Not scored.** Substituted with Higher-Order-Transformers (system 8), a readable StockNet-descendant with tweets. If TEANet must be scored, obtain the ESWA PDF via institutional access.
- No other requested system is unverified: systems 1–8 were all read from primary text (StockNet from the ACL PDF; MSMF from the arXiv PDF after HTML rendering failed; the rest from arXiv HTML / ar5iv).

---

## Source URLs (primary text used)

| System | Primary source read | Also |
|--------|--------------------|------|
| StockNet | https://homepages.inf.ed.ac.uk/scohen/acl18stock.pdf | ACL Anthology https://aclanthology.org/P18-1183/ |
| Adv-ALSTM | https://ar5iv.labs.arxiv.org/html/1810.09936 | https://arxiv.org/abs/1810.09936 (IJCAI 2019) |
| MSGCA | https://ar5iv.labs.arxiv.org/html/2406.06594 | https://link.springer.com/article/10.1007/s40747-025-02023-3 |
| CAMEF | https://arxiv.org/html/2502.04592v2 | https://arxiv.org/abs/2502.04592 · https://dl.acm.org/doi/10.1145/3711896.3736872 |
| MSMF | https://arxiv.org/pdf/2409.07855 (PDF, read in full) | https://arxiv.org/abs/2409.07855 |
| CMTF | https://arxiv.org/html/2504.13522v2 | https://arxiv.org/abs/2504.13522 · IOS Press https://ebooks.iospress.nl/DOI/10.3233/FAIA251474 |
| STONK | https://arxiv.org/html/2508.13327v1 | https://arxiv.org/abs/2508.13327 |
| Higher-Order-Transformers | https://arxiv.org/html/2412.10540v1 | https://arxiv.org/abs/2412.10540 |
| Hu et al. (bonus) | https://ar5iv.labs.arxiv.org/html/1712.02136 | https://arxiv.org/abs/1712.02136 (WSDM 2018) |
| BERTweet (bonus) | https://arxiv.org/html/2503.10957v1 | https://arxiv.org/abs/2503.10957 |
| TEANet (unverified) | paywalled | https://doi.org/10.1016/j.eswa.2022.117239 |

*Compiled 2026-07-15. Reporting census only — verifiable from each paper's own text; excludes our audit findings.*
