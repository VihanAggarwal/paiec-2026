# Local CV results

## Codabench (formative; one random draw each, sd about 0.05 per draw)

| date (ET) | submission ID | zip | score (Brier ALC) | notes |
|---|---|---|---|---|
| 2026-11-07 03:06 | 1039604 | v9.zip | (running) | in-context Sonnet judge |
| 2026-11-07 02:44 | 1039546 | v8.zip (fixed) | (running) | first build with all features active on the platform |
| 2026-11-07 00:43 | 1039269 | v7.zip (pre-fix) | 0.184850 | pre-fix build: judge and embeddings off on the platform |
| 2026-11-07 00:42 | 1039265 | v7.zip (pre-fix) | 0.199394 | pre-fix build |
| 2026-11-07 00:42 | 1039263 | v7.zip (pre-fix) | 0.182815 | pre-fix build |
| 2026-11-07 00:33 | 1039233 | v7.zip (pre-fix) | 0.199091 | pre-fix build |
| 2026-11-07 00:25 | 1039214 | v7.zip (pre-fix) | 0.163862 | Haiku judge; unknown whether the key file was added |
| 2026-11-06 23:34 | 1039074 | v6.zip | 0.165113 | same zip as 1039042: draw-to-draw spread 0.149 vs 0.165 |
| 2026-11-06 23:14 | 1039042 | v6.zip | 0.149149 | ECE 0.138556; on the public leaderboard at rank 24 (2026-11-06 23:26 ET) |
| 2026-11-06 22:41 | 1038991 | v5.zip | 0.154263 | final pick so far; different random draw than v4b, so not a paired comparison |
| 2026-11-06 22:19 | 1038965 | v4b.zip | 0.183503 | first scored entry; public top 50 ended at 0.1748 at the time |
| 2026-11-06 21:08 | 1038828 | v0.zip | (no score shown) | baseline, Finished |

## Local cross-validation

All numbers come from `eval/simulate.py` runs (files in `eval/results/`). Default protocol: 7-fold
leave-benchmarks-out (seed 0), up to 4 pairs per held-out benchmark (541 pairs / 151 benchmarks),
per-pair 50/50 acquisition/evaluation split, random acquisition, budgets 0/1/3/7/15/31, `labeled`
holds all pairs of the snapshot. ALC = 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31, mean over pairs.

| version | split | ALC | se | B0 | B1 | B3 | B7 | B15 | B31 | ECE_ALC | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| const 0.5 | pair | 0.2500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2200 | sanity check |
| v0 | pair | 0.2426 | 0.0042 | 0.2500 | 0.3595 | 0.2351 | 0.2059 | 0.1938 | 0.1875 | 0.1804 | baseline empirical mean |
| v1 | pair | 0.2518 | 0.0010 | 0.2518 | 0.2518 | 0.2518 | 0.2518 | 0.2518 | 0.2518 | 0.2246 | prior only: mu0 + name/provider subject prior, labels unused |
| v2 | pair | 0.2123 | 0.0028 | 0.2522 | 0.2272 | 0.2126 | 0.2021 | 0.1966 | 0.1936 | 0.1403 | + benchmark offset from ALL same-benchmark labels (GP, Laplace) |
| v3 | pair | 0.2089 | 0.0029 | 0.2518 | 0.2259 | 0.2097 | 0.1981 | 0.1915 | 0.1870 | 0.1309 | + pair offset (tau_p) |
| v4 | pair | 0.2071 | 0.0027 | 0.2502 | 0.2250 | 0.2085 | 0.1968 | 0.1890 | 0.1825 | 0.1373 | + item_features and TF-IDF text kernels; no subject prior; tau_b 2.2 |
| v4b | pair | 0.2071 | 0.0027 | 0.2502 | 0.2250 | 0.2085 | 0.1968 | 0.1890 | 0.1825 | 0.1373 | v4 + pure-Python fallback (identical predictions); SUBMITTED |
| eb | pair | 0.2089 | 0.0026 | 0.2508 | 0.2269 | 0.2091 | 0.1979 | 0.1915 | 0.1878 | 0.1317 | empirical-Bayes prior over pair accuracy, peer weight 0.3 (src_eb) |
| eb | pair | 0.2108 | 0.0026 | 0.2508 | 0.2313 | 0.2123 | 0.1994 | 0.1920 | 0.1870 | 0.1355 | empirical Bayes, own labels only |
| v4 | benchmark | 0.2077 | 0.0027 | 0.2502 | 0.2268 | 0.2083 | 0.1971 | 0.1891 | 0.1836 | 0.1350 | robustness: per-benchmark split |
| v3 | benchmark | 0.2093 | 0.0028 | 0.2518 | 0.2276 | 0.2091 | 0.1986 | 0.1913 | 0.1877 | 0.1289 | robustness: per-benchmark split |
| v0 | benchmark | 0.2443 | 0.0042 | 0.2500 | 0.3634 | 0.2362 | 0.2086 | 0.1942 | 0.1884 | 0.1830 | robustness: per-benchmark split |
| const 0.5 | benchmark | 0.2500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2187 | sanity |
| v4 | pair | 0.2085 | 0.0027 | 0.2504 | 0.2292 | 0.2088 | 0.1980 | 0.1898 | 0.1831 | 0.1413 | robustness: fold seed 1 |
| v3 | pair | 0.2105 | 0.0029 | 0.2525 | 0.2306 | 0.2104 | 0.1997 | 0.1921 | 0.1873 | 0.1344 | robustness: fold seed 1 |
| v5 | pair | 0.2085 | 0.0028 | 0.2589 | 0.2267 | 0.2091 | 0.1968 | 0.1890 | 0.1825 | 0.1402 | v4b + prior mean shifted by -0.8 logit (hard-regime prior); FINAL PICK |
| v5 | benchmark | 0.2090 | 0.0028 | 0.2592 | 0.2289 | 0.2086 | 0.1972 | 0.1891 | 0.1836 | 0.1378 | robustness: per-benchmark split |
| v6 | pair | 0.2075 | 0.0028 | 0.2587 | 0.2266 | 0.2088 | 0.1956 | 0.1872 | 0.1798 | 0.1462 | v5 + within-benchmark semantic kernel (MiniLM, tau_m 1.0), tau_t 0.5; bundled numpy encoder |
| v6 | benchmark | 0.2083 | 0.0028 | 0.2589 | 0.2290 | 0.2082 | 0.1959 | 0.1879 | 0.1818 | 0.1434 | robustness: per-benchmark split |
| v7 | pair | 0.2026 | 0.0028 | 0.2495 | 0.2224 | 0.2035 | 0.1908 | 0.1828 | 0.1775 | 0.1529 | v6 + Haiku 5.5 difficulty judge (prior-mean slope x2, within-benchmark kernel tau_j 0.6); run tune_judge1 |

## Findings (2026-11-07, all from local runs)
- Name-based subject ability does not transfer across benchmarks: correlation between a name's
  leave-benchmark-out mean residual and its residual on a new benchmark is -0.065 (3,752 pairs,
  103 benchmarks). v1's subject prior is therefore slightly worse than 0.5 and is dropped in v4.
- Leaderboard noise: with the simulator's snapshots (3.4 pairs each, as a formative evaluation
  samples <= 1000 subject-item pairs), v3's per-snapshot ALC has sd 0.049; the expected best of
  10 submissions of the same zip is 0.123 although its mean is 0.211.
- Oracle floors on 2,315 training pairs: knowing each pair's true accuracy gives Brier 0.183;
  additionally knowing each item's difficulty from other subjects gives 0.121.
- Tuning sweeps around v4 (eval/results/tune_g2.csv, tune_g3.csv): calibration sharpen 0.9-1.3 and
  probit scale 0.6-1.0, tau_f / tau_t in {1.0, 1.6}, exact-item tau_e in {0, 0.7}: every variant is
  within +0.0008 of v4 (0.2071); none is better. v4 keeps tau_b 2.2, tau_p 1.0, tau_f = tau_t = 1.
- Amortized item difficulty from hashed item text/features (ridge, leave-benchmarks-out, features
  present in >= 15 training benchmarks): within-benchmark correlation with true item difficulty
  0.03-0.05, benchmark-mean correlation 0.18-0.21. Too weak to use; not in any submission.
- Evaluator review (3 independent reviewers + adversarial verification): no confirmed defects;
  noted that the per-pair split hash makes each pair's split equal to the benchmark split or its
  complement, 3/159 snapshots exceed the 1000 cap by <= 95 items, and 9 benchmarks with
  non-"binary" response_type but 0/1 responses are included; each changes ALC by < 0.0002.
- Prior-mean shift (tune_g4, tune_g5; eval/regime_reweight.py). Public entries report that hidden
  test pairs are hard (nguefackuriel/paiec-gpirt: mean accuracy 0.35 in formative draws; paiksca and
  doctawho42 ship level-prior means of -1.0 and -2.5). Reweighting our 541 CV pairs to mean accuracy
  0.35 (w = exp(-lam * base_rate), ESS 416), ALC for mu0 shift 0 / -0.3 / -0.6 / -0.8 / -1.0 / -1.4:
  unweighted 0.2071 / 0.2073 / 0.2079 / 0.2085 / 0.2092 / 0.2112; hard regime 0.1922 / 0.1906 /
  0.1893 / 0.1887 / 0.1882 / 0.1878. v5 uses -0.8: +0.0014 if the hidden pool is like training,
  -0.0035 if it is hard; break-even at P(hard) = 0.29. Same under the per-benchmark split
  (0.2090 vs 0.2077 unweighted; 0.1885 vs 0.1920 hard).
- Response-weighted training pairs have mean accuracy 0.354 but an effective sample of only 6 of
  the 541 CV pairs (item-weighted: 0.49), so size-weighted sampling is NOT used as evidence.
- Formative-shaped snapshots (PAIEC_MAX_CHUNK=1 / 2 pairs per benchmark): v4 0.2099 / 0.2106,
  v5-like (-0.8) 0.2110 / 0.2119; the -0.8 cost in the training regime is the same (+0.0011-0.0013).
- Sentence embeddings (all-MiniLM-L6-v2, eval/emb_probe.py): as a CROSS-benchmark difficulty prior
  they fail like hashed text (within-benchmark r 0.02, benchmark-level r 0.17 on held-out folds).
  As a WITHIN-benchmark kernel learned from the revealed labels (centered cosine) they help a little:
  paired ALC difference v6 - v5 = -0.00098 (se 0.00026, benchmark-clustered 0.00042, per-pair split)
  and -0.00077 (clustered se 0.00030, per-benchmark split); hard regime 0.1877 vs 0.1887. tau_m 0.7
  alone (without other changes) gave -0.00077; tau_m 1.0 with tau_t 0.5 was best of 5 settings.
- The numpy MiniLM encoder (src/minilm_np.py) reproduces sentence-transformers exactly (cosine
  1.00000 on 48 pool items) at 39 ms per item on one CPU thread.
- Re-tune around v6 (tune_g6.csv: tau_p {0.8, 1.2} x tau_f {0.7, 1.0} x tau_m {1.0, 1.4}): 0.2074-0.2080,
  none better than v6 (0.2075) beyond 0.0001.
- Regime worlds (PAIEC_WORLD; held-out benchmarks restricted by their pool accuracy, < 0.45 "hard",
  >= 0.6 "easy"), v6 config: hard 0.1813 (210 pairs), easy 0.2074 (183 pairs), all 0.2075.
  Same without the -0.8 shift but with an adaptive pool-mean estimate (sigma_g 1.0): hard 0.1881,
  easy 0.1982, all 0.2065. The fixed shift is worth about 0.007 in a hard world and costs about 0.009
  in an easy one. Adding the adaptive pool mean on top of the shift (sigma_g 0.5 / 1.0): hard
  0.1815 / 0.1818, easy 0.2072 / 0.2068, all 0.2075 / 0.2077: no gain, not used. A formative-sized
  draw holds too few other benchmarks to estimate the pool's difficulty.
- LLM difficulty judge (src/judge.py; Claude Haiku 5.5 rates P(a typical current AI model succeeds),
  0-100, structured JSON output; 54,341 of 54,675 pool item texts rated). On a stratified sample of
  1,812 items (12 per benchmark): within-benchmark correlation with true item difficulty 0.195
  (Haiku 5.5) and 0.242 (Sonnet 5.5); benchmark-level correlation 0.26 and 0.33 (eval/judge_compare.py).
  Compare 0.02 (MiniLM ridge) and 0.04 (hashed text).
- Judge in the model (prior-mean offset = scale x fitted slope x centered judge logit; per-fold slope
  fitted on training benchmarks only, 0.20 for the full fit; plus a within-benchmark linear kernel
  tau_j on the judge logit), tune_judge1.csv: scale 1/2/3 x tau_j 0/0.6 gives 0.2046 / 0.2031 /
  0.2036 / 0.2026 / 0.2046 / 0.2053. Best (scale 2, tau_j 0.6) vs v6: paired diff -0.00489 (se
  0.00067, benchmark-clustered 0.00121), better in 59% of pairs; hard regime 0.1840 vs 0.1877.
- Prior shift with the judge on (tune_judge2): shift -0.4 / -0.8 / -1.2 gives 0.2014 / 0.2026 / 0.2066
  unweighted and 0.1851 / 0.1840 / 0.1862 in the 0.35-accuracy regime; v7 keeps -0.8.
- Sonnet 5.5 as the judge (ratings for 53,664 of 54,675 texts; 1.8% unrated), tune_sonnet1: scale 1.5 /
  2 / 3 with tau_j 0.6 gives 0.2002 / 0.1998 / 0.2034. Best (scale 2) vs v6: paired diff -0.00767 (se
  0.00074, benchmark-clustered 0.00146), better in 67% of pairs; vs the Haiku judge: -0.00278
  (clustered se 0.00087). Hard regime 0.1817 (Haiku 0.1840, v6 0.1877). Per budget: B0 0.2453,
  B1 0.2187, B3 0.2013, B7 0.1893, B15 0.1804, B31 0.1734.
- Runtime review of the v7/v8 package (3 reviewers + adversarial verification, 15 agents): confirmed
  (1) the official-style loader restores sys.path after importing model.py, so the LAZY imports of
  emb.py and judge.py inside predict() failed silently: the semantic kernel (v6, v7, v8) and the Haiku
  judge (v7) were OFF in Codabench runs 1039042, 1039074, 1039214, 1039233 (v8 bound judge.py at import,
  so only its embeddings were affected). Locally the old v6 zip gives synthetic budget-31 Brier 0.2148
  under the path-restoring loader vs 0.2190 with embeddings on. Fixed by binding sibling modules at
  import time; tools/run_smoke_test.py now restores sys.path like the official checker and reports
  whether each sibling module loaded. (2) Failed judge calls were not cached and were retried serially
  inside one predict call (per-call timeout risk). Fixed: negative cache, circuit breaker (auth/billing
  /local errors or 8 consecutive failures disable the judge), 30 s deadline on concurrent warm-up,
  cache-only reads after warm-up, robust key-file parsing (BOM/UTF-16/extra text). Smoke-tested: no key,
  invalid key (disabled at once), blackholed endpoint (worst call 1.0 s), live key. CV unchanged
  (v8 0.1998).
- In-context LLM judge (eval/icl_probe.py on episodes dumped by eval/simulate.py with PAIEC_DUMP: 46
  held-out pairs, 26 benchmarks, 25 evaluation items per pair, budgets 1-31, 5,750 predictions with the
  exact revealed context v8 saw). Sonnet 5.5 sees the target system's revealed outcomes (up to 31) and
  other systems' (up to 24) on the benchmark, then the new item. Brier weighted over budgets 1-31 (ALC
  weights; budget 0 unchanged): v8 0.2003, LLM alone 0.1896, average 0.1891, cross-fitted logistic blend
  0.1902. Per budget (v8 / LLM / average): B1 0.2469 / 0.2261 / 0.2293, B3 0.2326 / 0.2208 / 0.2204,
  B7 0.2255 / 0.2114 / 0.2105, B15 0.2022 / 0.1949 / 0.1934, B31 0.1890 / 0.1895 / 0.1840.
  Paired over pairs, average - v8 = -0.0109 (se 0.0029, benchmark-clustered 0.0029), 70% of pairs
  better. Fixed weight w on the LLM: 0.5 -0.0109, 0.6 -0.0117, 0.65 -0.0120 (se 0.0035; both benchmark
  halves -0.0123 / -0.0116), 0.7 -0.0121, 0.8 -0.0120; per-budget weights cross-fitted -0.0114.
  Haiku 5.5 as the in-context judge: -0.0077 (average). v9 = v8 + Sonnet in-context judge, w = 0.65.
  Runtime prompt verified byte-identical to the probe prompt on all 5,750 records.
- In-context judge variants on the same 46-pair probe (paired vs v8, budgets 1-31 ALC weights):
  Sonnet with v8's estimate in the prompt ("hint"): LLM alone 0.1883 vs 0.1896 without hint, average
  with v8 0.1894 vs 0.1891 (no gain); model 0.25 + Sonnet 0.375 + Sonnet-hint 0.375: -0.0130 (se 0.0041)
  vs -0.0121 for v9's blend, at twice the calls. Sonnet + Haiku in-context: at most -0.0003 extra.
  Opus 5.5 in-context (budgets 1, 3, 7 only; 3,246 of 3,450 rated): with the same 0.65 blend, 0.003
  better than Sonnet on those budgets (se 0.0014), LLM alone at budget 1 0.2178 vs 0.2261; about 4-5 s
  per call, too slow to risk on Codabench before the deadline. Not used.
- Second, disjoint probe set (snapshots 2-3 of each fold; 51 pairs; v8 budget-weighted Brier 0.1460,
  easier pairs than set 1): in-context Sonnet average with v8 -0.0038; LLM alone 0.1441. Pooled over
  both sets (97 pairs), weight w on the LLM: 0.3 -0.0053 (se 0.0011), 0.4 -0.0064 (0.0014), 0.5 -0.0071
  (0.0017), 0.65 -0.0077 (0.0021; 63% of pairs better). v9 keeps w = 0.65; the pooled -0.0077 is the
  estimate we report (set 1 alone, -0.0120, was optimistic).
- v9 runtime: the in-context call now runs in a background thread concurrently with the item-judge call
  and the GP solve (quick live smoke test: 812 ms mean per call vs 900 ms; worst 1.5 s vs 2.4 s).
- Budget 0 (no revealed labels): the same in-context prompt with an empty context (so the LLM sees the
  target system's name and the item). Dump of 1,455 budget-0 predictions over 97 pairs (4 snapshots per
  fold, 15 items per pair): v8 0.2533, Sonnet alone 0.2317, average 0.2322. Weight 0.65: B0 -0.0244
  (se 0.0064; halves -0.0134 / -0.0351; 67% of pairs better), i.e. ALC -0.0024. v10 = v9 + this
  (icl_b0). Estimated v10 vs v8: about -0.0101 ALC (-0.0077 budgets 1-31, -0.0024 budget 0).
- Blend variants on the pooled 97-pair probe (paired vs v8, budgets 1-31): probability blend w=0.65
  -0.0077 (v9/v10); LLM clipped to [0.03, 0.97] / [0.08, 0.92] / [0.15, 0.85]: -0.0077 / -0.0076 /
  -0.0067; logit-space blend w=0.5 / 0.65: -0.0069 / -0.0074; blend shrunk 10% toward 0.5: -0.0063.
  None better than the plain probability blend.
- Opus 5.5 as the in-context judge at low budgets (set 1, budgets 1/3/7, both models rated; ALC-weighted
  over those budgets, paired vs v8): Sonnet blend w=0.65 (v10) -0.0104 (se 0.0031); Opus w=0.65 -0.0134,
  w=0.8 -0.0145 (se 0.0036), w=0.9 -0.0147. v11 = v10 with Opus (w=0.8) when the target has 1-7 revealed
  labels, Sonnet otherwise; in-context time budget 45 min per process. Estimated about -0.004 ALC more
  than v10 (from 46 pairs only; budgets 15/31 unchanged). Live quick smoke test: 1.08 s mean per call.
- Opus check on the second, disjoint set (51 pairs, budgets 1/3/7; 3,819 of 3,825 rated): v11's routing
  (Opus w=0.8 at 1-7 own labels) vs v10's (Sonnet w=0.65): -0.0038 (se 0.0014) on set 2, -0.0040
  (se 0.0015) on set 1, ALC-weighted over budgets 1-7. Opus at budget 0 (zero-shot, 97 pairs): B0 -0.0305
  (w=0.8) vs Sonnet -0.0244, i.e. only -0.0006 ALC more; not used. Estimated v11 vs v8: about -0.014 ALC.
- Full v11 vs v8 on the 97 probe pairs (all budgets incl. 0, per-pair ALC): -0.0140 (se 0.0027); v8's
  ALC on these pairs 0.1965. Reweighted by pair accuracy (as for the prior shift): to mean accuracy 0.5
  -0.0127 (ESS 96), 0.35 -0.0070 (ESS 68), 0.25 -0.0037 (ESS 46). The LLM judges help most on
  mid-accuracy pairs, so the gain on a hard hidden pool may be closer to -0.007.
- Cross-fitted logistic calibration of the in-context LLM probabilities before the v11 blend (fitted per
  benchmark half; intercept about -0.1, slope about 1.1): budgets 1-31 vs v8 -0.0107 (se 0.0023) vs
  -0.0117 (se 0.0024) uncalibrated. In context, the LLM is already nearly calibrated; not used.
