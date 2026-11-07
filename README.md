# PAIEC entry: hierarchical Gaussian-process logistic predictor

Entry by Vihan Aggarwal (Millburn High School) to the
[Predictive AI Evaluation Competition](https://aims.stanford.edu/competition) (NeurIPS 2026).
The task is to predict P(an AI system answers a benchmark item correctly) on benchmarks never seen
in training, from subject and item attributes plus a few revealed labels (budgets 0, 1, 3, 7, 15, 31
per subject-benchmark pair). Scoring is Brier ALC (lower is better; constant 0.5 = 0.25).

**Method (submitted version v10).** For each benchmark in the revealed evidence, a Gaussian-process
logistic model over the latent logit, fitted by a Laplace approximation to ALL revealed labels that
share the target's `benchmark_id` (any subject):
benchmark offset + subject-benchmark pair offset + item similarity (cosine kernels on
`item_features` tokens, on TF-IDF of `item_content`, and, from v6, on sentence embeddings of the
item from all-MiniLM-L6-v2, run by a bundled numpy implementation so no extra package is needed),
plus, from v7, an LLM difficulty judge: Claude (Haiku 5.5 in v7, Sonnet 5.5 in v8) rates each item's
chance of success through the Anthropic API, and the rating shifts the item's prior mean and enters a
within-benchmark kernel. From v9, an in-context judge: Sonnet 5.5 sees the revealed outcomes on the
target's benchmark (the target system's first) and the new item, and its probability is blended with
the model's (weight 0.65); v10 also queries it at budget 0 with an empty context (it then sees only the
system's name and the item). Without an API key every judge switches off and the model is exactly v6. The prior mean and variances are
estimated on the training benchmarks and tuned by leave-benchmarks-out cross-validation. The
predictive probability uses the probit approximation, so budget-0 predictions are the
prior-predictive mean. `model.py` needs only numpy and falls back to an identical pure-Python
solver if numpy is missing. Details and results: `report/report.pdf`, `results.md`.

## Local cross-validation (7-fold leave-benchmarks-out, 541 pairs, 151 benchmarks)

| version | ALC | B0 | B1 | B3 | B7 | B15 | B31 |
|---|---|---|---|---|---|---|---|
| constant 0.5 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| v0 empirical mean (official baseline) | 0.2426 | 0.2500 | 0.3595 | 0.2351 | 0.2059 | 0.1938 | 0.1875 |
| v1 prior only | 0.2518 | 0.2518 | 0.2518 | 0.2518 | 0.2518 | 0.2518 | 0.2518 |
| v2 + benchmark offset | 0.2123 | 0.2522 | 0.2272 | 0.2126 | 0.2021 | 0.1966 | 0.1936 |
| v3 + pair offset | 0.2089 | 0.2518 | 0.2259 | 0.2097 | 0.1981 | 0.1915 | 0.1870 |
| v4 / v4b + item kernels, no subject prior | 0.2071 | 0.2502 | 0.2250 | 0.2085 | 0.1968 | 0.1890 | 0.1825 |
| v5 = v4b with prior mean shifted by -0.8 | 0.2085 | 0.2589 | 0.2267 | 0.2091 | 0.1968 | 0.1890 | 0.1825 |

| v6 = v5 + within-benchmark semantic kernel | 0.2075 | 0.2587 | 0.2266 | 0.2088 | 0.1956 | 0.1872 | 0.1798 |

| v7 = v6 + Haiku 5.5 judge | 0.2026 | 0.2495 | 0.2224 | 0.2035 | 0.1908 | 0.1828 | 0.1775 |
| v8 = v7 with Sonnet 5.5 judge | 0.1998 | 0.2453 | 0.2187 | 0.2013 | 0.1893 | 0.1804 | 0.1734 |

v9 and v10 (submitted) cannot be cross-validated in full (one long API call per item and budget), so
they were validated on held-out episodes dumped by the evaluator (97 pairs): the in-context judge lowers
the budget-weighted Brier over budgets 1-31 by 0.0077 (paired se 0.0021) and B0 by 0.0244 (se 0.0064),
an estimated total of about 0.010 ALC below v8. See `results.md` and the report.

v5 trades +0.0014 ALC under the training distribution for -0.0035 when held-out pairs are reweighted to
mean accuracy 0.35, the hidden-test level reported by public entries (see `results.md`, `eval/regime_reweight.py`).

## Layout

```
src/model.py, src/fit.py      model (predict) and prior fitting; copied into each submission
src/emb.py, src/emb_text.py   item text for the encoder and embedding provider (cache or live)
src/minilm_np.py, src/minilm/ numpy all-MiniLM-L6-v2 encoder and its exported weights (Apache-2.0)
src_eb/                       empirical-Bayes alternative (ablation)
submissions/<v>/              materialized submissions: model.py, config.json, state/prior.json
tools/download_data.py        fetch measurement-db tables (gated: needs an HF token)
tools/build_dataset.py        keep strictly binary item-level benchmarks -> data/train/
tools/build_cache.py          per-pair statistics + simulation pools -> data/cache/
tools/make_submission.py      fit on all training benchmarks and write submissions/<v>/
tools/build_zip.py            zip a submission folder (model.py at the root) -> dist/<v>.zip
tools/check_submission_zip.py official archive validator (from aims-foundations/paiec_baseline, MIT)
tools/streaming_ingestion.py  official streaming client helper (same source), used by the validator
tools/run_smoke_test.py       local smoke test of the streaming protocol (synthetic data)
eval/simulate.py              local evaluator (leave-benchmarks-out CV mimicking the platform)
eval/tune.py                  grid search over config overrides
eval/amortized_probe.py       probe: do hashed item features predict difficulty on new benchmarks?
eval/emb_probe.py             same probe for sentence embeddings
eval/paired.py                paired per-pair ALC difference between two runs (mean, clustered se)
eval/regime_reweight.py       ALC of saved runs reweighted to a target mean accuracy (hard regime)
tools/export_minilm.py        export MiniLM weights for the numpy encoder and verify equality
tools/embed_pool.py, tools/emb_bytext.py   embed simulation-pool items; re-key by text hash
report/                       LaTeX source of the technical report (NeurIPS 2026 style)
```

## Reproduce

Python 3.11. About 15 GB of disk for the raw tables.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Accept the terms of the gated dataset
[aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
and log in (`huggingface-cli login`, or set `HF_TOKEN`). Then:

```bash
python tools/download_data.py --items      # data/raw (tables only, no traces)
python tools/build_dataset.py              # data/train + benchmark_selection.csv
python tools/build_cache.py                # data/cache/pairs.parquet and data/cache/pool/
```

The semantic kernel (v6) needs the MiniLM weights exported for the numpy encoder (already
committed in `src/minilm/`; to regenerate them, and the pool embeddings used by the local evaluator):

```bash
pip install torch sentence-transformers          # export and pool embedding only, not at runtime
python tools/export_minilm.py src/minilm         # weights.npz + vocab.txt, checks numpy == reference
python tools/embed_pool.py sentence-transformers/all-MiniLM-L6-v2 minilm
python tools/emb_bytext.py minilm
```

The LLM judge needs Anthropic API ratings of the pool items (about 54,700 texts; roughly $3 with
Haiku 5.5 and $50 with Sonnet 5.5) for local validation and for fitting its slope:

```bash
export ANTHROPIC_API_KEY=...                       # your key; never commit it
python tools/judge_pool.py --workers 20                                        # Haiku 5.5 -> judge_haiku.json
PAIEC_JUDGE_MODEL=claude-sonnet-5-5 python tools/judge_pool.py --workers 20 --out data/cache/judge_sonnet.json
python eval/judge_compare.py 12 data/cache/judge_sonnet_sample.json           # correlation with true difficulty
```

Fit and build the submitted version (v8), then validate it (v6 config shown further below):

```bash
PAIEC_JUDGE_FIT_FILE=judge_sonnet.json python tools/make_submission.py v8 '{"subject_prior_scale": 0.0, "tau_b": 2.2, "tau_p": 1.0, "tau_f": 1.0, "tau_t": 0.5, "tau_m": 1.0, "mu0_shift": -0.8, "judge_scale": 2.0, "tau_j": 0.6, "judge_model": "claude-sonnet-5-5"}'
python tools/build_zip.py v8
python tools/check_submission_zip.py dist/v8.zip
python tools/run_smoke_test.py dist/v8.zip                 # without a key: judge off (identical to v6)
ANTHROPIC_API_KEY=... python tools/run_smoke_test.py dist/v8.zip   # live judge calls
```

**API key for Codabench.** The zip does not contain a key. Before uploading, add a file named
`anthropic_key.txt` (the key on one line) at the ZIP root; `src/judge.py` reads it at scoring time.
Do not commit it (`.gitignore` excludes it). Without it the submission behaves exactly like v6.

Local cross-validation of the judge versions uses the cached ratings:

```bash
PAIEC_JUDGE_FIT_FILE=judge_sonnet.json PAIEC_JUDGE_CACHE=$PWD/data/cache/judge_sonnet.json PAIEC_JUDGE_NO_LIVE=1 \
  PAIEC_EMB_CACHE=$PWD/data/cache/emb_minilm_bytext.npz PAIEC_EMB_NO_LIVE=1 python eval/simulate.py submissions/v8
```

In-context judge validation (v9/v10): dump held-out episodes with v8's predictions and the exact
revealed context, then query the judge on them (about $25 per 5,000 Sonnet calls):

```bash
PAIEC_DUMP=$PWD/data/cache/icl/dump PAIEC_JUDGE_FIT_FILE=judge_sonnet.json PAIEC_JUDGE_CACHE=$PWD/data/cache/judge_sonnet.json \
  PAIEC_JUDGE_NO_LIVE=1 PAIEC_EMB_CACHE=$PWD/data/cache/emb_minilm_bytext.npz PAIEC_EMB_NO_LIVE=1 python eval/simulate.py submissions/v8
python eval/icl_probe.py data/cache/icl/dump data/cache/icl/sonnet.json        # budgets 1-31
python eval/icl_weights.py data/cache/icl/sonnet.json                          # blend weight
# second set: PAIEC_DUMP_FROM=2; budget 0: PAIEC_DUMP_B0=1 PAIEC_DUMP_SNAPS=4 PAIEC_DUMP_ITEMS=15, then --budgets 0
PAIEC_JUDGE_FIT_FILE=judge_sonnet.json python tools/make_submission.py v10 '{"subject_prior_scale": 0.0, "tau_b": 2.2, "tau_p": 1.0, "tau_f": 1.0, "tau_t": 0.5, "tau_m": 1.0, "mu0_shift": -0.8, "judge_scale": 2.0, "tau_j": 0.6, "judge_model": "claude-sonnet-5-5", "icl_weight": 0.65, "icl_model": "claude-sonnet-5-5", "icl_b0": true}'
python tools/build_zip.py v10 && python tools/check_submission_zip.py dist/v10.zip
ANTHROPIC_API_KEY=... python tools/run_smoke_test.py dist/v10.zip --quick
```

The v6 version (no API) is built and validated like this:

```bash
python tools/make_submission.py v6 '{"subject_prior_scale": 0.0, "tau_b": 2.2, "tau_p": 1.0, "tau_f": 1.0, "tau_t": 0.5, "tau_m": 1.0, "mu0_shift": -0.8}'
python tools/build_zip.py v6
python tools/check_submission_zip.py dist/v6.zip
python tools/run_smoke_test.py dist/v6.zip
python tools/run_smoke_test.py dist/v6.zip --no-numpy
```

Local cross-validation (about 20 s on 4 CPUs):

```bash
export PAIEC_EMB_CACHE=$PWD/data/cache/emb_minilm_bytext.npz PAIEC_EMB_NO_LIVE=1   # precomputed embeddings
python eval/simulate.py submissions/v6                     # per-pair split (default)
python eval/simulate.py submissions/v6 --split benchmark   # per-benchmark split
python eval/paired.py eval/results/v5_pair_k7_s0_pairs.csv eval/results/v6_pair_k7_s0_pairs.csv
python eval/regime_reweight.py 0.35 eval/results/v5_pair_k7_s0_pairs.csv eval/results/v4_pair_k7_s0_pairs.csv
python eval/simulate.py eval/const_model                   # sanity check, ALC 0.2500
```

The other versions: `v1 '{"use_labels": false}'`, `v2 '{"use_pair": false}'`, `v3 '{}'`, and
`v4` / `v4b` with the v5 config minus `mu0_shift` (v4b only adds the numpy-free fallback).

## Data and disclosure

Training data: only `aims-foundations/measurement-db` (the competition's public training release,
part of the benchmark inventory). No other data. API use: v7-v10 call the Anthropic API (Claude
Haiku 5.5 / Sonnet 5.5) at scoring time to rate items (v9/v10 also with the revealed labels in context), and the same models rated the training items
offline; no web search. One pretrained model:
`sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0), used frozen as a text encoder; its weights are
redistributed in `src/minilm/` under that license.
LLM coding assistants (Claude Code) were used for development and for drafting the report. The
hard-regime prior shift in v5 follows public entries (nguefackuriel/paiec-gpirt, paiksca/paiec-2026,
doctawho42/PAIEC); ideas only, no code from other teams.
`tools/check_submission_zip.py` and `tools/streaming_ingestion.py` are copied unchanged from
[aims-foundations/paiec_baseline](https://github.com/aims-foundations/paiec_baseline) (MIT).

## License

MIT (see `LICENSE`). Downloaded data keep their upstream licenses (measurement-db: CC BY-SA 4.0).
