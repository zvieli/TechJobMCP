# Milestone 3 Benchmarks

**Reference artifact:** `docs/benchmarks/m3-reference.json` (schema `m3-benchmark-v2`)
**Quality status: diagnostic only — no unbiased independent test-set estimate is available.**

The validation/calibration corpus is contaminated by training overlap, and the calibrated
temperature was selected from that same validation set. View A is a primary reproduction of
the contaminated corpus. View B is a leakage-audited **sensitivity analysis**, not an
independent generalization estimate. Neither view may be presented as an unbiased test-set
result.

## Benchmark Lineage

| Field | Value |
| --- | --- |
| Publication harness (authoritative) | `8135c6a581527ee540c4e813769bc9862b42bde2` |
| Harness commit subject | `fix(evaluation): validate benchmark quality provenance` |
| Benchmark schema | `m3-benchmark-v2` |
| Lineage parent | `None` (fresh root run, no parent) |
| Status | `success` |
| Publication wall time | 594.668102 s (hard budget 4500 s) |
| Reference artifact SHA-256 | `sha256:fe0dd74b147bea82621c53467f373e89e04cfcc94cedf26aea6a3d08c8bfbd1e` |

Superseded lineage: an earlier publication attempt under harness `1569999` / schema
`m3-benchmark-v1` is **non-canonical**. The artifact published here was regenerated from
scratch on a clean checkout of `8135c6a`; no numeric value was carried
forward from the earlier run.

## Reproduction

From a clean checkout of `8135c6a581527ee540c4e813769bc9862b42bde2` with the local (git-ignored) model and
corpus artifacts present:

```bash
export TORCH_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

uv run python .scripts/benchmark_milestone3.py \
  --publish \
  --holdout data/holdout/laya_val.jsonl \
  --training data/training/laya_train.jsonl \
  --model data/models/laya-techjob \
  > /tmp/m3-reference.json
```

Only the flags supported by the committed harness are used. The command refuses a dirty
source tree, refuses a missing/changed artifact, prints JSON only after the canonical
publication validator returns zero blockers, and enforces a 4500-second
hard budget with a 120-second cold-observation subprocess timeout.
The committed JSON is the byte-identical validated output of this run and was not edited.

## Environment

| Field | Value |
| --- | --- |
| Python | CPython 3.14.4 |
| Platform | `Linux-7.0.0-29-generic-x86_64-with-glibc2.43` |
| Accelerator / observed device | cpu / cpu |
| `torch` / `transformers` | 2.14.0+cpu / 5.17.0 |
| `job-mcp` | 0.1.0 |
| Torch threads requested / effective | 4 / 4 |
| Quantization requested / effective | `True` / `dynamic-int8` |
| Quantized Linear modules | 74 |
| Calibrated temperature | `0.7500000000000002` |
| Temperature source | `rl_agent_config.json:calibrated_temperature` |

Execution is CPU-only and local. INT8 is verified as *effective* (dynamic INT8 across
74 Linear modules), not merely requested. The temperature value is
read from the model artifact's calibration config and is unchanged by this milestone; the
checksum below proves which config file was measured.

### Artifact provenance

| Artifact | SHA-256 |
| --- | --- |
| Holdout (`data/holdout/laya_val.jsonl`) | `7b3af1f5d863d16598b902c6cde93e6c4ec0067d24e85d0745b889de6822511e` |
| Training (`data/training/laya_train.jsonl`) | `908335f00fde5909451da77f40ace72c853bf2a2c4fb70a1435881c6e86984ed` |
| Model tree (`data/models/laya-techjob`) | `0b5eade8933a501811dd2e360a278db7ee5164d6df9f1e4e48c65ccc0a87edfc` |
| Calibration config (`rl_agent_config.json`) | `087241d5bfb894cbf68491ef865c401fcf66604ed587d87a5f566d2f8c482fae` |

## Corpus and Protocol

- Holdout records: **447**
- Match-scoring records: **240** → canonical ordered triples:
  **80 pairs** (240 / 3)
- Model-forward: 2 runs × 50 = **100** samples
- Warm ensemble: 2 runs × 50 = **100** samples
- Batch lanes N ∈ [1, 4, 8, 15]: 2 runs × 10 = **20** samples per lane
- `calculate_match_score`: 2 × 30 = **60**
- `filter_jobs`: 2 × 5 = **10**
- Cold load / first inference: **8** fresh-subprocess observations each
- Excluded warmups per warm run: **5** (never measured)
- Input rotation size: **16** distinct payloads (0–15), round-robin with wrap
- Production chunk size: **4** (unchanged; batch lanes measure larger forward batches only)
- Total measured samples across all 10 lanes: **366**

## Quality Methodology

**View A — `primary_reproduction` (A-primary).** Ordered reproduction of the current
validation/calibration corpus: 80 pairs, 240 records, with
0 clusters and 0 pairs excluded. It is a
reproduction view, not a held-out estimate.

**View B — `overlap_excluded_sensitivity` (B-sensitivity).** The same corpus after removing
complete shared-state dependency clusters that also appear in training:
16 clusters / 12 pairs removed, leaving
68 pairs and 204 records. B is a **sensitivity analysis** over the
same contaminated corpus. Removing overlap is not the same as obtaining an independent test
set, so B does not establish unbiased generalization either.

**ECE definition.** Ten equal-width bins over predicted-class confidence on `[0, 1]`
(`[low, high)`, final bin inclusive of `1.0`), exact-match correctness, pooled across the
`match_scoring_skill`, `match_scoring_seniority`, and `match_scoring_recruiter_fit` tasks:

```
ECE = Σ_bins (count_bin / n) · |mean_confidence_bin − accuracy_bin|
```

**Seniority mismatch.** Binary metric with `seniority_fit == 0` treated as the positive class
(one positive = predicted *not* a seniority match). Precision, recall, F1, specificity and
balanced accuracy are derived from the confusion cells exactly as shown; zero-denominator
rates are reported as `0.0`. Ordinal separation diagnostics remain separate from these
binary metrics and are not blended into them.

### Quality results

| View | Pairs | Records | ECE | TP | FP | FN | TN | Prevalence | Accuracy | Precision | Recall | F1 | Specificity | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A — primary reproduction (diagnostic) | 80 | 240 | 0.080689 | 11 | 13 | 14 | 42 | 31.25% | 0.662500 | 0.458333 | 0.440000 | 0.448980 | 0.763636 | 0.601818 |
| B — leakage-audited sensitivity | 68 | 204 | 0.074426 | 12 | 12 | 11 | 33 | 33.82% | 0.661765 | 0.500000 | 0.521739 | 0.510638 | 0.733333 | 0.627536 |
| Δ B − A (canonical `delta_b_minus_a`) | -12 | -36 | -0.006263 | +1 | -1 | -3 | -9 | +2.57 pp | -0.000735 | +0.041667 | +0.081739 | +0.061659 | -0.030303 | +0.025718 |

Δ values are taken verbatim from the canonical `quality.metrics.delta_b_minus_a` field of the
artifact (not recomputed by hand); confusion-cell and prevalence deltas are integer/derived
differences of the same published cells.

**Interpretation.** Relative to View A, View B shows a small ECE improvement
(-0.006263) and a classification profile that shifts slightly toward the positive
class (recall +0.081739, precision +0.041667, balanced accuracy
+0.025718, specificity -0.030303). Because both views
derive from the same contaminated calibration corpus and differ only by the exclusion of
12 overlapping pairs (68 vs 80 pairs), this
movement is a **sensitivity signal, not evidence of generalization**. No quality target is
declared met on this basis.

### Contamination audit (measured, not cleaned away)

| Signal | Value |
| --- | ---: |
| Exact (task, state, label) overlap rows | 30 |
| (task, state) overlap rows | 41 |
| Shared-state duplicate pair clusters | 23 |
| Distinct overlapping states (digests) | 16 |
| Conflicting-label entries | 17 |

Overlap and conflicting labels are **surfaced rather than silently repaired**. The
17 conflict entries are canonical, aggregated one per
(`task`, `state`) pair with merged sorted-unique label sets, restricted to the three approved
match-scoring tasks. Duplicate clusters are repeated *pairs*, not the expected three rows of a
single triple.

## Performance

Latencies are milliseconds derived from the raw nanosecond samples in the artifact. Every
lane uses the estimator recorded in the manifest — `Hyndman-Fan type 7 linear interpolation` — and the `Percentile` column
shows the harness's own resolution flag.

| Lane | Boundary measured | Samples | Runs | Warmups/run | Min ms | P50 ms | P95 ms | P99 ms | Max ms | Percentile |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `cold_load` | cold model load (fresh subprocess) | 8 | 8 | 0 | 2970.259 | 3015.844 | 3529.671 | 3638.184 | 3665.312 | low-N diagnostic |
| `first_inference` | first inference (paired cold subprocess) | 8 | 8 | 0 | 412.956 | 419.119 | 422.994 | 423.623 | 423.780 | low-N diagnostic |
| `model_forward` | model-only forward, batch 1 | 100 | 2 | 5 | 70.990 | 75.461 | 77.880 | 79.521 | 80.695 | type-7 estimate |
| `warm_ensemble` | warm ensemble inference | 100 | 2 | 5 | 409.091 | 444.280 | 469.374 | 483.285 | 483.950 | type-7 estimate |
| `batch_1` | model-only forward, batch 1 | 20 | 2 | 5 | 70.956 | 72.373 | 76.375 | 76.489 | 76.517 | low-N diagnostic |
| `batch_4` | model-only forward, batch 4 | 20 | 2 | 5 | 285.237 | 305.628 | 316.772 | 330.598 | 334.055 | low-N diagnostic |
| `batch_8` | model-only forward, batch 8 | 20 | 2 | 5 | 567.803 | 589.745 | 610.008 | 611.921 | 612.400 | low-N diagnostic |
| `batch_15` | model-only forward, batch 15 | 20 | 2 | 5 | 1126.002 | 1166.677 | 1196.359 | 1208.757 | 1211.857 | low-N diagnostic |
| `calculate_match_score` | production `calculate_match_score` | 60 | 2 | 5 | 496.511 | 544.620 | 581.693 | 594.168 | 602.471 | low-N diagnostic |
| `filter_jobs` | production `filter_jobs` collection scoring | 10 | 2 | 5 | 1480.879 | 1570.012 | 1612.557 | 1613.427 | 1613.644 | low-N diagnostic |

### Cold start and first inference

- **Cold model load** (8 fresh subprocesses): P50 3015.844 ms
  (3.02 s), range 2.97–3.67 s.
- **First inference** (paired in the same cold processes): P50 419.119 ms
  (0.42 s), range 0.41–0.42 s.
- Warm model-only forward at batch 1 is P50 75.461 ms,
  so cold load dominates first-request latency by roughly
  40×; warm ensemble scoring is
  P50 444.280 ms.

### Batch throughput

| Lane | Batch size | Unit | P50 | P95 | Max | Min |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `batch_1` | 1 | pairs_per_second | 13.8172 | 14.0912 | 14.0933 | 13.0690 |
| `batch_4` | 4 | pairs_per_second | 13.0880 | 13.9230 | 14.0234 | 11.9741 |
| `batch_8` | 8 | pairs_per_second | 13.5652 | 13.8958 | 14.0894 | 13.0634 |
| `batch_15` | 15 | pairs_per_second | 12.8570 | 13.3116 | 13.3215 | 12.3777 |

Throughput is flat-to-slightly-declining across batch sizes [1, 4, 8, 15]
(P50 13.82 → 12.86
pairs/s), i.e. under this CPU-only INT8 configuration batching does not yield super-linear
pair throughput; wall time grows roughly with batch size. This is a measured lane result, not
a latency or capacity service-level objective.

### Percentile resolution caveat

The same type-7 estimator is applied to every lane, but small sample counts limit upper-tail
resolution, so the harness labels these lanes **`low-N diagnostic`**: `cold_load`, `first_inference`, `batch_1`, `batch_4`, `batch_8`, `batch_15`, `calculate_match_score`, `filter_jobs`.
Only `model_forward`, `warm_ensemble` (n = 100) carry a `type-7 estimate` label.

Mechanically, type-7 evaluates the 99th percentile at fractional index `(n − 1) × 0.99` of the
sorted samples and linearly interpolates the two bracketing order statistics. The weight that
interpolation places on the single largest observation is what limits tail resolution:

| Lane group | n | Type-7 index | Weight on largest sample | Observations above P99 |
| --- | ---: | ---: | ---: | ---: |
| `cold_load`, `first_inference` | 8 | 6.93 | 0.93 | 1 of 8 |
| `filter_jobs` | 10 | 8.91 | 0.91 | 1 of 10 |
| batch lanes (N = {1, 4, 8, 15}) | 20 | 18.81 | 0.81 | 1 of 20 |
| `calculate_match_score` | 60 | 58.41 | 0.41 | 1 of 60 |
| `model_forward`, `warm_ensemble` | 100 | 98.01 | 0.01 | 1 of 100 |

In the n = 8 cold lanes the largest of 8 observations carries weight 0.93 in the
reported P99, so the tail estimate is effectively "close to the observed worst case" decided by a
single sample, rather than a stable estimate. In the n = 100 lanes the largest sample carries
weight 0.01, so P99 is governed by the second-largest order statistic and rests on materially
more data. That asymmetry in sample support — not a difference in estimator — is why the harness
labels the two long lanes differently from the eight short ones.

Two caveats hold for every lane. P99 is **not** the sample maximum anywhere: observed maxima
exceed the reported P99 by 0.01%–1.48%. And exactly one sample sits above P99 in each lane
— the expected order-statistic position, which also shows how few observations define the tail at
these counts. Low-N P99 values are descriptive summaries of very small samples; they must not be
quoted as tail-latency guarantees or service-level objectives.

## Scope, Guarantees, and Limitations

- **Diagnostic quality only.** No unbiased independent test-set estimate exists for this model;
  the only available corpus is the contaminated calibration/validation set.
- **CPU-only, local.** No GPU, network/provider, or hosted-service performance is measured or
  claimed.
- **Not measured here:** search recall, source freshness, provider coverage, or application
  success rates. Those belong to later milestones and are not implied by this report.
- **Percentiles are descriptive** summaries of a fixed deterministic protocol, not SLOs.
- **Reproduction requires the ignored local artifacts** listed above; their checksums are part
  of the manifest, and the validator fails closed on any mismatch, on a dirty tree, on GPU or
  non-INT8 execution, on fallback/error/synthetic inference origins, and on budget breach.
- Benchmark methodology, sample counts, latency protocol, calibration temperature, thresholds,
  and production scoring were **not** modified by this publication run.
