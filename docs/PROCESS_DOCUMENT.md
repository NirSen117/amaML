# Entity Resolution Process Document

## Section 1 — Executive summary

This process resolves noisy business records from Sources 2 and 3 against the
Source 1 reference. It combines deterministic normalization,
high-recall blocking, pairwise similarity features, and a persisted CatBoost
classifier (GPU preferred), with CPU CatBoost and logistic-regression
fallbacks.

## Section 2 — Business problem

Records from independent systems may describe the same real-world business with
different names, addresses, punctuation, abbreviations, or missing components.
The required result is the set of Source 2/Source 3 IDs matching each Source 1
ID.

## Section 3 — Input data contract

All challenge files are tab-separated. Record files contain `entity_id`,
`business_name`, `business_address`, and `country`. Ground truth contains
`source1_entity_id` and `matched_entity_ids`.

## Section 4 — Source relationships

Source 1 is the deduplicated reference source. Source 2 and Source 3 contain
records to match. Prefixes `S1-`, `S2-`, and `S3-` identify source membership.

## Section 5 — Country handling

The supplied country values are US and India. Country is retained as an explicit
string feature; no country-specific address parser or external country lookup is
used.

## Section 6 — Expected noise

Names may contain abbreviations, legal-suffix differences, DBA names,
transliterations, punctuation changes, word transpositions, and typos. Addresses
may contain abbreviations, transliteration variants, missing components,
landmarks, numbering differences, or reordered components.

## Section 7 — Data loading

The loader uses explicit `pathlib.Path` values, `sep="\t"`, string dtypes, and
validated required columns. It does not discover, sample, or infer files.

## Section 8 — Data validation

Required record columns and required ground-truth columns are checked before
processing. Invalid schemas raise a descriptive error.

## Section 9 — Text normalization

Values undergo Unicode NFKC normalization, case folding, punctuation removal,
whitespace collapsing, and tokenization. Legal suffixes can be excluded for name
comparison.

## Section 10 — Address normalization

Addresses use the same general text normalization and additionally expose digit
tokens. Digits support matching municipal numbers and postal-code-like fragments
without assuming a country-specific format.

## Section 11 — Blocking design

Candidates are generated using country keys, normalized name-token keys,
address-digit keys, and a compact phonetic name key. The country block serves as
the broad fallback for noisy fields.

## Section 12 — Candidate controls

`max_candidates` caps the number of candidates retained per Source 1 entity.
Candidate IDs are deduplicated and remain restricted to Sources 2 and 3.
Candidate recall ceiling and reduction ratio: **To be measured during execution.**

## Section 13 — Similarity features

The feature vector contains name sequence ratio, name token Jaccard, phonetic
agreement, address sequence ratio, address token Jaccard, digit overlap, country
agreement, and exact-field count.

## Section 14 — Training labels

Positive pairs are read from the supplied ground truth. Negative pairs are
blocked candidate pairs not present in the positive mapping and are sampled with
a configurable ratio and deterministic random seed.

## Section 15 — Matching model

The default model is CatBoost with GPU preferred. If CatBoost is not installed,
GPU initialization fails, or CPU/GPU training otherwise fails, training falls
back to CPU CatBoost and then balanced logistic regression. `--model-type
logistic` selects the deterministic scikit-learn backend directly, while
`--device auto|gpu|cpu` controls CatBoost device preference. The classifier is
serialized with joblib together with the feature schema and backend metadata;
older logistic artifacts without metadata remain loadable.

## Section 16 — Threshold policy

Pairs whose predicted match probability meets the configured threshold are
emitted. The default threshold is 0.78 and can be overridden on the CLI.
Selected threshold result: **To be measured during execution.**

## Section 17 — Inference process

Inference loads all three explicit test source files, generates candidates for
each Source 1 row, scores each candidate, and retains candidates meeting the
threshold. Source 1 ordering is preserved.

## Section 18 — Submission files

`matching_results.tsv` contains `source1_entity_id` and comma-separated
`matched_entity_ids`. `candidate_pairs.tsv` contains
`source1_entity_id` and comma-separated `candidate_entity_ids`. Empty lists are
valid for singleton entities.

## Section 19 — Output validation

The supplied standard-library validator checks completeness, duplicate IDs,
valid source prefixes, existence in test files, and that final matches are a
subset of candidates.

## Section 20 — Evaluation methodology

The challenge uses macro F_0.5 across Source 1 entities. Precision is weighted
more heavily than recall, and correctly predicting an empty list for a singleton
receives full credit. Validation F_0.5: **To be measured during execution.**

## Section 21 — Reproducibility and operations

The CLI exposes paths, threshold, batch size, candidate limit, job count,
verbosity, and random seed. Dependencies are pinned in `requirements.txt`; CatBoost is optional because GPU
installation depends on the host CUDA runtime. Install it separately with
`python -m pip install catboost` for CatBoost training. GPU acceleration applies
to model fitting only: file loading, blocking, feature construction, and
prediction orchestration remain CPU-bound.
Generated model artifacts are stored under `models/`; outputs are stored under
`output/` or the requested `--output` directory.

## Section 22 — Limitations and future work

The current phonetic key is lightweight, the logistic fallback is not a
nonlinear ensemble, and finite candidate caps can affect recall. CatBoost GPU
availability and CUDA compatibility are deployment-dependent. Future work may
add adaptive blocking, transliteration-aware features, validation-driven
threshold tuning, and scalable batch inference. Runtime, memory, and final
leaderboard results: **To be measured during execution.**
