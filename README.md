# Amazon ML Business Entity Resolution

## 1. Overview

This repository provides a reproducible, precision-oriented entity-resolution
pipeline for the ML Challenge 2026. It links records from Source 2 and Source 3
to every Source 1 reference entity and produces the required tab-separated
submission artifacts.

## 2. Problem definition

Given noisy business names, addresses, and country labels from three independent
sources, identify all Source 2/Source 3 records referring to each Source 1
business. A Source 1 entity may have zero, one, or multiple matches.

## 3. Dataset structure

All challenge files are TSV files. Record files contain `entity_id`,
`business_name`, `business_address`, and `country`. Source is identified by the
`S1-`, `S2-`, or `S3-` ID prefix and by the file name. Training additionally
contains `train_ground_truth.tsv` with `source1_entity_id` and a comma-separated
`matched_entity_ids` list. The supplied country values are US and India. Country is retained as an
explicit string feature rather than being inferred from entity IDs.

## 4. Restrictions

The implementation uses only supplied challenge data. It performs no external
business lookup, geocoding, API augmentation, or internet-based enrichment. It
does not assume a closed country vocabulary and uses no prohibited identity
source. The preferred matching model is CatBoost (GPU when available), with automatic
fallback to CPU CatBoost and then scikit-learn LogisticRegression. Blocking and
feature extraction remain CPU-bound.

## 5. Architecture

The modular flow is: explicit TSV loading → Unicode/text normalization →
high-recall blocking → pairwise similarity features → CatBoost (with a
logistic-regression fallback) → thresholded prediction → submission writing.
Public compatibility
modules live directly under `src/`; implementation modules are under
`src/entity_resolution/`.

## 6. Installation

```bash
python -m pip install -r requirements.txt
```

Python 3.11+ is recommended. The dependency list pins pandas, NumPy,
scikit-learn, joblib, Jupyter, and nbformat. CatBoost is intentionally optional
because its GPU build/environment is platform-specific; install it separately
when CatBoost is desired:

```bash
python -m pip install catboost
```

## 7. Notebook order

Run notebooks in this order when an interactive walkthrough is desired:

1. `notebooks/01_dataset_intelligence.ipynb`
2. `notebooks/02_matching_experiments.ipynb`
3. `notebooks/03_model_training.ipynb`
4. `notebooks/04_final_pipeline.ipynb`

The notebooks document contracts and demonstrate in-memory APIs; the CLI remains
the canonical end-to-end entry point.

## 8. Training

Train a persisted model from the supplied training directory:

```bash
python run_pipeline.py --mode train --data-dir dataset --model-dir models
```

By default training tries CatBoost on GPU, then CPU CatBoost, then
LogisticRegression if CatBoost or its requested device is unavailable. Select a
model/device explicitly when needed:

```bash
python run_pipeline.py --mode train --model-type catboost --device gpu
python run_pipeline.py --mode train --model-type logistic
```

For GPU training, install a CatBoost package compatible with the host CUDA
runtime and verify the NVIDIA driver/CUDA setup before running the command.
`--device auto` (the default) attempts GPU first and is safe on CPU-only hosts.
Only model training is accelerated; blocking, feature generation, and
prediction orchestration remain CPU-bound.

Positive pairs come from the supplied ground truth. Blocked, unlabelled pairs are
sampled as negatives. The resulting model is saved as `models/pair_model.joblib`. Existing artifacts
created by the logistic implementation remain loadable; newly saved artifacts
include the selected backend metadata.

## 9. Prediction

Train and predict together:

```bash
python run_pipeline.py --mode all --data-dir dataset --model-dir models --output-dir output
```

For explicit source files and an existing model:

```bash
python run_pipeline.py \
  --source1 dataset/test/test_source1.tsv \
  --source2 dataset/test/test_source2.tsv \
  --source3 dataset/test/test_source3.tsv \
  --model models/pair_model.joblib --output output \
  --threshold 0.78 --batch-size 4096 --max-candidates 500 --n-jobs 1 --verbose
```

## 10. Output

Prediction writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv` (or the same two names beneath the directory
passed to `--output`). Every Source 1 row is retained. ID lists are comma
separated; empty lists represent singletons.

## 11. Evaluation

Use `python run_pipeline.py --mode evaluate` for the training diagnostic. The
pipeline reports pair-level and entity-level metrics where available. The
primary threshold-selection metric is F0.5, which weights precision more
heavily than recall; precision, recall, F1, and confusion information remain
visible.

## 12. Computational considerations

Blocking avoids an unrestricted Cartesian comparison and uses country, name-token,
address-digit, and phonetic keys. The inverted index is stored in a temporary
SQLite database in the working directory, so the 10M+ comparison rows are not
converted into Python dictionaries or nested dict/set indexes. `--max-candidates`
is enforced in each indexed per-key lookup and bounded merge for every Source 1
row. The `(key, ordinal)` WITHOUT ROWID primary key lets SQLite return each
blocking slice in ordinal order; Python merges only those bounded slices, avoiding
the previous per-row global `GROUP BY`/`ORDER BY` temp sort. Training uses reservoir
sampling to retain only the configured `max_training_pairs` total (500,000 by
default), split according to `negative_ratio`, and never materializes all
positive or negative pairs;
prediction still emits the same candidate mapping and output files.

Candidate and feature progress is logged at the smaller of every 10,000 records
or 1%, including elapsed rate and ETA when the input size is known. For a 16GB
host, keep the candidate bound and process one command at a time:

```bash
python run_pipeline.py --mode train --max-candidates 500 --max-training-pairs 500000 --device auto --verbose
python run_pipeline.py --mode predict --max-candidates 500 --device auto --verbose
```

The temporary `.blocking-*.sqlite` file is removed automatically. Blocking,
feature extraction, and prediction orchestration remain CPU-bound; CatBoost
still attempts GPU first with `--device auto` and falls back to CPU.

## 13. Reproducibility

All paths, thresholds, candidate limits, and random seeds are configurable.
Dependencies are pinned, normalization is deterministic, and negative sampling
uses a fixed default seed of 42. Model artifacts are generated locally rather
than committed.

## 14. Limitations

The phonetic key is intentionally lightweight and language agnostic. Logistic
regression cannot model every nonlinear interaction, and aggressive candidate
limits can reduce recall on unusually dense blocks. No dataset-specific measured
score is claimed in this documentation.

## 15. Future improvements

Potential extensions include calibrated gradient-boosted pair models, learned
country-aware transliteration, adaptive multi-pass blocking, uncertainty
sampling, larger-scale batch scoring, and systematic threshold selection on a
held-out validation split.
