"""End-to-end training, inference and artifact writing."""
from pathlib import Path
import logging
import os
import random
import sqlite3
import time
import uuid
import pandas as pd
from .blocking import generate_candidates, iter_candidate_lists
from .config import PipelineConfig
from .io import read_records, read_ground_truth, write_candidates, write_submission
from .model import PairModel
from .evaluate import macro_fbeta

LOGGER = logging.getLogger(__name__)

class EntityResolutionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model_path = config.model_dir / "pair_model.joblib"

    def _paths(self, split: str):
        root = self.config.data_dir / split
        return [read_records(root / f"{split}_source{i}.tsv") for i in (1, 2, 3)]

    @staticmethod
    def _rows(frame: pd.DataFrame):
        return frame.to_dict(orient="records")

    def train(self) -> Path:
        LOGGER.info("training started")
        s1, s2, s3 = self._paths("train")
        left_rows = self._rows(s1)
        left_by_id = {str(row["entity_id"]): row for row in left_rows}
        all_other = pd.concat([s2, s3], ignore_index=True)
        left_index = s1.set_index("entity_id", drop=False)
        other_index = all_other.set_index("entity_id", drop=False)
        left_columns = tuple(s1.columns)
        other_columns = tuple(all_other.columns)
        left_positions = left_index.index
        other_positions = other_index.index

        def row_at(frame: pd.DataFrame, positions: pd.Index, columns: tuple[str, ...], ident: str):
            position = positions.get_loc(ident)
            if not isinstance(position, int):
                position = int(position)
            values = frame.iloc[position].to_numpy(copy=False)
            return dict(zip(columns, values))

        truth = read_ground_truth(self.config.data_dir / "train" / "train_ground_truth.tsv")
        rng = random.Random(self.config.random_state)
        positive_cap = max(1, self.config.max_training_pairs // (1 + self.config.negative_ratio))
        negative_limit = max(1, self.config.max_training_pairs - positive_cap)
        positives = []
        seen_positives = 0
        positive_db_path = f".positive-keys-{os.getpid()}-{uuid.uuid4().hex}.sqlite"
        positive_db = sqlite3.connect(positive_db_path)
        positive_db.execute(
            "CREATE TABLE positive_keys (left_id TEXT NOT NULL, right_id TEXT NOT NULL, "
            "PRIMARY KEY (left_id, right_id))"
        )
        truth_rows = truth.itertuples(index=False, name=None)
        LOGGER.info("generating positive pairs from %d labelled rows", len(truth))
        LOGGER.info("positive-pair lookup index ready; processing ground truth")
        progress_step = max(1, min(10_000, (len(truth) + 99) // 100))
        for position, (source1_id, matched_ids) in enumerate(truth_rows, 1):
            left = left_by_id.get(str(source1_id))
            if left is None:
                continue
            for ident in filter(None, str(matched_ids).split(",")):
                if ident in other_positions:
                    positive_db.execute(
                        "INSERT OR IGNORE INTO positive_keys VALUES (?, ?)",
                        (str(source1_id), ident),
                    )
                    seen_positives += 1
                    pair = (
                        row_at(s1, left_positions, left_columns, str(source1_id)),
                        row_at(all_other, other_positions, other_columns, ident),
                    )
                    if len(positives) < positive_cap:
                        positives.append(pair)
                    else:
                        slot = rng.randrange(seen_positives)
                        if slot < positive_cap:
                            positives[slot] = pair
            if position == len(truth) or position % progress_step == 0:
                LOGGER.info("positive pair progress: %d/%d labelled rows", position, len(truth))
        positive_db.commit()
        LOGGER.info(
            "positive pair generation complete: %d retained from %d valid pairs",
            len(positives), seen_positives,
        )
        negatives = []
        seen_negatives = 0
        LOGGER.info("generating blocked negative pairs from %d source-1 records", len(left_rows))
        progress_step = max(1, min(10_000, (len(left_rows) + 99) // 100))
        negative_started = time.monotonic()
        for position, (left_id, ids) in enumerate(
            iter_candidate_lists(left_rows, all_other, self.config.max_candidates_per_entity), 1
        ):
            left = left_by_id[left_id]
            for ident in ids:
                is_positive = positive_db.execute(
                    "SELECT 1 FROM positive_keys WHERE left_id = ? AND right_id = ?",
                    (str(left["entity_id"]), ident),
                ).fetchone()
                if is_positive is None:
                    seen_negatives += 1
                    pair = (left, row_at(all_other, other_positions, other_columns, ident))
                    if len(negatives) < negative_limit:
                        negatives.append(pair)
                    else:
                        slot = rng.randrange(seen_negatives)
                        if slot < negative_limit:
                            negatives[slot] = pair
            if position == len(left_rows) or position % progress_step == 0:
                elapsed = max(time.monotonic() - negative_started, 1e-9)
                rate = position / elapsed
                LOGGER.info(
                    "negative pair progress: %d/%d source-1 records "
                    "(%.1f records/s, elapsed %.1fs, ETA %.1fs)",
                    position, len(left_rows), rate, elapsed,
                    max(0, len(left_rows) - position) / rate,
                )
        rng.shuffle(negatives)
        positive_db.close()
        try:
            os.unlink(positive_db_path)
        except FileNotFoundError:
            pass
        LOGGER.info("negative pair generation complete: %d pairs retained", len(negatives))
        if not positives or not negatives:
            raise ValueError("Training requires at least one positive and one blocked negative pair")
        PairModel(model_type=self.config.model_type, device=self.config.device).fit(
            positives + negatives,
            [1] * len(positives) + [0] * len(negatives),
            random_state=self.config.random_state,
        ).save(self.model_path)
        LOGGER.info("trained on %d positive and %d negative pairs", len(positives), len(negatives))
        LOGGER.info("training completed")
        return self.model_path

    def predict(self) -> tuple[Path, Path]:
        LOGGER.info("prediction started")
        model = PairModel.load(self.model_path)
        s1, s2, s3 = self._paths("test")
        outputs = self.predict_frames(s1, s2, s3, model)
        LOGGER.info("prediction completed")
        return outputs

    def predict_frames(self, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                       model: PairModel | None = None) -> tuple[Path, Path]:
        model = model or PairModel.load(self.model_path)
        left_rows, other = self._rows(s1), pd.concat([s2, s3], ignore_index=True)
        by_id = other.set_index("entity_id", drop=False)
        LOGGER.info(
            "prediction candidate generation started: %d source-1 and %d comparison records",
            len(left_rows), len(other),
        )
        candidates = generate_candidates(left_rows, other, self.config.max_candidates_per_entity)
        LOGGER.info("prediction candidate generation completed; beginning scoring")
        result_rows, candidate_rows = [], []
        total = len(left_rows)
        progress_step = max(1, min(10_000, (total + 99) // 100))
        started = time.monotonic()
        LOGGER.info("prediction scoring started for %d source-1 records", total)
        try:
            for position, left in enumerate(left_rows, 1):
                ids = candidates.get(left["entity_id"], [])
                pairs = [(left, by_id.loc[i]) for i in ids]
                scores = model.predict_proba(pairs) if pairs else []
                selected = [
                    ident for ident, score in zip(ids, scores)
                    if score >= self.config.threshold
                ]
                candidate_rows.append({
                    "source1_entity_id": left["entity_id"],
                    "candidate_entity_ids": ",".join(ids),
                })
                result_rows.append({
                    "source1_entity_id": left["entity_id"],
                    "matched_entity_ids": ",".join(selected),
                })
                if position % progress_step == 0 or position == total:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = position / elapsed
                    remaining = max(0, total - position)
                    LOGGER.info(
                        "prediction scoring progress: %d/%d source-1 records "
                        "(%.1f records/s, elapsed %.1fs, ETA %.1fs)",
                        position, total, rate, elapsed, remaining / rate,
                    )
        except KeyboardInterrupt:
            LOGGER.warning("prediction scoring interrupted; stopping cleanly")
            raise
        elapsed = max(time.monotonic() - started, 1e-9)
        LOGGER.info(
            "prediction scoring completed: %d/%d source-1 records "
            "(%.1f records/s, elapsed %.1fs)",
            total, total, total / elapsed, elapsed,
        )
        LOGGER.info("writing prediction outputs")
        write_candidates(self.config.output_dir / "candidate_pairs.tsv", candidate_rows)
        write_submission(self.config.output_dir / "matching_results.tsv", result_rows)
        LOGGER.info("prediction outputs saved for %d source-1 records", len(left_rows))
        return self.config.output_dir / "matching_results.tsv", self.config.output_dir / "candidate_pairs.tsv"

    def evaluate(self) -> float:
        """Score the persisted model against labelled training rows (diagnostic only)."""
        model = PairModel.load(self.model_path)
        s1, s2, s3 = self._paths("train")
        left_rows, other = self._rows(s1), pd.concat([s2, s3], ignore_index=True)
        by_id = other.set_index("entity_id", drop=False)
        truth = {
            str(row.source1_entity_id): set(filter(None, str(row.matched_entity_ids).split(",")))
            for row in read_ground_truth(
                self.config.data_dir / "train" / "train_ground_truth.tsv"
            ).itertuples(index=False)
        }
        candidates = generate_candidates(left_rows, other, self.config.max_candidates_per_entity)
        scored = []
        for left in left_rows:
            ids = candidates.get(left["entity_id"], [])
            pairs = [(left, by_id.loc[i]) for i in ids]
            scores = model.predict_proba(pairs) if pairs else []
            scored.append((truth.get(left["entity_id"], set()),
                           {i for i, score in zip(ids, scores) if score >= self.config.threshold}))
        score = macro_fbeta(scored)
        LOGGER.info("training diagnostic macro F0.5: %.4f", score)
        return score
