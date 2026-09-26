#!/usr/bin/env python3
"""Command-line entry point for training and/or predicting."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from entity_resolution.config import PipelineConfig
from entity_resolution.pipeline import EntityResolutionPipeline
from entity_resolution.io import read_records
from entity_resolution.model import PairModel

def main() -> int:
    parser = argparse.ArgumentParser(description="Run business entity resolution")
    parser.add_argument("--mode", choices=("train", "predict", "evaluate", "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--source1", type=Path, help="Source 1 TSV for explicit prediction")
    parser.add_argument("--source2", type=Path, help="Source 2 TSV for explicit prediction")
    parser.add_argument("--source3", type=Path, help="Source 3 TSV for explicit prediction")
    parser.add_argument("--model", type=Path, help="Persisted model path")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--max-training-pairs", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-type", choices=("catboost", "logistic"), default="catboost",
                        help="Preferred pair model; CatBoost falls back automatically")
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto",
                        help="CatBoost device preference (GPU is tried first by default)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger(__name__).info("pipeline starting in %s mode", args.mode)
    output_dir = args.output or args.output_dir
    config = PipelineConfig(
        data_dir=args.data_dir, model_dir=args.model_dir, output_dir=output_dir,
        threshold=args.threshold, max_candidates_per_entity=args.max_candidates,
        random_state=args.random_state, max_training_pairs=args.max_training_pairs,
        model_type=args.model_type, device=args.device,
    )
    pipeline = EntityResolutionPipeline(config)
    explicit = all((args.source1, args.source2, args.source3))
    if explicit:
        if args.mode == "train":
            parser.error("explicit source files support prediction; training uses --data-dir")
        model_path = args.model or (args.model_dir / "pair_model.joblib")
        pipeline.predict_frames(read_records(args.source1), read_records(args.source2),
                                read_records(args.source3), PairModel.load(model_path))
        logging.getLogger(__name__).info("pipeline completed successfully")
        return 0
    if args.mode in ("train", "all"):
        pipeline.train()
    if args.mode == "evaluate":
        pipeline.evaluate()
    if args.mode in ("predict", "all"):
        pipeline.predict()
    logging.getLogger(__name__).info("pipeline completed successfully")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("pipeline interrupted; stopping cleanly")
        raise SystemExit(130)
