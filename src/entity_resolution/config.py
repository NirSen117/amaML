"""Configuration objects for the entity-resolution pipeline."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path = Path("dataset")
    model_dir: Path = Path("models")
    output_dir: Path = Path("output")
    threshold: float = 0.78
    max_candidates_per_entity: int = 500
    random_state: int = 42
    negative_ratio: int = 3
    max_training_pairs: int = 500_000
    model_type: str = "catboost"
    device: str = "auto"
