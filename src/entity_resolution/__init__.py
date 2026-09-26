"""Business entity resolution package."""

from .config import PipelineConfig
from .pipeline import EntityResolutionPipeline

__all__ = ["EntityResolutionPipeline", "PipelineConfig"]
