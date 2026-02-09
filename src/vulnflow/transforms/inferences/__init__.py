"""
Inferences - Fill gaps with heuristics
Export EventInference, TagInference, STRIDEInference
"""

from __future__ import annotations

from vulnflow.transforms.inferences.events import EventInference
from vulnflow.transforms.inferences.stride import STRIDEInference
from vulnflow.transforms.inferences.tags import TagInference

__all__ = ["EventInference", "STRIDEInference", "TagInference"]
