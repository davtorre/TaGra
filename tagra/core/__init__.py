"""
TaGra Core Module.

This module provides the foundational types and classes for the TaGra package.
"""

from .graph import TaGraGraph
from .types import (
    GraphMetadata,
    ScalingMethod,
    EncodingMethod,
    ConstructionMethod,
    ManifoldMethod,
    NaNAction,
    UnknownColumnAction,
    PreprocessingResult,
    AnalysisResult
)

__all__ = [
    'TaGraGraph',
    'GraphMetadata',
    'ScalingMethod',
    'EncodingMethod',
    'ConstructionMethod',
    'ManifoldMethod',
    'NaNAction',
    'UnknownColumnAction',
    'PreprocessingResult',
    'AnalysisResult'
]
