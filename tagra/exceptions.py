"""
TaGra exception classes.

This module defines the exception hierarchy for the TaGra package.
All TaGra-specific exceptions inherit from TaGraError.
"""


class TaGraError(Exception):
    """Base exception for all TaGra errors."""
    pass


class ConfigurationError(TaGraError):
    """Raised when there's an error in configuration."""
    pass


class PreprocessingError(TaGraError):
    """Raised when preprocessing fails."""
    pass


class GraphConstructionError(TaGraError):
    """Raised when graph construction fails."""
    pass


class AnalysisError(TaGraError):
    """Raised when graph analysis fails."""
    pass


class IOError(TaGraError):
    """Raised when reading or writing files fails."""
    pass


class VisualizationError(TaGraError):
    """Raised when visualization fails."""
    pass
