"""
Core types and enumerations for TaGra.

This module defines the fundamental types used throughout the TaGra package,
including enumerations for methods and a dataclass for graph metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class ScalingMethod(Enum):
    """Methods for scaling numeric columns."""
    STANDARD = "standard"
    MINMAX = "minmax"
    NONE = "none"


class EncodingMethod(Enum):
    """Methods for encoding categorical columns."""
    ONE_HOT = "one-hot"
    LABEL = "label"
    NONE = "none"


class ConstructionMethod(Enum):
    """Methods for constructing graph edges."""
    KNN = "knn"
    DISTANCE = "distance"
    SIMILARITY = "similarity"


class ManifoldMethod(Enum):
    """Methods for manifold learning dimensionality reduction."""
    UMAP = "UMAP"
    TSNE = "TSNE"
    ISOMAP = "Isomap"
    NONE = "none"


class NaNAction(Enum):
    """Actions for handling NaN values."""
    DROP_ROW = "drop row"
    DROP_COLUMN = "drop column"
    INFER = "infer"


class UnknownColumnAction(Enum):
    """Actions for handling unknown columns."""
    INFER = "infer"
    IGNORE = "ignore"


@dataclass
class GraphMetadata:
    """
    Metadata associated with a TaGra graph.

    Stores information about how the graph was constructed, including
    preprocessing parameters, construction method, and column classifications.

    Attributes
    ----------
    construction_method : ConstructionMethod
        Method used to construct graph edges
    k : Optional[int]
        Number of neighbors for KNN method
    distance_threshold : Optional[float]
        Threshold for distance-based construction
    similarity_threshold : Optional[float]
        Threshold for similarity-based construction
    numeric_columns : List[str]
        Columns treated as numeric during preprocessing
    categorical_columns : List[str]
        Columns treated as categorical during preprocessing
    target_columns : List[str]
        Columns designated as targets
    ignore_columns : List[str]
        Columns ignored during preprocessing
    scaling_method : ScalingMethod
        Method used for numeric scaling
    encoding_method : EncodingMethod
        Method used for categorical encoding
    manifold_method : ManifoldMethod
        Manifold learning method used
    manifold_dim : int
        Dimensionality of manifold embedding
    created_at : datetime
        When the graph was created
    source_file : Optional[str]
        Path to the source data file
    n_original_rows : Optional[int]
        Number of rows in original data
    n_original_columns : Optional[int]
        Number of columns in original data
    custom : Dict[str, Any]
        Custom metadata fields
    """
    construction_method: ConstructionMethod = ConstructionMethod.KNN
    k: Optional[int] = 5
    distance_threshold: Optional[float] = None
    similarity_threshold: Optional[float] = None
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    target_columns: List[str] = field(default_factory=list)
    ignore_columns: List[str] = field(default_factory=list)
    scaling_method: ScalingMethod = ScalingMethod.STANDARD
    encoding_method: EncodingMethod = EncodingMethod.ONE_HOT
    manifold_method: ManifoldMethod = ManifoldMethod.UMAP
    manifold_dim: int = 2
    created_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None
    n_original_rows: Optional[int] = None
    n_original_columns: Optional[int] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            'construction_method': self.construction_method.value,
            'k': self.k,
            'distance_threshold': self.distance_threshold,
            'similarity_threshold': self.similarity_threshold,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'target_columns': self.target_columns,
            'ignore_columns': self.ignore_columns,
            'scaling_method': self.scaling_method.value,
            'encoding_method': self.encoding_method.value,
            'manifold_method': self.manifold_method.value,
            'manifold_dim': self.manifold_dim,
            'created_at': self.created_at.isoformat(),
            'source_file': self.source_file,
            'n_original_rows': self.n_original_rows,
            'n_original_columns': self.n_original_columns,
            'custom': self.custom
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphMetadata':
        """Create metadata from dictionary."""
        return cls(
            construction_method=ConstructionMethod(data.get('construction_method', 'knn')),
            k=data.get('k'),
            distance_threshold=data.get('distance_threshold'),
            similarity_threshold=data.get('similarity_threshold'),
            numeric_columns=data.get('numeric_columns', []),
            categorical_columns=data.get('categorical_columns', []),
            target_columns=data.get('target_columns', []),
            ignore_columns=data.get('ignore_columns', []),
            scaling_method=ScalingMethod(data.get('scaling_method', 'standard')),
            encoding_method=EncodingMethod(data.get('encoding_method', 'one-hot')),
            manifold_method=ManifoldMethod(data.get('manifold_method', 'UMAP')),
            manifold_dim=data.get('manifold_dim', 2),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now(),
            source_file=data.get('source_file'),
            n_original_rows=data.get('n_original_rows'),
            n_original_columns=data.get('n_original_columns'),
            custom=data.get('custom', {})
        )


@dataclass
class PreprocessingResult:
    """
    Result of preprocessing a dataframe.

    Attributes
    ----------
    dataframe : Any
        The preprocessed pandas DataFrame
    manifold_positions : Optional[Any]
        Manifold embedding positions (numpy array)
    column_info : Dict[str, List[str]]
        Dictionary with column classifications
    """
    dataframe: Any  # pd.DataFrame
    manifold_positions: Optional[Any] = None  # np.ndarray
    column_info: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """
    Result of graph analysis.

    Attributes
    ----------
    metrics : Dict[str, Any]
        Computed metrics
    communities : Optional[List[List[int]]]
        Detected communities
    neighborhood_probabilities : Optional[Dict]
        Neighborhood probability matrix
    """
    metrics: Dict[str, Any] = field(default_factory=dict)
    communities: Optional[List[List[int]]] = None
    neighborhood_probabilities: Optional[Dict] = None
