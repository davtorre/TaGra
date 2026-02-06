"""
TaGraGraph class - wrapper around NetworkX graph with metadata.

This module provides the core TaGraGraph class that wraps a NetworkX graph
with additional metadata about how the graph was constructed.
"""

import pickle
from datetime import datetime
from typing import Optional, Dict, Any, Union, Iterator, List
import networkx as nx
import numpy as np

from .types import GraphMetadata, ConstructionMethod


class TaGraGraph:
    """
    A NetworkX graph wrapper with TaGra-specific metadata.

    TaGraGraph wraps a NetworkX Graph object and adds metadata about
    how the graph was constructed from tabular data. It provides a
    familiar interface while preserving construction details.

    Parameters
    ----------
    graph : nx.Graph, optional
        Existing NetworkX graph to wrap. If None, creates an empty graph.
    metadata : GraphMetadata, optional
        Metadata about graph construction. If None, creates default metadata.

    Attributes
    ----------
    graph : nx.Graph
        The underlying NetworkX graph
    metadata : GraphMetadata
        Metadata about graph construction
    manifold_positions : Optional[np.ndarray]
        Node positions from manifold learning

    Examples
    --------
    Create from an existing NetworkX graph:

    >>> G = nx.karate_club_graph()
    >>> tg = TaGraGraph(G)
    >>> print(tg.number_of_nodes())
    34

    Access the underlying NetworkX graph:

    >>> nx_graph = tg.graph
    >>> # or use TaGraGraph directly - most nx methods work
    >>> for node in tg.nodes():
    ...     print(node)
    """

    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        metadata: Optional[GraphMetadata] = None
    ):
        self._graph = graph if graph is not None else nx.Graph()
        self._metadata = metadata if metadata is not None else GraphMetadata()
        self._manifold_positions: Optional[np.ndarray] = None

    @property
    def graph(self) -> nx.Graph:
        """Get the underlying NetworkX graph."""
        return self._graph

    @property
    def metadata(self) -> GraphMetadata:
        """Get the graph metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: GraphMetadata) -> None:
        """Set the graph metadata."""
        self._metadata = value

    @property
    def manifold_positions(self) -> Optional[np.ndarray]:
        """Get manifold learning positions for nodes."""
        return self._manifold_positions

    @manifold_positions.setter
    def manifold_positions(self, value: Optional[np.ndarray]) -> None:
        """Set manifold learning positions."""
        self._manifold_positions = value

    # Delegate common NetworkX methods
    def nodes(self, data: bool = False) -> Any:
        """Return graph nodes."""
        return self._graph.nodes(data=data)

    def edges(self, data: bool = False) -> Any:
        """Return graph edges."""
        return self._graph.edges(data=data)

    def number_of_nodes(self) -> int:
        """Return the number of nodes."""
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        """Return the number of edges."""
        return self._graph.number_of_edges()

    def degree(self, nbunch=None, weight=None):
        """Return node degree(s)."""
        return self._graph.degree(nbunch, weight)

    def neighbors(self, node: int) -> Iterator[int]:
        """Return neighbors of a node."""
        return self._graph.neighbors(node)

    def add_node(self, node: int, **attr) -> None:
        """Add a node with optional attributes."""
        self._graph.add_node(node, **attr)

    def add_edge(self, u: int, v: int, **attr) -> None:
        """Add an edge with optional attributes."""
        self._graph.add_edge(u, v, **attr)

    def has_node(self, node: int) -> bool:
        """Check if node exists."""
        return self._graph.has_node(node)

    def has_edge(self, u: int, v: int) -> bool:
        """Check if edge exists."""
        return self._graph.has_edge(u, v)

    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self._graph)

    def __iter__(self) -> Iterator[int]:
        """Iterate over nodes."""
        return iter(self._graph)

    def __contains__(self, node: int) -> bool:
        """Check if node is in graph."""
        return node in self._graph

    def __getitem__(self, node: int) -> Dict[int, Dict]:
        """Get adjacency dict for node."""
        return self._graph[node]

    def copy(self) -> 'TaGraGraph':
        """Return a copy of the graph."""
        new_graph = TaGraGraph(
            graph=self._graph.copy(),
            metadata=GraphMetadata.from_dict(self._metadata.to_dict())
        )
        if self._manifold_positions is not None:
            new_graph.manifold_positions = self._manifold_positions.copy()
        return new_graph

    def get_node_attribute(self, node: int, attr: str, default: Any = None) -> Any:
        """Get a node attribute."""
        return self._graph.nodes[node].get(attr, default)

    def set_node_attribute(self, node: int, attr: str, value: Any) -> None:
        """Set a node attribute."""
        self._graph.nodes[node][attr] = value

    def get_node_attributes(self, attr: str) -> Dict[int, Any]:
        """Get an attribute for all nodes."""
        return nx.get_node_attributes(self._graph, attr)

    def get_positions_dict(self) -> Optional[Dict[int, tuple]]:
        """
        Get manifold positions as a dictionary.

        Returns
        -------
        Optional[Dict[int, tuple]]
            Dictionary mapping node IDs to (x, y) positions,
            or None if no positions are set.
        """
        if self._manifold_positions is None:
            return None
        return {i: (self._manifold_positions[i, 0], self._manifold_positions[i, 1])
                for i in range(len(self._manifold_positions))}

    def save(self, filepath: str) -> None:
        """
        Save the graph to a file.

        Parameters
        ----------
        filepath : str
            Path to save the graph (should end with .tagra or .pickle)
        """
        data = {
            'graph': self._graph,
            'metadata': self._metadata.to_dict(),
            'manifold_positions': self._manifold_positions
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'TaGraGraph':
        """
        Load a graph from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved graph

        Returns
        -------
        TaGraGraph
            Loaded graph with metadata
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Handle both old format (just nx.Graph) and new format (dict with metadata)
        if isinstance(data, nx.Graph):
            # Old format - just a NetworkX graph
            return cls(graph=data)
        elif isinstance(data, dict):
            # New format with metadata
            tg = cls(
                graph=data.get('graph'),
                metadata=GraphMetadata.from_dict(data.get('metadata', {}))
            )
            tg.manifold_positions = data.get('manifold_positions')
            return tg
        else:
            raise ValueError(f"Unknown graph format in file: {filepath}")

    def to_networkx(self) -> nx.Graph:
        """
        Get the underlying NetworkX graph.

        This is useful for using NetworkX algorithms directly.

        Returns
        -------
        nx.Graph
            The underlying NetworkX graph
        """
        return self._graph

    @classmethod
    def from_networkx(
        cls,
        graph: nx.Graph,
        method: str = "unknown",
        **kwargs
    ) -> 'TaGraGraph':
        """
        Create a TaGraGraph from a NetworkX graph.

        Parameters
        ----------
        graph : nx.Graph
            NetworkX graph to wrap
        method : str
            Construction method used (for metadata)
        **kwargs
            Additional metadata fields

        Returns
        -------
        TaGraGraph
            Wrapped graph with metadata
        """
        try:
            construction_method = ConstructionMethod(method)
        except ValueError:
            construction_method = ConstructionMethod.KNN

        metadata = GraphMetadata(
            construction_method=construction_method,
            k=kwargs.get('k'),
            distance_threshold=kwargs.get('distance_threshold'),
            similarity_threshold=kwargs.get('similarity_threshold'),
            numeric_columns=kwargs.get('numeric_columns', []),
            categorical_columns=kwargs.get('categorical_columns', []),
            target_columns=kwargs.get('target_columns', []),
            ignore_columns=kwargs.get('ignore_columns', [])
        )

        return cls(graph=graph, metadata=metadata)

    def __repr__(self) -> str:
        return (f"TaGraGraph(nodes={self.number_of_nodes()}, "
                f"edges={self.number_of_edges()}, "
                f"method={self._metadata.construction_method.value})")
