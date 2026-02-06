"""
Registry for graph constructors.

This module provides a registry pattern for managing and extending
graph construction methods.
"""

from typing import Dict, Type, Optional, List
from .base import GraphConstructor


class ConstructorRegistry:
    """
    Registry for graph construction methods.

    Allows registration and retrieval of graph constructor classes,
    enabling extensibility with custom construction methods.

    Examples
    --------
    Register a custom constructor:

    >>> from tagra.construction import ConstructorRegistry, GraphConstructor
    >>> class MyConstructor(GraphConstructor):
    ...     @property
    ...     def method_name(self):
    ...         return "my_method"
    ...     def construct(self, G, values, **kwargs):
    ...         pass
    >>> ConstructorRegistry.register(MyConstructor)

    Get a constructor by name:

    >>> constructor_cls = ConstructorRegistry.get("knn")
    >>> constructor = constructor_cls(k=5)
    """

    _constructors: Dict[str, Type[GraphConstructor]] = {}

    @classmethod
    def register(cls, constructor_class: Type[GraphConstructor]) -> None:
        """
        Register a graph constructor class.

        Parameters
        ----------
        constructor_class : Type[GraphConstructor]
            The constructor class to register
        """
        # Create temporary instance to get method name
        temp_instance = constructor_class.__new__(constructor_class)
        if hasattr(temp_instance, 'method_name'):
            name = temp_instance.method_name
        else:
            name = constructor_class.__name__.lower().replace('constructor', '')
        cls._constructors[name] = constructor_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[GraphConstructor]]:
        """
        Get a constructor class by name.

        Parameters
        ----------
        name : str
            Name of the construction method

        Returns
        -------
        Optional[Type[GraphConstructor]]
            The constructor class, or None if not found
        """
        return cls._constructors.get(name.lower())

    @classmethod
    def list_methods(cls) -> List[str]:
        """
        List all registered construction methods.

        Returns
        -------
        List[str]
            List of method names
        """
        return list(cls._constructors.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[GraphConstructor]:
        """
        Create a constructor instance by name.

        Parameters
        ----------
        name : str
            Name of the construction method
        **kwargs
            Arguments passed to the constructor

        Returns
        -------
        Optional[GraphConstructor]
            Constructor instance, or None if method not found
        """
        constructor_class = cls.get(name)
        if constructor_class is None:
            return None
        return constructor_class(**kwargs)


def register_default_constructors() -> None:
    """Register the default graph constructors."""
    from .knn import KNNConstructor
    from .distance import DistanceThresholdConstructor
    from .similarity import SimilarityThresholdConstructor

    ConstructorRegistry.register(KNNConstructor)
    ConstructorRegistry.register(DistanceThresholdConstructor)
    ConstructorRegistry.register(SimilarityThresholdConstructor)


# Register defaults when module is imported
register_default_constructors()
