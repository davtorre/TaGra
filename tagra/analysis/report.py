"""
Report generation for graph analysis.

This module provides functions for generating human-readable
reports of graph analysis results.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List


def generate_report(
    metrics: Dict[str, Any],
    target_attribute: Optional[str] = None,
    communities: Optional[List[List[int]]] = None,
    filepath: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Generate a human-readable analysis report.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of computed metrics
    target_attribute : str, optional
        Target attribute used in analysis
    communities : List[List[int]], optional
        Detected communities
    filepath : str, optional
        If provided, save report to this file
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    str
        The generated report text

    Examples
    --------
    >>> report = generate_report(metrics, target_attribute='label')
    >>> print(report)
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Network Metrics Report")
    lines.append("=" * 60)
    lines.append(f"\nGenerated: {datetime.now()}\n")

    # Basic metrics
    lines.append("-" * 40)
    lines.append("Basic Metrics")
    lines.append("-" * 40)

    if 'nodes' in metrics:
        lines.append(f"Nodes: {metrics['nodes']}")
    if 'edges' in metrics:
        lines.append(f"Edges: {metrics['edges']}")
    if 'density' in metrics:
        lines.append(f"Density: {metrics['density']:.6f}")
        lines.append(f"  (Fraction of possible connections that exist)")
    if 'avg_clustering' in metrics:
        lines.append(f"Average Clustering Coefficient: {metrics['avg_clustering']:.6f}")
        lines.append(f"  (Measure of how nodes tend to cluster)")
    if 'connected_components' in metrics:
        lines.append(f"Connected Components: {metrics['connected_components']}")
    if 'largest_component_size' in metrics and 'nodes' in metrics:
        pct = metrics['largest_component_size'] / metrics['nodes'] * 100
        lines.append(f"Largest Component Size: {metrics['largest_component_size']} ({pct:.1f}%)")
    if 'assortativity' in metrics and metrics['assortativity'] is not None:
        lines.append(f"Degree Assortativity: {metrics['assortativity']:.6f}")
        lines.append(f"  (Tendency to connect to similar-degree nodes)")

    # Community metrics
    if 'community_count' in metrics:
        lines.append("\n" + "-" * 40)
        lines.append("Community Structure")
        lines.append("-" * 40)
        lines.append(f"Communities Found: {metrics['community_count']}")
        if 'modularity' in metrics and metrics['modularity'] is not None:
            lines.append(f"Modularity Score: {metrics['modularity']:.6f}")
            lines.append(f"  (Strength of community division)")

    # Target attribute analysis
    if target_attribute is not None:
        lines.append("\n" + "-" * 40)
        lines.append(f"Target Attribute Analysis: '{target_attribute}'")
        lines.append("-" * 40)

        if 'chi2_stat' in metrics and metrics['chi2_stat'] is not None:
            lines.append(f"Chi-square Statistic: {metrics['chi2_stat']:.6f}")
            lines.append(f"Chi-square p-value: {metrics['chi2_p_value']:.6f}")
            if metrics['chi2_p_value'] < 0.05:
                lines.append("  Significant: Connection patterns are non-random")
            else:
                lines.append("  Not significant: Patterns may be random")

        if 'homophily_score' in metrics and metrics['homophily_score'] is not None:
            lines.append(f"\nHomophily Score: {metrics['homophily_score']:.6f}")
            lines.append(f"  (1.0 = perfect homophily, lower = more mixing)")

            if 'homophily_p_value' in metrics:
                lines.append(f"Homophily p-value: {metrics['homophily_p_value']:.6f}")
            if 'homophily_z_score' in metrics:
                lines.append(f"Homophily Z-score: {metrics['homophily_z_score']:.6f}")
                if abs(metrics['homophily_z_score']) > 2:
                    lines.append("  Strong evidence of non-random connectivity")

    # Interpretation
    lines.append("\n" + "-" * 40)
    lines.append("Interpretation")
    lines.append("-" * 40)

    if 'density' in metrics:
        density = metrics['density']
        if density < 0.01:
            lines.append("Graph Structure: Very sparse")
        elif density < 0.1:
            lines.append("Graph Structure: Sparse")
        elif density < 0.3:
            lines.append("Graph Structure: Moderately connected")
        else:
            lines.append("Graph Structure: Dense")

    if 'modularity' in metrics and metrics['modularity'] is not None:
        mod = metrics['modularity']
        if mod < 0.2:
            lines.append("Community Structure: Weak")
        elif mod < 0.4:
            lines.append("Community Structure: Moderate")
        else:
            lines.append("Community Structure: Strong")

    if target_attribute and 'homophily_score' in metrics and metrics['homophily_score'] is not None:
        homo = metrics['homophily_score']
        if homo < 0.3:
            lines.append("Class Separation: Poor (high mixing)")
        elif homo < 0.7:
            lines.append("Class Separation: Moderate")
        else:
            lines.append("Class Separation: Strong (high homophily)")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)

    # Save to file if requested
    if filepath:
        if verbose:
            print(f"{datetime.now()}: Saving report to {filepath}")
        with open(filepath, 'w') as f:
            f.write(report)

    return report


def format_metrics_dict(
    metrics: Dict[str, Any],
    precision: int = 4
) -> Dict[str, str]:
    """
    Format metrics dictionary for display.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Raw metrics dictionary
    precision : int, default=4
        Decimal precision for floats

    Returns
    -------
    Dict[str, str]
        Formatted metrics as strings
    """
    formatted = {}
    for key, value in metrics.items():
        if value is None:
            formatted[key] = "N/A"
        elif isinstance(value, float):
            formatted[key] = f"{value:.{precision}f}"
        else:
            formatted[key] = str(value)
    return formatted
