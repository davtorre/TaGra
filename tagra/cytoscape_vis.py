"""
Cytoscape visualization module for TaGra.

This module provides interactive graph visualization using Cytoscape.js
via ipycytoscape (for Jupyter notebooks) and export functionality for
Cytoscape Desktop.

Features:
- Interactive node selection with feature display
- Node positioning from manifold learning (UMAP, t-SNE, Isomap) or force-directed layout
- Color coding by target attribute
- Export to Cytoscape-compatible formats (JSON, GraphML)

Requirements:
    pip install ipycytoscape

Usage in Jupyter:
    >>> from tagra.cytoscape_vis import CytoscapeVisualizer
    >>> viz = CytoscapeVisualizer(graph, target_attribute='outcome')
    >>> viz.show()  # Display interactive widget

Usage for export:
    >>> viz.export_json('graph.json')  # For Cytoscape.js web apps
    >>> viz.export_cytoscape_json('graph_cytoscape.json')  # For Cytoscape Desktop
"""

import json
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Any, Union, Callable
import networkx as nx
import numpy as np

# Optional imports - gracefully handle missing dependencies
try:
    import ipycytoscape
    from ipywidgets import HTML, VBox, HBox, Output, Layout
    IPYCYTOSCAPE_AVAILABLE = True
except ImportError:
    IPYCYTOSCAPE_AVAILABLE = False
    ipycytoscape = None


# Default color palettes
DEFAULT_PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9a6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
]

# Default Cytoscape.js stylesheet
DEFAULT_STYLESHEET = [
    {
        'selector': 'node',
        'style': {
            'width': 2,
            'height': 2,
            'label': 'data(label)',
            'font-size': '8px',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'data(color)',
            'border-width': 1,
            'border-color': '#333'
        }
    },
    {
        'selector': 'node:selected',
        'style': {
            'border-width': 3,
            'border-color': '#ff0',
            'width': 15,
            'height': 15
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 1,
            'line-color': '#ccc',
            'opacity': 0.6,
            'curve-style': 'bezier'
        }
    },
    {
        'selector': 'edge:selected',
        'style': {
            'line-color': '#ff0',
            'width': 2,
            'opacity': 1
        }
    }
]


class CytoscapeVisualizer:
    """
    Interactive Cytoscape.js visualization for TaGra graphs.

    This class provides an alternative to the matplotlib-based visualization,
    offering interactivity through Cytoscape.js widgets in Jupyter notebooks.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to visualize (typically from TaGra's create_graph)
    target_attribute : str, optional
        Node attribute to use for coloring nodes
    pos : dict, optional
        Dictionary mapping node IDs to (x, y) positions.
        If None, positions will be computed using the specified layout.
    layout : str, default='preset'
        Layout algorithm to use if pos is None.
        Options: 'preset' (use provided pos), 'cose', 'grid', 'circle',
                 'concentric', 'breadthfirst', 'random'
    palette : list, optional
        List of hex color codes for node coloring
    node_size : int, default=20
        Base node size in pixels
    show_labels : bool, default=False
        Whether to show node labels by default
    verbose : bool, default=True
        Print status messages

    Examples
    --------
    Basic usage in Jupyter:

    >>> from tagra.cytoscape_vis import CytoscapeVisualizer
    >>> viz = CytoscapeVisualizer(G, target_attribute='Death', pos=manifold_pos)
    >>> viz.show()

    Export for Cytoscape Desktop:

    >>> viz.export_cytoscape_json('my_graph.cyjs')
    """

    def __init__(
        self,
        graph: nx.Graph,
        target_attribute: Optional[str] = None,
        pos: Optional[Dict[int, tuple]] = None,
        layout: str = 'preset',
        palette: Optional[List[str]] = None,
        node_size: int = 20,
        show_labels: bool = False,
        verbose: bool = True
    ):
        self.graph = graph
        self.target_attribute = target_attribute
        self.pos = pos
        self.layout = layout if pos is None else 'preset'
        self.palette = palette or DEFAULT_PALETTE
        self.node_size = node_size
        self.show_labels = show_labels
        self.verbose = verbose

        # Internal state
        self._cytoscape_widget = None
        self._info_widget = None
        self._color_map = {}
        self._cytoscape_data = None

        # Build color map for target attribute
        if target_attribute:
            self._build_color_map()

        # Prepare Cytoscape data
        self._prepare_data()

    def _build_color_map(self) -> None:
        """Build mapping from attribute values to colors."""
        if not self.target_attribute:
            return

        values = set()
        for node in self.graph.nodes():
            val = self.graph.nodes[node].get(self.target_attribute)
            if val is not None:
                values.add(val)

        sorted_values = sorted(values, key=str)
        for i, val in enumerate(sorted_values):
            self._color_map[val] = self.palette[i % len(self.palette)]

    def _get_node_color(self, node: int) -> str:
        """Get color for a node based on target attribute."""
        if not self.target_attribute:
            return self.palette[0]

        val = self.graph.nodes[node].get(self.target_attribute)
        return self._color_map.get(val, '#888888')

    def _compute_positions(self) -> Dict[int, tuple]:
        """Compute node positions if not provided."""
        if self.pos is not None:
            # Handle both dictionary and numpy array formats
            if isinstance(self.pos, dict):
                return {int(k): (float(v[0]), float(v[1])) for k, v in self.pos.items()}
            else:
                # Assume numpy array with shape (n_nodes, 2)
                return {k: (float(self.pos[k][0]), float(self.pos[k][1])) for k in range(self.pos.shape[0])}

        if self.verbose:
            print(f"{datetime.now()}: Computing layout using spring_layout...")

        # Use NetworkX spring layout as default
        pos = nx.spring_layout(self.graph, seed=42, scale=500)

        # Convert to the format we need (ensuring integer keys)
        return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}

    def _scale_positions(self, pos: Dict[int, tuple], scale: float = 1000) -> Dict[int, tuple]:
        """Scale positions to reasonable pixel coordinates."""
        if pos is None or len(pos) == 0:
            return {}

        # Get bounds
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1

        # Add padding (30% on each side)
        padding = scale * 0.3

        # Scale to [padding, scale - padding] range
        effective_scale = scale - 2 * padding
        scaled = {}
        for node, (x, y) in pos.items():
            scaled_x = padding + ((x - x_min) / x_range) * effective_scale
            scaled_y = padding + ((y - y_min) / y_range) * effective_scale
            scaled[node] = (scaled_x, scaled_y)

        return scaled

    def _format_node_features(self, node_id: int) -> str:
        """Format node features as HTML for display."""
        attrs = dict(self.graph.nodes[node_id])

        html_lines = [
            f"<div style='font-family: monospace; padding: 10px; "
            f"background: #f5f5f5; border-radius: 5px; max-width: 400px;'>",
            f"<h4 style='margin: 0 0 10px 0; color: #333;'>Node {node_id}</h4>",
            "<table style='width: 100%; border-collapse: collapse;'>"
        ]

        for key, value in sorted(attrs.items(), key=lambda x: str(x[0])):
            # Highlight target attribute
            if key == self.target_attribute:
                style = "background: #e3f2fd; font-weight: bold;"
            else:
                style = ""

            # Format numeric values
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            html_lines.append(
                f"<tr style='{style}'>"
                f"<td style='padding: 4px; border-bottom: 1px solid #ddd;'><b>{key}</b></td>"
                f"<td style='padding: 4px; border-bottom: 1px solid #ddd;'>{value_str}</td>"
                f"</tr>"
            )

        html_lines.append("</table></div>")
        return "".join(html_lines)

    def _prepare_data(self) -> None:
        """Prepare graph data in Cytoscape.js format."""
        positions = self._compute_positions()
        scaled_pos = self._scale_positions(positions)

        nodes = []
        edges = []

        # Build nodes
        for node in self.graph.nodes():
            node_data = {
                'id': str(node),
                'label': str(node) if self.show_labels else '',
                'color': self._get_node_color(node)
            }

            # Add all node attributes as data
            for key, value in self.graph.nodes[node].items():
                # Convert non-serializable types to string
                if isinstance(value, (int, float, str, bool)):
                    node_data[key] = value
                else:
                    node_data[key] = str(value)

            node_entry = {'data': node_data}

            # Add position if available
            if node in scaled_pos:
                x, y = scaled_pos[node]
                node_entry['position'] = {'x': x, 'y': y}

            nodes.append(node_entry)

        # Build edges
        for i, (source, target) in enumerate(self.graph.edges()):
            edges.append({
                'data': {
                    'id': f'e{i}',
                    'source': str(source),
                    'target': str(target)
                }
            })

        self._cytoscape_data = {
            'nodes': nodes,
            'edges': edges
        }

    def _create_widget(self) -> Any:
        """Create the ipycytoscape widget."""
        if not IPYCYTOSCAPE_AVAILABLE:
            raise ImportError(
                "ipycytoscape is required for interactive visualization. "
                "Install with: pip install ipycytoscape"
            )

        # Create Cytoscape widget
        cyto = ipycytoscape.CytoscapeWidget()

        # Set graph data
        cyto.graph.add_graph_from_json(self._cytoscape_data)

        # Apply stylesheet
        cyto.set_style(DEFAULT_STYLESHEET)

        # Set layout
        if self.layout == 'preset' and self.pos is not None:
            cyto.set_layout(name='preset')
        else:
            cyto.set_layout(name=self.layout)

        # Set widget size
        cyto.layout = Layout(width='100%', height='600px')

        return cyto

    def show(self, height: str = '600px') -> Any:
        """
        Display the interactive Cytoscape visualization in Jupyter.

        Parameters
        ----------
        height : str, default='600px'
            Height of the visualization widget

        Returns
        -------
        widget
            IPython widget containing the visualization

        Notes
        -----
        Click on nodes to see their features in the info panel below.
        """
        if not IPYCYTOSCAPE_AVAILABLE:
            raise ImportError(
                "ipycytoscape is required for interactive visualization. "
                "Install with: pip install ipycytoscape"
            )

        # Create Cytoscape widget
        self._cytoscape_widget = self._create_widget()
        self._cytoscape_widget.layout.height = height

        # Create info display area
        self._info_widget = Output(
            layout=Layout(
                width='100%',
                min_height='100px',
                border='1px solid #ddd',
                padding='10px',
                margin='10px 0'
            )
        )

        # Initial message
        with self._info_widget:
            from IPython.display import display, HTML as IPyHTML
            display(IPyHTML(
                "<div style='color: #666; font-style: italic;'>"
                "Click on a node to see its features</div>"
            ))

        # Set up click handler
        def on_node_click(event):
            """Handle node click events."""
            with self._info_widget:
                from IPython.display import clear_output, display, HTML as IPyHTML
                clear_output(wait=True)

                if event and 'data' in event:
                    node_id = event['data'].get('id')
                    if node_id is not None:
                        try:
                            node_id_int = int(node_id)
                            html = self._format_node_features(node_id_int)
                            display(IPyHTML(html))
                        except (ValueError, KeyError):
                            display(IPyHTML(f"<div>Node: {node_id}</div>"))

        self._cytoscape_widget.on('node', 'click', on_node_click)

        # Create legend if target attribute exists
        legend_widget = self._create_legend() if self.target_attribute else None

        # Combine widgets
        if legend_widget:
            container = VBox([
                HBox([self._cytoscape_widget, legend_widget]),
                self._info_widget
            ])
        else:
            container = VBox([self._cytoscape_widget, self._info_widget])

        if self.verbose:
            print(f"{datetime.now()}: Cytoscape visualization ready. Click nodes to inspect features.")

        return container

    def _create_legend(self) -> Any:
        """Create a legend widget for the color mapping."""
        if not self._color_map:
            return None

        html_parts = [
            "<div style='padding: 10px; background: #f9f9f9; border: 1px solid #ddd; "
            "border-radius: 5px; min-width: 150px;'>",
            f"<b>{self.target_attribute}</b><br><br>"
        ]

        for value, color in sorted(self._color_map.items(), key=lambda x: str(x[0])):
            html_parts.append(
                f"<div style='margin: 3px 0;'>"
                f"<span style='display: inline-block; width: 15px; height: 15px; "
                f"background: {color}; border: 1px solid #333; vertical-align: middle;'></span>"
                f"<span style='margin-left: 5px;'>{value}</span>"
                f"</div>"
            )

        html_parts.append("</div>")

        return HTML("".join(html_parts))

    def export_json(self, filepath: str) -> None:
        """
        Export graph to Cytoscape.js JSON format.

        This format can be loaded directly by Cytoscape.js web applications.

        Parameters
        ----------
        filepath : str
            Output file path (should end with .json)
        """
        export_data = {
            'elements': {
                'nodes': self._cytoscape_data['nodes'],
                'edges': self._cytoscape_data['edges']
            },
            'style': DEFAULT_STYLESHEET,
            'layout': {'name': self.layout}
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        if self.verbose:
            print(f"{datetime.now()}: Graph exported to {filepath} (Cytoscape.js JSON format)")

    def export_cytoscape_json(self, filepath: str) -> None:
        """
        Export graph to Cytoscape Desktop JSON format (.cyjs).

        This format can be imported directly into Cytoscape Desktop application.

        Parameters
        ----------
        filepath : str
            Output file path (should end with .cyjs or .json)
        """
        # Cytoscape Desktop format
        positions = self._compute_positions()
        scaled_pos = self._scale_positions(positions)

        nodes = []
        for node in self.graph.nodes():
            node_data = {
                'id': str(node),
                'name': str(node),
                'SUID': node
            }

            # Add all attributes
            for key, value in self.graph.nodes[node].items():
                if isinstance(value, (int, float, str, bool)):
                    node_data[key] = value
                else:
                    node_data[key] = str(value)

            node_entry = {'data': node_data}

            if node in scaled_pos:
                x, y = scaled_pos[node]
                node_entry['position'] = {'x': x, 'y': y}

            nodes.append(node_entry)

        edges = []
        for i, (source, target) in enumerate(self.graph.edges()):
            edges.append({
                'data': {
                    'id': str(i),
                    'source': str(source),
                    'target': str(target),
                    'SUID': i
                }
            })

        export_data = {
            'format_version': '1.0',
            'generated_by': 'TaGra',
            'target_cytoscapeVersion': '3.9',
            'data': {
                'name': 'TaGra Graph',
                'description': 'Graph generated by TaGra from tabular data'
            },
            'elements': {
                'nodes': nodes,
                'edges': edges
            }
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        if self.verbose:
            print(f"{datetime.now()}: Graph exported to {filepath} (Cytoscape Desktop format)")

    def export_graphml(self, filepath: str) -> None:
        """
        Export graph to GraphML format.

        GraphML is widely supported by graph visualization tools including
        Cytoscape Desktop, Gephi, and yEd.

        Parameters
        ----------
        filepath : str
            Output file path (should end with .graphml)
        """
        # Add positions as node attributes if available
        positions = self._compute_positions()
        if positions:
            for node in self.graph.nodes():
                if node in positions:
                    self.graph.nodes[node]['x'] = positions[node][0]
                    self.graph.nodes[node]['y'] = positions[node][1]

        # Add color as attribute
        for node in self.graph.nodes():
            self.graph.nodes[node]['color'] = self._get_node_color(node)

        nx.write_graphml(self.graph, filepath)

        if self.verbose:
            print(f"{datetime.now()}: Graph exported to {filepath} (GraphML format)")

    def get_cytoscape_data(self) -> Dict[str, Any]:
        """
        Get the raw Cytoscape.js formatted data.

        Returns
        -------
        dict
            Dictionary containing 'nodes' and 'edges' in Cytoscape.js format
        """
        return self._cytoscape_data.copy()

    def export_html(self, filepath: str, title: str = "TaGra Graph Visualization") -> None:
        """
        Export graph to a standalone HTML file with embedded Cytoscape.js.

        This creates an interactive visualization that can be opened in any
        modern web browser without requiring Cytoscape Desktop or Jupyter.

        Parameters
        ----------
        filepath : str
            Output file path (should end with .html)
        title : str, default="TaGra Graph Visualization"
            Title for the HTML page
        """
        # Generate legend HTML if target attribute exists
        legend_html = ""
        if self._color_map:
            legend_items = []
            for value, color in sorted(self._color_map.items(), key=lambda x: str(x[0])):
                legend_items.append(
                    f'<div style="margin: 3px 0;">'
                    f'<span style="display: inline-block; width: 15px; height: 15px; '
                    f'background: {color}; border: 1px solid #333; vertical-align: middle;"></span>'
                    f'<span style="margin-left: 5px;">{value}</span></div>'
                )
            legend_html = f'''
            <div id="legend" style="position: absolute; top: 10px; right: 10px;
                 background: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;
                 padding: 10px; z-index: 1000;">
                <b>{self.target_attribute}</b><br><br>
                {"".join(legend_items)}
            </div>
            '''

        html_template = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        #header {{
            background: #2c3e50;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #header h1 {{
            margin: 0;
            font-size: 1.5em;
        }}
        #main {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        #cy {{
            flex: 1;
            background: #f8f9fa;
            position: relative;
        }}
        #info-panel {{
            width: 350px;
            background: #fff;
            border-left: 1px solid #ddd;
            padding: 15px;
            overflow-y: auto;
        }}
        #info-panel h3 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .feature-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .feature-table td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
        }}
        .feature-table td:first-child {{
            font-weight: bold;
            color: #555;
            width: 40%;
        }}
        .target-row {{
            background: #e3f2fd !important;
        }}
        #placeholder {{
            color: #888;
            font-style: italic;
            text-align: center;
            margin-top: 50px;
        }}
        #stats {{
            font-size: 0.85em;
            color: #666;
        }}
        .controls {{
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }}
        .controls button {{
            padding: 8px 15px;
            margin-right: 5px;
            border: 1px solid #ddd;
            background: #fff;
            border-radius: 4px;
            cursor: pointer;
        }}
        .controls button:hover {{
            background: #f0f0f0;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <div id="stats">Loading...</div>
    </div>
    <div id="main">
        <div id="cy">
            {legend_html}
        </div>
        <div id="info-panel">
            <h3>Node Information</h3>
            <div class="controls">
                <button onclick="cy.fit()">Fit to Screen</button>
                <button onclick="cy.zoom(cy.zoom() * 1.2)">Zoom In</button>
                <button onclick="cy.zoom(cy.zoom() / 1.2)">Zoom Out</button>
            </div>
            <div id="node-info">
                <div id="placeholder">Click on a node to see its features</div>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        var graphData = {json.dumps(self._cytoscape_data)};

        // Target attribute for highlighting
        var targetAttribute = {json.dumps(self.target_attribute)};

        // Initialize Cytoscape
        var cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: graphData,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'width': 2,
                        'height': 2,
                        'background-color': 'data(color)',
                        'border-width': 1,
                        'border-color': '#333'
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': 3,
                        'border-color': '#ff0',
                        'width': 15,
                        'height': 15
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 1,
                        'line-color': '#ccc',
                        'opacity': 0.6,
                        'curve-style': 'bezier'
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                        'line-color': '#ff0',
                        'width': 2,
                        'opacity': 1
                    }}
                }}
            ],
            layout: {{
                name: 'preset'
            }},
            minZoom: 0.1,
            maxZoom: 5
        }});

        // Update stats
        document.getElementById('stats').innerHTML =
            'Nodes: ' + cy.nodes().length + ' | Edges: ' + cy.edges().length;

        // Node click handler
        cy.on('tap', 'node', function(evt) {{
            var node = evt.target;
            var data = node.data();

            var html = '<table class="feature-table">';

            // Sort keys alphabetically
            var keys = Object.keys(data).sort();

            for (var i = 0; i < keys.length; i++) {{
                var key = keys[i];
                // Skip internal properties
                if (key === 'id' || key === 'color' || key === 'label') continue;

                var value = data[key];
                if (typeof value === 'number') {{
                    value = value.toFixed(4);
                }}

                var rowClass = (key === targetAttribute) ? 'target-row' : '';
                html += '<tr class="' + rowClass + '"><td>' + key + '</td><td>' + value + '</td></tr>';
            }}

            html += '</table>';
            html += '<p style="margin-top: 15px; font-size: 0.85em; color: #666;">';
            html += 'Node ID: ' + data.id + '<br>';
            html += 'Degree: ' + node.degree() + '</p>';

            document.getElementById('node-info').innerHTML = html;
        }});

        // Click on background to deselect
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                document.getElementById('node-info').innerHTML =
                    '<div id="placeholder">Click on a node to see its features</div>';
            }}
        }});
    </script>
</body>
</html>'''

        with open(filepath, 'w') as f:
            f.write(html_template)

        if self.verbose:
            print(f"{datetime.now()}: Interactive HTML visualization saved to {filepath}")

    def update_node_colors(
        self,
        color_func: Callable[[int], str]
    ) -> None:
        """
        Update node colors using a custom function.

        Parameters
        ----------
        color_func : callable
            Function that takes a node ID and returns a hex color string

        Examples
        --------
        >>> def color_by_degree(node_id):
        ...     degree = G.degree(node_id)
        ...     if degree > 10:
        ...         return '#ff0000'
        ...     return '#0000ff'
        >>> viz.update_node_colors(color_by_degree)
        """
        for node_entry in self._cytoscape_data['nodes']:
            node_id = int(node_entry['data']['id'])
            node_entry['data']['color'] = color_func(node_id)

        # Update widget if it exists
        if self._cytoscape_widget is not None:
            self._cytoscape_widget.graph.clear()
            self._cytoscape_widget.graph.add_graph_from_json(self._cytoscape_data)


def visualize_cytoscape(
    graph: nx.Graph,
    target_attribute: Optional[str] = None,
    pos: Optional[Dict[int, tuple]] = None,
    output_path: Optional[str] = None,
    output_format: str = 'cyjs',
    verbose: bool = True,
    **kwargs
) -> Optional[Any]:
    """
    Convenience function for Cytoscape visualization.

    In Jupyter environments, displays an interactive widget.
    Can also export to file for use with Cytoscape Desktop.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to visualize
    target_attribute : str, optional
        Node attribute for coloring
    pos : dict, optional
        Node positions from manifold learning
    output_path : str, optional
        If provided, export to this file instead of showing widget
    output_format : str, default='cyjs'
        Export format: 'cyjs' (Cytoscape Desktop), 'json' (Cytoscape.js), 'graphml'
    verbose : bool, default=True
        Print status messages
    **kwargs
        Additional arguments passed to CytoscapeVisualizer

    Returns
    -------
    widget or None
        Returns widget if showing, None if exporting

    Examples
    --------
    Interactive display:

    >>> visualize_cytoscape(G, target_attribute='outcome', pos=manifold_pos)

    Export for Cytoscape Desktop:

    >>> visualize_cytoscape(G, pos=manifold_pos, output_path='graph.cyjs')
    """
    viz = CytoscapeVisualizer(
        graph,
        target_attribute=target_attribute,
        pos=pos,
        verbose=verbose,
        **kwargs
    )

    if output_path:
        if output_format == 'cyjs':
            viz.export_cytoscape_json(output_path)
        elif output_format == 'json':
            viz.export_json(output_path)
        elif output_format == 'graphml':
            viz.export_graphml(output_path)
        else:
            raise ValueError(f"Unknown format: {output_format}. Use 'cyjs', 'json', or 'graphml'")
        return None
    else:
        return viz.show()


# Alias for backward compatibility with potential future API
cytoscape_graph_visualization = visualize_cytoscape
