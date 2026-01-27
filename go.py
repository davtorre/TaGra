import json
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from tagra.preprocessing import preprocess_dataframe
from tagra.graph import create_graph
from tagra.analysis import analyze_graph
from tagra.config import *
from tagra.cytoscape_vis import CytoscapeVisualizer

def main(config_path, dataset_path, target_class):
    start_time = datetime.now()

    config = load_config(config_path, dataset_path)
    if target_class is not None:
        config['target_columns'] = target_class
    # Preprocessing
    df_preprocessed, manifold_pos = preprocess_dataframe(
        input_dataframe=config['input_dataframe'],
        output_directory=config['output_directory'],
        preprocessed_filename=config['preprocessed_filename'],
        inferred_columns_filename=config['inferred_columns_filename'],
        numeric_columns=config['numeric_columns'],
        categorical_columns=config['categorical_columns'],
        target_columns=config['target_columns'],
        unknown_column_action=config['unknown_column_action'],
        ignore_columns=config['ignore_columns'],
        numeric_threshold=config['numeric_threshold'],
        numeric_scaling=config['numeric_scaling'],
        categorical_encoding=config['categorical_encoding'],
        nan_action=config['nan_action'],
        nan_threshold=config['nan_threshold'],
        verbose=config['verbose'],
        manifold_method=config['manifold_method'],
        overwrite=config['overwrite']
    )

    # Graph Creation
    graph = create_graph(
        input_dataframe=config['input_dataframe'],
        output_directory=config['output_directory'],
        graph_filename=config['graph_filename'],
        inferred_columns_filename=config['inferred_columns_filename'],
        numeric_columns=config['numeric_columns'],
        preprocessed_dataframe=df_preprocessed,
        similarity_threshold=config['similarity_threshold'],
        distance_threshold=config['distance_threshold'],
        method=config['method'],
        k=config['k'],
        verbose=config['verbose'],
        overwrite=config['overwrite']
    )
    if config['manifold_method'] is not None:
        pos = manifold_pos
    else:
        pos = None

    # Graph Analysis (without graph visualization - we'll use Cytoscape instead)
    # Temporarily disable matplotlib visualization in analyze_graph
    original_viz_filename = config.get('graph_visualization_filename')

    analyze_graph(
        graph,
        target_attributes=config['target_columns'],
        verbose=config['verbose'],
        output_directory=config['output_directory'],
        degree_distribution_filename=config['degree_distribution_filename'],
        community_filename=config['community_filename'],
        graph_visualization_filename=None,  # Disable matplotlib visualization
        prob_heatmap_filename=config['prob_heatmap_filename'],
        pos=pos,
        overwrite=config['overwrite'],
        network_metrics_filename=config['network_metrics_filename']
    )

    # Cytoscape Visualization (replaces matplotlib)
    if config['verbose']:
        print(f"{datetime.now()}: Creating Cytoscape visualization...")

    # Determine target attribute for coloring
    target_attr = None
    if config['target_columns']:
        if isinstance(config['target_columns'], list) and len(config['target_columns']) > 0:
            target_attr = str(tuple(config['target_columns']))
        elif isinstance(config['target_columns'], str):
            target_attr = config['target_columns']

    # Create Cytoscape visualizer
    viz = CytoscapeVisualizer(
        graph=graph,
        target_attribute=target_attr,
        pos=pos,
        verbose=config['verbose']
    )

    # Determine output filenames
    output_dir = config.get('output_directory', 'results/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate timestamp for filenames if not overwriting
    time_str = datetime.now().strftime('%Y%m%d%H%M')

    # HTML visualization (interactive, can be opened in browser)
    if original_viz_filename:
        base, ext = os.path.splitext(original_viz_filename)
        if not config['overwrite']:
            html_filename = f"{base}_{time_str}.html"
        else:
            html_filename = f"{base}.html"
    else:
        if not config['overwrite']:
            html_filename = f"graph_{time_str}.html"
        else:
            html_filename = "graph.html"

    html_path = os.path.join(output_dir, html_filename)
    viz.export_html(html_path, title="TaGra Graph Visualization")

    # Also export Cytoscape Desktop format (.cyjs)
    if original_viz_filename:
        base, ext = os.path.splitext(original_viz_filename)
        if not config['overwrite']:
            cyjs_filename = f"{base}_{time_str}.cyjs"
        else:
            cyjs_filename = f"{base}.cyjs"
    else:
        if not config['overwrite']:
            cyjs_filename = f"graph_{time_str}.cyjs"
        else:
            cyjs_filename = "graph.cyjs"

    cyjs_path = os.path.join(output_dir, cyjs_filename)
    viz.export_cytoscape_json(cyjs_path)

    end_time = datetime.now()

    print(f"\nAnalysis complete. Execution time: {end_time - start_time}")
    print(f"\nVisualization outputs:")
    print(f"  - Interactive HTML: {html_path}")
    print(f"    (Open in browser for interactive exploration)")
    print(f"  - Cytoscape Desktop: {cyjs_path}")
    print(f"    (Import into Cytoscape Desktop for advanced analysis)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TaGra example with configuration file.')
    parser.add_argument('-c', '--config', type=str, required=False, default=None, help='Path to the configuration file.')
    parser.add_argument('-d', '--dataframe', type=str, required=False, default=None, help='Path to the input dataframe.')
    parser.add_argument('-a', '--attribute', type=str, required=False, default=None, help='Target attribute column name.')
    args = parser.parse_args()

    if args.config is not None and not os.path.isfile(args.config):
        print(f"Error: The configuration file {args.config} does not exist.")
        sys.exit(1)
    if args.dataframe is not None and not os.path.isfile(args.dataframe):
        print(f"Error: The dataset file {args.dataframe} does not exist.")
        sys.exit(1)
    if args.config is None and args.dataframe is None:
        print(f"Error: Either --config or --dataframe must be specified.")
        sys.exit(1)

    main(args.config, args.dataframe, args.attribute)
