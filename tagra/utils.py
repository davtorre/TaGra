import datetime
import functools
import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb


def measure_emissions(func):
    """Decorator that tracks energy consumption and CO2 emissions of a function using CodeCarbon."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from codecarbon import EmissionsTracker
        except ImportError:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "codecarbon"])
            from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            emissions = tracker.stop() * 1000
            energy_wh = tracker._total_energy.kWh * 1000
            execution_time = end_time - start_time
            print(f"\n~ : ~ consumption measured through CodeCarbon ~ : ~")
            print(f"{func.__name__} energy: {energy_wh:.20f} Wh")
            print(f"{func.__name__} emissions: {emissions:.20f} CO₂eq grams")
            print(f"{func.__name__} execution time: {execution_time:.20f} seconds")
            print(f"~ : ~ : ~ : ~ : ~ : ~ : ~")
        return result
    return wrapper


def track_energy(func, *args, skip=False, **kwargs):
    """Run func(*args, **kwargs) with CodeCarbon tracking. Returns (result, energy_wh, emissions_grams).

    Pass skip=True (e.g. via --skip-codecarbon CLI flag) to bypass tracking entirely.
    In that case returns (result, None, None) without touching CodeCarbon.
    """
    if skip:
        return func(*args, **kwargs), None, None

    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "codecarbon"])
        from codecarbon import EmissionsTracker

    tracker = EmissionsTracker(log_level="error", save_to_file=False)
    tracker.start()
    try:
        result = func(*args, **kwargs)
    finally:
        emissions_grams = tracker.stop() * 1000
        energy_wh = tracker._total_energy.kWh * 1000
    return result, energy_wh, emissions_grams


plt.rcParams.update({
    'font.size': 22,  # General font size
    'axes.titlesize': 22,  # Axes title font size
    'axes.labelsize': 22,  # Axes labels font size
    'xtick.labelsize': 16,  # X-tick labels font size
    'ytick.labelsize': 16,  # Y-tick labels font size
    'legend.fontsize': 18,  # Legend font size
    'figure.titlesize': 26  # Figure title font size
    })

def analyze_neighborhood_attributes(graph, target_attribute, return_probs=False):
    """
    Analyzes attributes in the neighborhoods of each node in a graph, optionally returning probabilities.

    Parameters:
    - graph (networkx.Graph): The input graph.
    - attribute_name (str): The name of the node attribute to analyze.
    - return_probs (bool): If True, returns the probability of each attribute in the neighborhood.

    Returns:
    - pd.DataFrame: A DataFrame with each row representing a node. Columns include the node's attribute,
                    degree, and either the count or probability of each attribute in its neighborhood.
    """
    NONE_STR = 'None'
    unique_attributes = set([graph.nodes[n].get(target_attribute, NONE_STR) for n in graph.nodes])
    # unique_attributes = set(nx.get_node_attributes(graph, target_attribute).values())
    data = []
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_attrs = [graph.nodes[n].get(target_attribute, NONE_STR) for n in neighbors]

        attr_counts = {}
        attr_counts[f"node_{target_attribute}"] = graph.nodes[node].get(target_attribute, NONE_STR)
        attr_counts["node_index"] = node
        attr_counts["degree"] = len(neighbors)

        for attr in unique_attributes:
            if return_probs and attr_counts["degree"] > 0:
                attr_counts[f"p_{attr}"] = neighbor_attrs.count(attr) / attr_counts["degree"]
            else:
                attr_counts[f"n_{attr}"] = neighbor_attrs.count(attr)

        data.append(attr_counts)

    cols = ["node_index", f"node_{target_attribute}", "degree"] + \
           [f"{'p' if return_probs else 'n'}_{attr}" for attr in unique_attributes]
    df = pd.DataFrame(data, columns=cols)

    return df

def print_neighbors_prob(df_neigh, label_col):
    # pdb.set_trace()
    probabilities = {}
    for label_i in df_neigh[f'node_{label_col}'].unique():
        nodes_with_label_i = df_neigh[df_neigh[f'node_{label_col}'] == label_i]
        total_degree_i = nodes_with_label_i['degree'].sum()
        for label_j in df_neigh[f'node_{label_col}'].unique():
            col_name = f'n_{label_j}'
            total_neighbors_with_label_j = nodes_with_label_i[col_name].sum()
            probabilities[(label_i, label_j)] = total_neighbors_with_label_j / total_degree_i if total_degree_i else 0

    return probabilities

def heat_map_prob(probabilities, df_neigh, label_col, prob_heatmap_path, verbose):

    # Convert labels to string for better handling
    labels = sorted([str(label) for label in df_neigh[f'node_{label_col}'].unique()])
    probabilities = {(str(k[0]), str(k[1])): v for k, v in probabilities.items()}
    prob_matrix = pd.DataFrame(index=labels, columns=labels, data=0.0)

    for (i, j), prob in probabilities.items():
        prob_matrix.loc[i, j] = prob

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(prob_matrix, cmap='seismic', vmin=0, vmax=1)

    fig.colorbar(cax)

    text_size = max(5, 36 - len(labels))  # Adjust the multiplier as needed

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = prob_matrix.iloc[i, j]
            color = 'w' if value < 0.35 else ('w' if value > 0.65 else 'black')
            text = ax.text(j, i, f"{value:.2f}",
                           ha="center", va="center", color=color, fontsize = text_size)

    ax.set_title('Probability Distribution Heatmap')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Label j')
    plt.ylabel('Label i')
    if prob_heatmap_path:
        plt.savefig(prob_heatmap_path, dpi = 300)
        if verbose: print(f"{datetime.datetime.now()}: Probability heatmap saved in {prob_heatmap_path}")

def plot_distribution(data_dict, outpath, verbose, bins = None, double_log = True):
    data = data_dict['data']
    if bins is None:
        bins = range(0, max(data))
    hist, bin_edges = np.histogram(data, bins = bins)
    fig, ax = plt.subplots()
    ax.scatter(bin_edges[:-1], hist, alpha=0.75, edgecolor='black')
    ax.set_title(data_dict['title'])
    ax.set_xlim((1,None))
    ax.set_ylim((1,None))
    ax.set_xlabel(data_dict['xlabel'])
    ax.set_ylabel(data_dict['ylabel'])
    if double_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.grid()
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi = 300)
        if verbose: print(f"{datetime.datetime.now()}: {data_dict['title']} saved in {outpath}")

def plot_community_composition(G, attribute_name, communities, outpath, verbose, palette = 'seismic'):
    NONE_STR = 'None'
    # communities_generator = nx.algorithms.community.girvan_newman(G)
    # top_level_communities = next(communities_generator)
    # communities = [list(c) for c in sorted(top_level_communities, key=len, reverse=True)]
    if attribute_name is not None:
        labels_per_node = [G.nodes[n].get(attribute_name, NONE_STR) for n in G.nodes]
        unique_labels = set(labels_per_node)
    else:
        labels_per_node = [0 for node in G.nodes()]
        unique_labels = [0]

    community_compositions = {}
    cmap = plt.get_cmap(palette)

    if attribute_name is not None:
        for comm_id, community in enumerate(communities):
            if len(community) == 1:
                continue
            labels_community = [G.nodes[node].get(attribute_name, NONE_STR) for node in community]
            community_compositions[comm_id] = {label: 0 for label in unique_labels}
            measured_unique_labels, counts = np.unique(labels_community, return_counts=True)
            for label, count in zip(measured_unique_labels, counts):
                community_compositions[comm_id][label] = count
    else:
        for comm_id, community in enumerate(communities):
            labels_community = [0 for _ in community]
            unique_labels, counts = np.unique(labels_community, return_counts=True)
            community_compositions[comm_id] = {label: 0 for label in unique_labels}
            for label, count in zip(unique_labels, counts):
                community_compositions[comm_id][label] = count

    if len(community_compositions) == 0:
        print(f"{datetime.datetime.now()}: No communities with more than 1 node found")
        return 0
    indices = list(community_compositions.keys())
    bar_width = 0.9
    fig, ax = plt.subplots(figsize=(8,6))

    bottoms = [0] * len(indices)
    colors = {label: cmap(i) for label, i in zip(unique_labels, np.linspace(0, 1, len(unique_labels)))}

    # Plot bars
    for label in unique_labels:
        values = [community_compositions[idx].get(label, 0) for idx in indices]
        ax.bar(indices, values, bar_width, label=f"{attribute_name}={label}", bottom=bottoms, color=colors[label])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    y_mx = max(bottoms)
    y_max = max(y_mx * 1.1, 1)
    ax.set_xticks([])
    ax.set_xlabel('Community ID')
    ax.set_ylabel('Counts')
    ax.set_title('Counts of outcomes by community ID')
    ax.set_ylim([0, y_max])
    # ax.grid()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if outpath:
        fig.tight_layout()
        fig.savefig(outpath, dpi = 300)
        if verbose: print(f"{datetime.datetime.now()}: Community composition saved in {outpath}")

    return 1
def matplotlib_graph_visualization(G, attribute, outpath, verbose, palette = 'seismic', pos = None):
    NONE_STR = 'None'
    plt.figure(figsize=(10, 10))
    if pos is None:
        pos = nx.spring_layout(G, seed=2112)
        title_string = "Graph of Relations"
    else:
        title_string = "Graph of Relations with manifold learning"

    node_color = []
    cmap = plt.get_cmap(palette)
    if attribute is not None:
        classification_attribute_name = attribute
        y = np.array([G.nodes[node].get(classification_attribute_name, NONE_STR) for node in G.nodes()])
        unique = np.unique(y)
        unique_to_int = {key: index for index, key in enumerate(unique)}
        color_array = [cmap(r) for r in np.linspace(0, 1, len(unique))]
        node_color = [color_array[unique_to_int[key]] for key in y]
    else:
        color = cmap(0)
        node_color = [color for _ in G.nodes()]

    nx.draw(G, pos, with_labels=False, node_size=50, font_color="white", font_size=10, node_color = node_color)
    plt.title(title_string)
    if outpath:
        plt.savefig(outpath, dpi = 300)
        if verbose: print(f'{datetime.datetime.now()}: Graph saved in {outpath}')

def measure_mixing_matrix(G, communities):
    community_edge_count = {(i, j): 0 for i in communities.keys() for j in communities.keys()}

    # Create a mapping from node to its community
    node_to_community = {}
    for community, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = community

    # Check if all nodes in the graph are covered by the communities
    graph_nodes = set(G.nodes())
    community_nodes = set(node_to_community.keys())
    if graph_nodes != community_nodes:
        missing_in_communities = graph_nodes - community_nodes
        missing_in_graph = community_nodes - graph_nodes
        raise ValueError(
            f"The nodes in the communities do not match the nodes in the graph.\n"
            f"Nodes in the graph not in communities: {missing_in_communities}\n"
            f"Nodes in communities not in the graph: {missing_in_graph}"
        )

    # Iterate over all edges in the graph
    for source, target in G.edges():
        if source in node_to_community and target in node_to_community:
            c_source = node_to_community[source]
            c_target = node_to_community[target]
            if not G.is_directed():
                community_edge_count[c_source, c_target] += 1
                community_edge_count[c_target, c_source] += 1
            else:
                community_edge_count[c_source, c_target] += 1

    return community_edge_count
