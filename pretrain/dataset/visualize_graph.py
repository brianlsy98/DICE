import torch
import networkx as nx
import matplotlib.pyplot as plt

# Load the saved dataset
dataset_name = 'pretraining_dataset_wo_device_params'  # Adjust if necessary
torch_data = torch.load(f'{dataset_name}.pt')

graphs = torch_data['all_data']

for idx, graph in enumerate(graphs):
    # Extract node features and edge attributes from the graph
    nf = graph.x                       # Node features
    edge_indices = graph.edge_index  # Edge indices
    ef = graph.edge_attr               # Edge features

    G = nx.Graph()

    # Add nodes to the NetworkX graph
    for i, node_attr in enumerate(nf):
        G.add_node(i, **{'attr': node_attr})

    # Add edges to the NetworkX graph
    for i in range(edge_indices.shape[1]):
        src = edge_indices[0, i]
        dst = edge_indices[1, i]
        edge_attr = ef[i]
        G.add_edge(src, dst, **{'attr': edge_attr})

    # Draw the graph
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    
    # Optionally, draw node and edge labels
    node_labels = {i: f"{i}\n{nf[i]}" for i in G.nodes()}
    edge_labels = { (u, v): f"{G[u][v]['attr']}" for u, v in G.edges()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title(f"Graph {idx}")
    plt.axis('off')
    plt.show()
