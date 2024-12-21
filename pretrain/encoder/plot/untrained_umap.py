import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Ensure umap-learn is installed
import umap.umap_ as umap

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader
from model import Encoder as DICE

if __name__ == "__main__":

    # Load dataset
    dataset = torch.load('./pretrain/dataset/pretraining_dataset_wo_device_params.pt')
    test_data = dataset['test_data']
    dataloader = GraphDataLoader(test_data, batch_size=256, shuffle=True)
    print()
    print("Dataset loaded")

    # Load Parameters
    with open(f"./params.json", 'r') as f:
        params = json.load(f)

    # Initialize an untrained model
    untrained_encoder = DICE(params['model']['dice'])
    print()
    print("Model loaded")

    # Lists to accumulate embeddings and labels
    untrained_node_embeddings = []
    untrained_edge_embeddings = []
    untrained_graph_embeddings = []

    node_labels_all = []
    edge_labels_all = []
    graph_labels_all = []

    # Collect embeddings from all batches
    for batch in tqdm(dataloader, desc='Processing Batches'):
        nf = batch['x']
        ef = batch['edge_attr']

        node_labels = batch['node_y'].detach().cpu().numpy()
        edge_labels = batch['edge_y'].detach().cpu().numpy()
        graph_labels = batch['circuit'].detach().cpu().numpy()

        # Untrained model embeddings
        nh_u, eh_u, gh_u, info_u = untrained_encoder(batch)
        nh_u_np = nh_u.detach().cpu().numpy()
        eh_u_np = eh_u.detach().cpu().numpy()
        gh_u_np = gh_u.detach().cpu().numpy()

        # Append to lists
        untrained_node_embeddings.append(nh_u_np)
        untrained_edge_embeddings.append(eh_u_np)
        untrained_graph_embeddings.append(gh_u_np)

        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)
        graph_labels_all.append(graph_labels)

    # Concatenate all batches
    untrained_node_embeddings = np.concatenate(untrained_node_embeddings, axis=0)
    untrained_edge_embeddings = np.concatenate(untrained_edge_embeddings, axis=0)
    untrained_graph_embeddings = np.concatenate(untrained_graph_embeddings, axis=0)

    node_labels_all = np.concatenate(node_labels_all, axis=0)
    edge_labels_all = np.concatenate(edge_labels_all, axis=0)
    graph_labels_all = np.concatenate(graph_labels_all, axis=0)

    unique_node_labels = np.unique(node_labels_all)
    unique_edge_labels = np.unique(edge_labels_all)
    unique_graph_labels = np.unique(graph_labels_all)

    node_cmap = plt.get_cmap('tab10', len(unique_node_labels))
    edge_cmap = plt.get_cmap('tab10', len(unique_edge_labels))
    graph_cmap = plt.get_cmap('tab10', len(unique_graph_labels))

    # Create figure and axes for subplots: 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    # axes[0]: Untrained GNN Node Embeddings
    # axes[1]: Untrained GNN Edge Embeddings
    # axes[2]: Untrained GNN Graph Embeddings

    # UMAP parameters can be tuned as needed
    umap_params = {
        'n_components': 2,
        'random_state': 42,
        'n_neighbors': 15,
        'min_dist': 0.1
    }

    import time

    print()
    start = time.time()
    print("Node embeddings UMAP (untrained)...")
    umap_node_untrained = umap.UMAP(**umap_params)
    node_embeddings_umap_untrained = umap_node_untrained.fit_transform(untrained_node_embeddings)
    end = time.time()
    print()
    print("Node embeddings UMAP (untrained)")
    print(f"done in {end-start:.2f} seconds")
    print()

    start = time.time()
    print("Edge embeddings UMAP (untrained)...")
    umap_edge_untrained = umap.UMAP(**umap_params)
    edge_embeddings_umap_untrained = umap_edge_untrained.fit_transform(untrained_edge_embeddings)
    end = time.time()
    print()
    print("Edge embeddings UMAP (untrained)")
    print(f"done in {end-start:.2f} seconds")
    print()

    start = time.time()
    print("Graph embeddings UMAP (untrained)...")
    umap_graph_untrained = umap.UMAP(**umap_params)
    graph_embeddings_umap_untrained = umap_graph_untrained.fit_transform(untrained_graph_embeddings)
    end = time.time()
    print()
    print("Graph embeddings UMAP (untrained)")
    print(f"done in {end-start:.2f} seconds")
    print()

    # Plot: Untrained GNN Node Embeddings
    ax = axes[0]
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=node_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Node Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    # Plot: Untrained GNN Edge Embeddings
    ax = axes[1]
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=edge_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Edge Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    # Plot: Untrained GNN Graph Embeddings
    ax = axes[2]
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=graph_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Graph Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plt.tight_layout()
    plt.savefig('./pretrain/encoder/plot/untrained_umap.png')
    plt.show()