import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader
from model import Encoder

if __name__ == "__main__":
    # Load dataset
    dataset = torch.load('../../dataset/pretraining_dataset_wo_device_params.pt')
    test_data = dataset['test_data']
    dataloader = GraphDataLoader(test_data, batch_size=256, shuffle=True)
    print("Dataset loaded")

    # Load Parameters
    with open(f"../params.json", 'r') as f:
        model_params = json.load(f)['model']

    # Initialize both an untrained and trained model
    untrained_encoder = Encoder(model_params)
    trained_encoder = Encoder(model_params)
    trained_encoder.load(f"../saved_models/dice_pretraining.pt")
    print("Model loaded")

    # Lists to accumulate embeddings and labels
    untrained_node_embeddings = []
    untrained_edge_embeddings = []
    untrained_graph_embeddings = []
    trained_node_embeddings = []
    trained_edge_embeddings = []
    trained_graph_embeddings = []

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

        # Trained model embeddings
        nh_t, eh_t, gh_t, info_t = trained_encoder(batch)
        nh_t_np = nh_t.detach().cpu().numpy()
        eh_t_np = eh_t.detach().cpu().numpy()
        gh_t_np = gh_t.detach().cpu().numpy()

        untrained_node_embeddings.append(nh_u_np)
        untrained_edge_embeddings.append(eh_u_np)
        untrained_graph_embeddings.append(gh_u_np)
        trained_node_embeddings.append(nh_t_np)
        trained_edge_embeddings.append(eh_t_np)
        trained_graph_embeddings.append(gh_t_np)

        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)
        graph_labels_all.append(graph_labels)

    # Concatenate all embeddings
    untrained_node_embeddings = np.concatenate(untrained_node_embeddings, axis=0)
    untrained_edge_embeddings = np.concatenate(untrained_edge_embeddings, axis=0)
    untrained_graph_embeddings = np.concatenate(untrained_graph_embeddings, axis=0)
    trained_node_embeddings = np.concatenate(trained_node_embeddings, axis=0)
    trained_edge_embeddings = np.concatenate(trained_edge_embeddings, axis=0)
    trained_graph_embeddings = np.concatenate(trained_graph_embeddings, axis=0)

    node_labels_all = np.concatenate(node_labels_all, axis=0)
    edge_labels_all = np.concatenate(edge_labels_all, axis=0)
    graph_labels_all = np.concatenate(graph_labels_all, axis=0)

    unique_node_labels = np.unique(node_labels_all)
    unique_edge_labels = np.unique(edge_labels_all)
    unique_graph_labels = np.unique(graph_labels_all)

    node_cmap = plt.get_cmap('tab10', len(unique_node_labels))
    edge_cmap = plt.get_cmap('tab10', len(unique_edge_labels))
    graph_cmap = plt.get_cmap('tab10', len(unique_graph_labels))

    fig, axes = plt.subplots(2, 3, figsize=(14, 18))

    import time
    from umap import UMAP

    umap_params = {
        'n_components': 2,
        'random_state': 42,
        'n_neighbors': 15,
        'min_dist': 0.1
    }

    # Untrained embeddings UMAP
    print("Node embeddings UMAP untrained...")
    umap_node_untrained = UMAP(**umap_params)
    node_embeddings_umap_untrained = umap_node_untrained.fit_transform(untrained_node_embeddings)
    
    print("Edge embeddings UMAP untrained...")
    umap_edge_untrained = UMAP(**umap_params)
    edge_embeddings_umap_untrained = umap_edge_untrained.fit_transform(untrained_edge_embeddings)

    print("Graph embeddings UMAP untrained...")
    umap_graph_untrained = UMAP(**umap_params)
    graph_embeddings_umap_untrained = umap_graph_untrained.fit_transform(untrained_graph_embeddings)

    # Trained embeddings UMAP
    print("Node embeddings UMAP trained...")
    umap_node_trained = UMAP(**umap_params)
    node_embeddings_umap_trained = umap_node_trained.fit_transform(trained_node_embeddings)

    print("Edge embeddings UMAP trained...")
    umap_edge_trained = UMAP(**umap_params)
    edge_embeddings_umap_trained = umap_edge_trained.fit_transform(trained_edge_embeddings)

    print("Graph embeddings UMAP trained...")
    umap_graph_trained = UMAP(**umap_params)
    graph_embeddings_umap_trained = umap_graph_trained.fit_transform(trained_graph_embeddings)

    # Plot results
    ax = axes[0, 0]
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=node_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Node Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    ax = axes[0, 1]
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=edge_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Edge Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    ax = axes[0, 2]
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=graph_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Graph Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    ax = axes[1, 0]
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_umap_trained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=node_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(trained) Node Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    ax = axes[1, 1]
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_umap_trained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=edge_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(trained) Edge Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    ax = axes[1, 2]
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_umap_trained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=graph_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(trained) Graph Embeddings UMAP')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    plt.tight_layout()
    plt.savefig('./umap.png')
    plt.show()