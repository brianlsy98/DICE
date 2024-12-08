import os
import sys
import json

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader
from model import Encoder

if __name__ == "__main__":

    # Load dataset
    dataset = torch.load('../dataset/pretraining_dataset_wo_device_params.pt')
    test_data = dataset['test_data']
    dataloader = GraphDataLoader(test_data, batch_size=256, shuffle=True)
    print()
    print("Dataset loaded")

    # Load Parameters
    with open(f"./params.json", 'r') as f:
        model_params = json.load(f)['model']

    # Initialize both an untrained and trained model
    untrained_encoder = Encoder(model_params)
    trained_encoder = Encoder(model_params)
    trained_encoder.load(f"./saved_models/dice_pretraining.pt")
    print()
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
        edge_i = batch['edge_index']
        ef = batch['edge_attr']

        node_labels = batch['node_y'].detach().cpu().numpy()
        edge_labels = batch['edge_y'].detach().cpu().numpy()
        graph_labels = batch['circuit'].detach().cpu().numpy()

        # Initial embeddings
        nf_np = nf.detach().cpu().numpy()
        ef_np = ef.detach().cpu().numpy()

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

        # Append to lists
        untrained_node_embeddings.append(nh_u_np)
        untrained_edge_embeddings.append(eh_u_np)
        untrained_graph_embeddings.append(gh_u_np)
        trained_node_embeddings.append(nh_t_np)
        trained_edge_embeddings.append(eh_t_np)
        trained_graph_embeddings.append(gh_t_np)

        node_labels_all.append(node_labels)
        edge_labels_all.append(edge_labels)
        graph_labels_all.append(graph_labels)


    # Concatenate all batches
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

    # Create figure and axes for subplots: 3 rows x 2 columns
    fig, axes = plt.subplots(2, 3, figsize=(14, 18))
    # axes[0,0]: Untrained GNN Node Embeddings
    # axes[0,1]: Untrained GNN Edge Embeddings
    # axes[1,0]: Trained GNN Node Embeddings
    # axes[1,1]: Trained GNN Edge Embeddings

    # Run TSNE
    import time

    print()
    start = time.time()
    print("Node embeddings tsne untrained...")
    tsne_node_untrained = TSNE(n_components=2, random_state=42)
    node_embeddings_tsne_untrained = tsne_node_untrained.fit_transform(untrained_node_embeddings)
    end = time.time()
    print(f"done in {end-start:.2f} seconds")

    start = time.time()
    print("Edge embeddings tsne untrained...")
    tsne_edge_untrained = TSNE(n_components=2, random_state=42)
    edge_embeddings_tsne_untrained = tsne_edge_untrained.fit_transform(untrained_edge_embeddings)
    end = time.time()
    print(f"done in {end-start:.2f} seconds")

    start = time.time()
    print("Graph embeddings tsne untrained...")
    tsne_graph_untrained = TSNE(n_components=2, random_state=42)
    graph_embeddings_tsne_untrained = tsne_graph_untrained.fit_transform(untrained_graph_embeddings)
    end = time.time()
    print(f"done in {end-start:.2f} seconds")

    start = time.time()
    print("Node embeddings tsne trained...")
    tsne_node_trained = TSNE(n_components=2, random_state=42)
    node_embeddings_tsne_trained = tsne_node_trained.fit_transform(trained_node_embeddings)
    end = time.time()
    print(f"done in {end-start:.2f} seconds")

    print("Edge embeddings tsne trained...")
    start = time.time()
    tsne_edge_trained = TSNE(n_components=2, random_state=42)
    edge_embeddings_tsne_trained = tsne_edge_trained.fit_transform(trained_edge_embeddings)
    end = time.time()
    print(f"done in {end-start:.2f} seconds")

    print("Graph embeddings tsne trained...")
    start = time.time()
    tsne_graph_trained = TSNE(n_components=2, random_state=42)
    graph_embeddings_tsne_trained = tsne_graph_trained.fit_transform(trained_graph_embeddings)
    end = time.time()
    print(f"done in {end-start:.2f} seconds")


    # Plot: Untrained GNN Node Embeddings
    ax = axes[0, 0]
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_tsne_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=node_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Node Embeddings t-SNE')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend(fontsize='small')

    # Plot: Untrained GNN Edge Embeddings
    ax = axes[0, 1]
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_tsne_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=edge_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Edge Embeddings t-SNE')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend(fontsize='small')
    
    # Plot: Untrained GNN Graph Embeddings
    ax = axes[0, 2]
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_tsne_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=graph_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('(untrained) Graph Embeddings t-SNE')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend(fontsize='small')



    # Plot: Trained GNN Node Embeddings
    ax = axes[1, 0]
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_tsne_trained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=node_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('t-SNE of (trained) GNN-passed Node Embeddings')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend(fontsize='small')

    # Plot: Trained GNN Edge Embeddings
    ax = axes[1, 1]
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_tsne_trained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=edge_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('t-SNE of (trained) GNN-passed Edge Embeddings')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend(fontsize='small')

    # Plot: Trained GNN Graph Embeddings
    ax = axes[1, 2]
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_tsne_trained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=graph_cmap(i), label=f'Label {label}', s=15)
    ax.set_title('t-SNE of (trained) GNN-passed Graph Embeddings')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend(fontsize='small')


    plt.tight_layout()
    plt.show()
