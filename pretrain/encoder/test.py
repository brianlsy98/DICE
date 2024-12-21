import os
import sys
import json

import numpy as np
from tqdm import tqdm

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader
from model import Encoder as DICE


def cosine_similarity_mean(X, Y=None, chunk_size=10000):
    """
    Compute the mean cosine similarity between every vector in X and every vector in Y.
    If Y is None, we compute the mean of the self-similarities in X.
    To reduce memory usage, we do this in chunks.
    
    Returns: scalar mean similarity.
    """
    if Y is None:
        Y = X

    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    M = X_norm.shape[0]
    N = Y_norm.shape[0]

    # We'll accumulate sum of similarities and count
    sim_sum = 0.0
    sim_count = 0

    # If Y is the same as X, we can compute triangle to avoid double calculation.
    # But since we may not need this optimization right now, let's just compute the full matrix.
    # For large arrays, let's chunk over Y:
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        Y_chunk = Y_norm[start_idx:end_idx]

        # Result: (M, end_idx - start_idx)
        sim_chunk = X_norm @ Y_chunk.T
        sim_sum += sim_chunk.sum()
        sim_count += sim_chunk.size

    mean_sim = sim_sum / sim_count
    return mean_sim


def cosine_similarity_mean_self(X, chunk_size=10000):
    """
    Compute the mean cosine similarity of X with itself.
    This includes the diagonal where similarity=1.0.
    If needed, we can exclude it.
    """
    # Same approach, but we only need to do upper-triangular computations to save on memory if desired.
    # For simplicity, let's just do the full computation and then adjust if needed.

    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    M = X_norm.shape[0]

    sim_sum = 0.0
    sim_count = 0

    for start_idx in range(0, M, chunk_size):
        end_idx = min(start_idx + chunk_size, M)
        X_chunk = X_norm[start_idx:end_idx]
        sim_chunk = X_chunk @ X_norm.T
        sim_sum += sim_chunk.sum()
        sim_count += sim_chunk.size

    mean_sim = sim_sum / sim_count
    return mean_sim


def compute_labelwise_similarities(embeddings, labels, label_type="Node", chunk_size=10000):
    unique_labels = np.unique(labels)

    results = []
    for lbl in tqdm(unique_labels):
        positive_indices = np.where(labels == lbl)[0]
        negative_indices = np.where(labels != lbl)[0]

        pos_emb = embeddings[positive_indices]
        neg_emb = embeddings[negative_indices]

        if len(pos_emb) == 0 or len(neg_emb) == 0:
            continue

        # Positive-Positive similarity mean
        pp_mean = cosine_similarity_mean_self(pos_emb, chunk_size=chunk_size)

        # Positive-Negative similarity mean
        pn_mean = cosine_similarity_mean(pos_emb, neg_emb, chunk_size=chunk_size)

        results.append((lbl, pp_mean, pn_mean))

    # Print aggregated results
    print(f"{label_type} Labels")
    for lbl, pp_mean, pn_mean in results:
        print(f"Label: {lbl}")
        print("  Positive-Positive similarity:", pp_mean)
        print("  Positive-Negative similarity:", pn_mean)


if __name__ == "__main__":
    # Load Parameters
    with open(f"./params.json", 'r') as f:
        params = json.load(f)

    # Load dataset
    dataset = torch.load('./pretrain/dataset/pretraining_dataset_wo_device_params.pt')
    test_data = dataset['test_data']
    dataloader = GraphDataLoader(test_data, batch_size=params['pretraining']['test']['batch_size'], shuffle=True)
    print("Dataset loaded")

    # Initialize models
    untrained_encoder = DICE(params['model']['dice'])
    trained_encoder = DICE(params['model']['dice'])
    trained_encoder.load(f"./pretrain/encoder/saved_models/{params['project_name']}_pretrained_model.pt")
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

    for batch in tqdm(dataloader):
        # Extract batch data
        nf = batch['x']
        ef = batch['edge_attr']

        node_labels = batch['node_y'].detach().cpu().numpy()
        edge_labels = batch['edge_y'].detach().cpu().numpy()
        graph_labels = batch['circuit'].detach().cpu().numpy()

        # Compute embeddings from untrained model
        nh_u, eh_u, gh_u, info_u = untrained_encoder(batch)
        nh_u_np = nh_u.detach().cpu().numpy()
        eh_u_np = eh_u.detach().cpu().numpy()
        gh_u_np = gh_u.detach().cpu().numpy()

        # Compute embeddings from trained model
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

    print("\nCosine Similarity\n")

    chunk_size = 10000  # Adjust this as needed based on your memory constraints

    # Node similarities
    print("Node Label Similarities (Untrained)")
    compute_labelwise_similarities(untrained_node_embeddings, node_labels_all, label_type="Node (Untrained)", chunk_size=chunk_size)

    print("\nNode Label Similarities (Trained)")
    compute_labelwise_similarities(trained_node_embeddings, node_labels_all, label_type="Node (Trained)", chunk_size=chunk_size)

    # Edge similarities
    print("\nEdge Label Similarities (Untrained)")
    compute_labelwise_similarities(untrained_edge_embeddings, edge_labels_all, label_type="Edge (Untrained)", chunk_size=chunk_size)

    print("\nEdge Label Similarities (Trained)")
    compute_labelwise_similarities(trained_edge_embeddings, edge_labels_all, label_type="Edge (Trained)", chunk_size=chunk_size)

    # Graph similarities
    print("\nGraph Label Similarities (Untrained)")
    compute_labelwise_similarities(untrained_graph_embeddings, graph_labels_all, label_type="Graph (Untrained)", chunk_size=chunk_size)

    print("\nGraph Label Similarities (Trained)")
    compute_labelwise_similarities(trained_graph_embeddings, graph_labels_all, label_type="Graph (Trained)", chunk_size=chunk_size)

    print()