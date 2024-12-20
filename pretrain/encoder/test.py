import os
import sys
import json

import numpy as np
from tqdm import tqdm

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader
from model import Encoder

def cosine_similarity_matrix(X, Y=None):
    """
    Compute the cosine similarity matrix between two sets of vectors.
    If Y is None, compute self-similarity of X.
    X: (M, D)
    Y: (N, D) or None
    Returns: (M, N) similarity matrix
    """
    if Y is None:
        Y = X
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X_norm @ Y_norm.T

def compute_labelwise_similarities(embeddings, labels, label_type="Node"):
    unique_labels = np.unique(labels)

    # We'll store results and print after computing
    # to avoid printing inside a loop.
    results = []
    for lbl in tqdm(unique_labels):
        positive_indices = np.where(labels == lbl)[0]
        negative_indices = np.where(labels != lbl)[0]

        pos_emb = embeddings[positive_indices]
        neg_emb = embeddings[negative_indices]

        if len(pos_emb) == 0 or len(neg_emb) == 0:
            continue

        # Positive-Positive similarity
        pp_sim = cosine_similarity_matrix(pos_emb, pos_emb)
        # Positive-Negative similarity
        pn_sim = cosine_similarity_matrix(pos_emb, neg_emb)

        # We compute the mean of all entries for pp and pn.
        # Note that pp includes the diagonal (self-similarity = 1.0).
        # If you want to exclude the diagonal, you could mask it out.
        pp_mean = np.mean(pp_sim)
        pn_mean = np.mean(pn_sim)

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

    model_params = params['model']
    test_params = params['test']

    # Load dataset
    dataset = torch.load('../dataset/pretraining_dataset_wo_device_params.pt')
    test_data = dataset['test_data']
    dataloader = GraphDataLoader(test_data, batch_size=test_params['batch_size'], shuffle=True)
    print("Dataset loaded")

    # Initialize models
    untrained_encoder = Encoder(model_params)
    trained_encoder = Encoder(model_params)
    trained_encoder.load(f"./saved_models/dice_pretraining.pt")
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

    # Node similarities
    print("Node Label Similarities (Untrained)")
    compute_labelwise_similarities(untrained_node_embeddings, node_labels_all, label_type="Node (Untrained)")

    print("\nNode Label Similarities (Trained)")
    compute_labelwise_similarities(trained_node_embeddings, node_labels_all, label_type="Node (Trained)")

    # Edge similarities (uncomment if needed)
    # print("\nEdge Label Similarities (Untrained)")
    # compute_labelwise_similarities(untrained_edge_embeddings, edge_labels_all, label_type="Edge (Untrained)")

    # print("\nEdge Label Similarities (Trained)")
    # compute_labelwise_similarities(trained_edge_embeddings, edge_labels_all, label_type="Edge (Trained)")

    # Graph similarities
    print("\nGraph Label Similarities (Untrained)")
    compute_labelwise_similarities(untrained_graph_embeddings, graph_labels_all, label_type="Graph (Untrained)")

    print("\nGraph Label Similarities (Trained)")
    compute_labelwise_similarities(trained_graph_embeddings, graph_labels_all, label_type="Graph (Trained)")

    print()