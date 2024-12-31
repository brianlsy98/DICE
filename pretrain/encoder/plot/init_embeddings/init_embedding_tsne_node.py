import os
import sys
import json
import time
import argparse

import numpy as np
import test
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

# Replace UMAP with TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import init_weights, send_to_device
from dataloader import GraphDataLoader

def main():
    # Load dataset
    dataset_path = './pretrain/dataset/pretraining_dataset_wo_device_params_test.pt'
    dataset = torch.load(dataset_path)
    test_data = []
    for circuit_name, pos_neg_graphs in dataset.items():
        for pos_graph in pos_neg_graphs['pos']:
            test_data.append(pos_graph)
        # for neg_graph in pos_neg_graphs['neg']:
        #     test_data.append(neg_graph)
    dataloader = GraphDataLoader(test_data, batch_size=128, shuffle=True)
    print("\nDataset loaded")

    # Initialize lists to store node-level embeddings and labels
    initial_node_embeddings = []
    node_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        tsne_batch = send_to_device(batch, 'cuda')

        # Append data to lists
        initial_node_embeddings.append(tsne_batch['x'].detach().cpu().numpy())
        node_labels_all.append(tsne_batch['node_y'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del dataloader
    del dataset
    del test_data
    del tsne_batch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate all batches
    initial_node_embeddings = np.concatenate(initial_node_embeddings, axis=0)
    node_labels_all = np.concatenate(node_labels_all, axis=0)

    # Unique labels and colormap (up to 50 colors)
    unique_node_labels = np.unique(node_labels_all)
    max_colors = 9
    node_cmap = plt.get_cmap('hsv', max_colors)

    # Run t-SNE
    print("Node embeddings t-SNE (initial)...")
    start = time.time()
    tsne_node_init = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    node_embeddings_tsne_init = tsne_node_init.fit_transform(initial_node_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    node_types = [
        'gnd', 'vdd', 'voltage_net', 'current_source',
        'nmos', 'pmos', 'resistor', 'capacitor', 'inductor'
    ]
    
    # Plot t-SNE embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_tsne_init[indices]
        color = node_cmap(i % max_colors)

        # Guard against out-of-range labels in node_types
        label_name = node_types[int(label)] if int(label) < len(node_types) else f"Label {label}"

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            color=color,
            label=label_name,
            s=15
        )

    ax.set_title("Node Embeddings t-SNE (Initial)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

    # Save and show the plot
    plot_path = "./pretrain/encoder/plot/init_embeddings/init_embedding_nf.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()
