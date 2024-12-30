import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

from umap import UMAP
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


    # Initialize lists to store graph-level embeddings and labels
    initial_edge_embeddings = []
    edge_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        umap_batch = send_to_device(batch, 'cuda')

        # Append data to lists
        initial_edge_embeddings.append(umap_batch['edge_attr'].detach().cpu().numpy())
        edge_labels_all.append(umap_batch['edge_y'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del dataloader
    del dataset
    del test_data
    del umap_batch
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate all batches
    initial_edge_embeddings = np.concatenate(initial_edge_embeddings, axis=0)
    edge_labels_all = np.concatenate(edge_labels_all, axis=0)

    # Unique labels and colormap
    unique_edge_labels = np.unique(edge_labels_all)
    edge_cmap = plt.get_cmap('tab10', len(unique_edge_labels))

    # Run UMAP (instead of TSNE)
    print("Edge embeddings UMAP (initial)...")
    start = time.time()
    umap_edge_init = UMAP(n_components=2, random_state=42)
    edge_embeddings_umap_init = umap_edge_init.fit_transform(initial_edge_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    edge_types = ['current_net', 'v2ng', 'v2pg', 'v2nb', 'v2pb']
    # Plot UMAP embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_umap_init[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                   color=edge_cmap(i), label=f'{edge_types[int(label)]}', s=15)

    ax.set_title(f"Edge Embeddings UMAP (Initial)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/init_embeddings/init_embedding_ef.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()
