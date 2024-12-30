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
    initial_node_embeddings = []
    node_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        umap_batch = send_to_device(batch, 'cuda')

        # Append data to lists
        initial_node_embeddings.append(umap_batch['x'].detach().cpu().numpy())
        node_labels_all.append(umap_batch['node_y'].detach().cpu().numpy())

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
    initial_node_embeddings = np.concatenate(initial_node_embeddings, axis=0)
    node_labels_all = np.concatenate(node_labels_all, axis=0)

    # Unique labels and colormap
    unique_node_labels = np.unique(node_labels_all)
    node_cmap = plt.get_cmap('tab10', len(unique_node_labels))

    # Run UMAP (instead of TSNE)
    print("Node embeddings UMAP (initial)...")
    start = time.time()
    umap_node_init = UMAP(n_components=2, random_state=42)
    node_embeddings_umap_init = umap_node_init.fit_transform(initial_node_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    node_types = ['gnd', 'vdd', 'voltage_net', 'current_source',
                  'nmos', 'pmos', 'resistor', 'capacitor', 'inductor']
    
    # Plot UMAP embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_umap_init[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                   color=node_cmap(i), label=f'{node_types[int(label)]}', s=15)

    ax.set_title(f"Node Embeddings UMAP (Initial)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/init_embeddings/init_embedding_nf.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()
