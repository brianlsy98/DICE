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
    initial_graph_embeddings = []
    graph_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        umap_batch = send_to_device(batch, 'cuda')
        nf, ef, edge_index = umap_batch['x'], umap_batch['edge_attr'], umap_batch['edge_index']
        nf, ef = F.pad(nf, (0, 5)), F.pad(ef, (0, 9))

        gf = scatter_add(nf, umap_batch['batch'], dim=0,
                         dim_size=umap_batch['batch'].max().item() + 1)\
            + scatter_add(ef, umap_batch['batch'][edge_index[0]], dim=0,
                          dim_size=umap_batch['batch'][edge_index[0]].max().item() + 1)

        # Append data to lists
        initial_graph_embeddings.append(gf.detach().cpu().numpy())
        graph_labels_all.append(umap_batch['circuit'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del dataloader
    del dataset
    del test_data
    del umap_batch
    del gf
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate all batches
    initial_graph_embeddings = np.concatenate(initial_graph_embeddings, axis=0)
    graph_labels_all = np.concatenate(graph_labels_all, axis=0)

    # Unique labels and colormap
    unique_graph_labels = np.unique(graph_labels_all)
    graph_cmap = plt.get_cmap('tab10', len(unique_graph_labels))

    # Run UMAP (instead of TSNE)
    print("\nGraph embeddings UMAP (initial)...")
    start = time.time()
    umap_graph_init = UMAP(n_components=2, random_state=42)
    graph_embeddings_umap_init = umap_graph_init.fit_transform(initial_graph_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    # Plot UMAP embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_umap_init[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                   color=graph_cmap(i), label=f'Label {label}', s=15)

    ax.set_title(f"Graph Embeddings UMAP (Initial)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/init_embeddings/init_embedding_gf.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    main()
