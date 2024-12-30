import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
# Replace TSNE with UMAP
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import init_weights, send_to_device
from dataloader import GraphDataLoader
from model import DICE

def main(args):
    # Load dataset
    dataset_path = './pretrain/dataset/pretraining_dataset_wo_device_params_test.pt'
    dataset = torch.load(dataset_path)
    test_data = []
    for circuit_name, pos_neg_graphs in dataset.items():
        for pos_graph in pos_neg_graphs['pos']:
            test_data.append(pos_graph)
        # for neg_graph in pos_neg_graphs['neg']:
        #     test_data.append(neg_graph)
    random.shuffle(test_data)
    test_data = test_data[:]
    dataloader = GraphDataLoader(test_data, batch_size=128, shuffle=True)
    print("\nDataset loaded")

    # Load parameters
    params_path = "./params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Initialize
    model_params = params['model']['encoder']['dice']
    untrained_encoder = DICE(model_params, args.gnn_depth)
    untrained_encoder.apply(init_weights)
    untrained_encoder = untrained_encoder.to('cuda')
    untrained_encoder.eval()
    print("\nModel loaded")

    # Initialize lists to store edge-level embeddings and labels
    untrained_edge_embeddings = []
    edge_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        umap_batch = send_to_device(batch, 'cuda')

        # Get untrained model embeddings
        _, ef, _ = untrained_encoder(umap_batch)

        # Append data to lists
        untrained_edge_embeddings.append(ef.detach().cpu().numpy())
        edge_labels_all.append(umap_batch['edge_y'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del untrained_encoder
    del dataloader
    del dataset
    del test_data
    del umap_batch
    del ef
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate all batches
    untrained_edge_embeddings = np.concatenate(untrained_edge_embeddings, axis=0)
    edge_labels_all = np.concatenate(edge_labels_all, axis=0)

    # Unique labels and colormap
    unique_edge_labels = np.unique(edge_labels_all)
    edge_cmap = plt.get_cmap('tab10', len(unique_edge_labels))

    # Run UMAP (instead of TSNE)
    print("\Edge embeddings UMAP (untrained)...")
    start = time.time()
    umap_graph_untrained = UMAP(n_components=2, random_state=42)
    edge_embeddings_umap_untrained = umap_graph_untrained.fit_transform(untrained_edge_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    edge_types = ['current_net', 'v2ng', 'v2pg', 'v2nb', 'v2pb']
    # Plot UMAP embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                   color=edge_cmap(i), label=f'{edge_types[label]}', s=15)

    ax.set_title(f"DICE ({model_params['gnn_type']}) Graph Embeddings UMAP (untrained)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/untrained_gnn_embeddings/umap_untrained_{model_params['gnn_type']}_depth{args.gnn_depth}_ef.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot untrained graph embeddings using UMAP')
    parser.add_argument('--gnn_depth', type=int, default=3, help='GNN depth for the Encoder')
    args = parser.parse_args()
    main(args)
