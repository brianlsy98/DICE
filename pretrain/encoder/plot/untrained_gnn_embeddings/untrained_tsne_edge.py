import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
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
        tsne_batch = send_to_device(batch, 'cuda')

        # Get untrained model embeddings
        _, ef, _ = untrained_encoder(tsne_batch)

        untrained_edge_embeddings.append(ef.detach().cpu().numpy())
        edge_labels_all.append(tsne_batch['edge_y'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del untrained_encoder
    del dataloader
    del dataset
    del test_data
    del tsne_batch
    del ef
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate all batches
    untrained_edge_embeddings = np.concatenate(untrained_edge_embeddings, axis=0)
    edge_labels_all = np.concatenate(edge_labels_all, axis=0)

    # Unique labels and colormap (up to 50 colors)
    unique_edge_labels = np.unique(edge_labels_all)
    max_colors = 5
    edge_cmap = plt.get_cmap('hsv', max_colors)

    # Run t-SNE
    print("\nEdge embeddings t-SNE (untrained)...")
    start = time.time()
    tsne_edge_untrained = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    edge_embeddings_tsne_untrained = tsne_edge_untrained.fit_transform(untrained_edge_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    edge_types = ['current_net', 'v2ng', 'v2pg', 'v2nb', 'v2pb']
    # Plot t-SNE embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_edge_labels):
        indices = np.where(edge_labels_all == label)
        embeddings = edge_embeddings_tsne_untrained[indices]
        color = edge_cmap(i % max_colors)

        label_name = edge_types[label] if label < len(edge_types) else f"Label {label}"

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            color=color,
            label=label_name,
            s=15
        )

    ax.set_title(f"DICE ({model_params['gnn_type']}) Edge Embeddings t-SNE (untrained)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/untrained_gnn_embeddings/tsne_untrained_{model_params['gnn_type']}_depth{args.gnn_depth}_ef.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot untrained edge embeddings using t-SNE')
    parser.add_argument('--gnn_depth', type=int, default=3, help='GNN depth for the Encoder')
    args = parser.parse_args()
    main(args)
