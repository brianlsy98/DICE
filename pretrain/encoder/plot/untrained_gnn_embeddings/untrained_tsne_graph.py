import os
import sys
import json
import time
import argparse

import numpy as np
import torch
# Replace UMAP with t-SNE
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

    # Initialize lists to store graph-level embeddings and labels
    untrained_graph_embeddings = []
    graph_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        tsne_batch = send_to_device(batch, 'cuda')

        # Get untrained model embeddings
        _, _, gf = untrained_encoder(tsne_batch)

        # Append data to lists
        untrained_graph_embeddings.append(gf.detach().cpu().numpy())
        graph_labels_all.append(tsne_batch['circuit'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del untrained_encoder
    del dataloader
    del dataset
    del test_data
    del tsne_batch
    del gf
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Concatenate all batches
    untrained_graph_embeddings = np.concatenate(untrained_graph_embeddings, axis=0)
    graph_labels_all = np.concatenate(graph_labels_all, axis=0)

    # Get unique labels
    unique_graph_labels = np.unique(graph_labels_all)
    num_labels = len(unique_graph_labels)
    print(f"Found {num_labels} unique labels.")

    # Use a colormap that can comfortably handle up to 50 labels
    max_colors = 50
    cmap = plt.get_cmap('hsv', max_colors)

    # Run t-SNE
    print("\nGraph embeddings t-SNE (untrained)...")
    start = time.time()
    tsne_graph_untrained = TSNE(n_components=2, random_state=98, perplexity=50, max_iter=3000)
    graph_embeddings_tsne_untrained = tsne_graph_untrained.fit_transform(untrained_graph_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    # Plot t-SNE embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_tsne_untrained[indices]

        # Cycle through up to 50 colors
        color = cmap(i % max_colors)

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            color=color,
            label=f'Label {label}',
            s=15
        )

    # ax.set_title(f"DICE ({model_params['gnn_type']}) Graph Embeddings t-SNE (untrained)")
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')

    # If you want to show a legend of labels (note that 50 might be quite large):
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save and show the plot
    plot_path = (
        f"./pretrain/encoder/plot/untrained_gnn_embeddings/"
        f"tsne_untrained_{model_params['gnn_type']}_depth{args.gnn_depth}_gf.png"
    )
    plt.tight_layout()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot untrained graph embeddings using t-SNE with up to 50 labels')
    parser.add_argument('--gnn_depth', type=int, default=2, help='GNN depth for the Encoder')
    args = parser.parse_args()
    main(args)
