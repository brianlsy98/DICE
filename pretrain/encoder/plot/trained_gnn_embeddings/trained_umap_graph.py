import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
# Change: import UMAP instead of TSNE
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directories to system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

# Import project-specific modules
from utils import send_to_device
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
        #     test_data.append(neg_graph)  --> Excluding negative pairs
    dataloader = GraphDataLoader(test_data, batch_size=64, shuffle=True)
    print("\nDataset loaded")

    # Load parameters
    params_path = "./params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Initialize and load the trained model
    model_params = params['model']['encoder']['dice']
    trained_encoder = DICE(model_params, args.gnn_depth)
    model_path = (
        f"./pretrain/encoder/saved_models/{params['project_name']}_pretrained_model_"
        f"{model_params['gnn_type']}_depth{args.gnn_depth}.pt"
    )
    trained_encoder.load(model_path)
    trained_encoder = trained_encoder.to('cuda')
    trained_encoder.eval()
    print("\nModel loaded")

    # Initialize lists to store graph-level embeddings and labels
    trained_graph_embeddings = []
    graph_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        umap_batch = send_to_device(batch, 'cuda')

        # Get trained model embeddings
        _, _, gf = trained_encoder(umap_batch)

        # Append data to lists
        trained_graph_embeddings.append(gf.detach().cpu().numpy())
        graph_labels_all.append(umap_batch['circuit'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del trained_encoder
    del dataloader
    del dataset
    del test_data
    del umap_batch
    del gf
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Concatenate all batches
    trained_graph_embeddings = np.concatenate(trained_graph_embeddings, axis=0)
    graph_labels_all = np.concatenate(graph_labels_all, axis=0)

    # Unique labels and colormap
    unique_graph_labels = np.unique(graph_labels_all)
    max_colors = 50
    cmap = plt.get_cmap('hsv', max_colors)

    # Run UMAP (replaces TSNE)
    print("\nGraph embeddings UMAP (trained)...")
    start = time.time()
    umap_graph_trained = UMAP(n_components=2, random_state=42)
    graph_embeddings_umap_trained = umap_graph_trained.fit_transform(trained_graph_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    # Plot UMAP embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_umap_trained[indices]
        color = cmap(i % max_colors)
        ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1], 
            color=color, 
            label=f'Label {label}', 
            s=15
        )

    ax.set_title(f"DICE ({model_params['gnn_type']}) Graph Embeddings UMAP (trained)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    # ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/trained_gnn_embeddings/umap_trained_{model_params['gnn_type']}_depth{args.gnn_depth}_gf.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trained graph embeddings using UMAP')
    parser.add_argument('--gnn_depth', type=int, default=3, help='GNN depth for the Encoder')
    args = parser.parse_args()
    main(args)