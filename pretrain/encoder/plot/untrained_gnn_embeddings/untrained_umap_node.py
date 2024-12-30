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

    # Initialize lists to store embeddings and labels
    untrained_node_embeddings = []
    node_labels_all = []

    # Process batches and collect embeddings
    for batch in tqdm(dataloader, desc='Processing Batches'):
        umap_batch = send_to_device(batch, 'cuda')
        
        # Get trained model embeddings
        nf, _, _ = untrained_encoder(umap_batch)

        untrained_node_embeddings.append(nf.detach().cpu().numpy())
        node_labels_all.append(umap_batch['node_y'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del untrained_encoder  # Remove reference to your model
    del dataloader         # Remove reference to your dataloader
    del dataset            # Remove reference to your dataset
    del test_data          # Remove reference to your test data
    del umap_batch         # Remove reference to your batch
    del nf                 # Remove reference to your graph embeddings
    import gc
    gc.collect()           # Run Python garbage collector
    torch.cuda.empty_cache()  # Clear PyTorch’s CUDA cache
    
    # Concatenate all batches
    untrained_node_embeddings = np.concatenate(untrained_node_embeddings, axis=0)
    node_labels_all = np.concatenate(node_labels_all, axis=0)

    # Unique labels and colormap
    unique_node_labels = np.unique(node_labels_all)
    node_cmap = plt.get_cmap('tab10', len(unique_node_labels))

    # Run UMAP (replaces TSNE)
    print("\nNode embeddings UMAP untrained...")
    start = time.time()
    umap_node_untrained = UMAP(n_components=2, random_state=42)
    node_embeddings_umap_untrained = umap_node_untrained.fit_transform(untrained_node_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    node_types = ['gnd', 'vdd', 'voltage_net', 'current_source',
                  'nmos', 'pmos', 'resistor', 'capacitor', 'inductor']
    
    # Plot UMAP embeddings
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(unique_node_labels):
        indices = np.where(node_labels_all == label)
        embeddings = node_embeddings_umap_untrained[indices]
        ax.scatter(embeddings[:, 0], embeddings[:, 1], color=node_cmap(i), 
                   label=f'{node_types[label]}', s=15)

    ax.set_title(f"DICE ({model_params['gnn_type']}) Node Embeddings UMAP (untrained)")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/untrained_gnn_embeddings/umap_untrained_{model_params['gnn_type']}_depth{args.gnn_depth}_nf.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot untrained graph embeddings using UMAP')
    parser.add_argument('--gnn_depth', type=int, default=3, help='GNN depth for the Encoder')
    args = parser.parse_args()
    main(args)
