import os
import sys
import json
import time
import argparse

import numpy as np
import torch
# Change: import TSNE instead of UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directories to system path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

# Import project-specific modules
from utils import send_to_device, set_seed
from dataloader import GraphDataLoader
from model import DICE

def main(args):
    # Set seed for reproducibility
    set_seed(7)

    dataset_path = './pretrain/dataset/pretraining_dataset_wo_device_params_test_pda.pt'
    dataset = torch.load(dataset_path)
    test_data = []
    for circuit_name, pos_neg_graphs in dataset.items():
        for pos_graph in pos_neg_graphs['pos']:
            test_data.append(pos_graph)
    dataloader = GraphDataLoader(test_data, batch_size=64, shuffle=True)
    print("\nDataset loaded")

    # Load parameters
    params_path = "./params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    taup = f"{args.taup}".replace(".", "")
    tau = f"{args.tau}".replace(".", "")
    taun = f"{args.taun}".replace(".", "")

    # Initialize and load the trained model
    model_params = params['model']['encoder']['dice']
    trained_encoder = DICE(model_params, args.gnn_depth)
    model_path = (
        f"./pretrain/encoder/{params['project_name']}_pretrained_model_"
        f"{model_params['gnn_type']}_depth{args.gnn_depth}_taup{taup}tau{tau}taun{taun}.pt"
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
        tsne_batch = send_to_device(batch, 'cuda')

        # Get trained model embeddings
        _, _, gf = trained_encoder(tsne_batch)

        # Append data to lists
        trained_graph_embeddings.append(gf.detach().cpu().numpy())
        graph_labels_all.append(tsne_batch['circuit'].detach().cpu().numpy())

    # Free GPU memory
    print("\nFreeing up GPU memory...")
    del trained_encoder
    del dataloader
    del dataset
    del test_data
    del tsne_batch
    del gf
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Concatenate all batches
    trained_graph_embeddings = np.concatenate(trained_graph_embeddings, axis=0)
    graph_labels_all = np.concatenate(graph_labels_all, axis=0)

    # Identify unique labels
    unique_graph_labels = np.unique(graph_labels_all)
    num_labels = len(unique_graph_labels)
    print(f"Found {num_labels} unique labels.")

    # Use a colormap that can comfortably handle up to 50 labels
    # Here, 'hsv' is used with 50 discrete bins. If you have more than 50 labels, colors will repeat.
    max_colors = 55
    cmap = plt.get_cmap('hsv', max_colors)

    # Run t-SNE
    print(f"\nGraph embeddings t-SNE (trained, taup{taup}, tau{tau}, taun{taun})...")
    start = time.time()
    tsne = TSNE(n_components=2, random_state=98, perplexity=30, max_iter=1500)
    graph_embeddings_tsne_trained = tsne.fit_transform(trained_graph_embeddings)
    end = time.time()
    print(f"done in {end - start:.2f} seconds")

    # Plot t-SNE embeddings
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, label in enumerate(unique_graph_labels):
        indices = np.where(graph_labels_all == label)
        embeddings = graph_embeddings_tsne_trained[indices]
        # Map the label index to a color, wrapping around if we exceed max_colors
        color = cmap(i % max_colors)

        ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1], 
            color=color, 
            label=f'Label {label}', 
            s=15
        )

    # ax.set_title(f"DICE ({model_params['gnn_type']}) Graph Embeddings t-SNE (trained)")
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')

    # Optionally enable the legend if you want to see label names
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save and show the plot
    plot_path = f"./pretrain/encoder/plot/trained_gnn_embeddings/tsne_trained_{model_params['gnn_type']}_depth{args.gnn_depth}_taup{taup}tau{tau}taun{taun}_gf.png"
    plt.tight_layout()
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    plt.savefig(plot_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trained graph embeddings using t-SNE')
    parser.add_argument('--taup', type=str, default="0.2", help='Temperature parameter for contrastive loss (positive samples)')
    parser.add_argument('--tau', type=str, default="0.05", help='Temperature parameter for contrastive loss (anchor samples)')
    parser.add_argument('--taun', type=str, default="0.05", help='Temperature parameter for contrastive loss (negative samples)')
    parser.add_argument('--gnn_depth', type=int, default=2, help='GNN depth for the Encoder')
    args = parser.parse_args()
    main(args)
