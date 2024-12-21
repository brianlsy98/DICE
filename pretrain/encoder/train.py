import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import send_to_device
from dataloader import GraphDataLoader
from model import DICE



def contrastive_learning_loss(batch, nf, ef, gf):

    # for training with same number of label pairs
    def sample_indices(labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        min_count = counts.min().item()

        indices = []
        for lbl in unique_labels:
            lbl_indices = torch.where(labels == lbl)[0]
            perm = torch.randperm(len(lbl_indices))
            lbl_indices = lbl_indices[perm[:min_count]]
            indices.append(lbl_indices)
        indices = torch.cat(indices, dim=0)
        indices = indices[torch.randperm(len(indices))]

        return indices

    node_indices = sample_indices(batch['node_y'])
    edge_indices = sample_indices(batch['edge_y'])
    graph_indices = sample_indices(batch['circuit'])

    node_labels, node_features = batch['node_y'][node_indices], nf[node_indices]
    edge_labels, edge_features = batch['edge_y'][edge_indices], ef[edge_indices]
    graph_labels, graph_features = batch['circuit'][graph_indices], gf[graph_indices]

    # Compute Cosine Similarity
    node_similarity  = torch.mm(F.normalize(node_features, dim=1),
                                F.normalize(node_features, dim=1).t())
    edge_similarity  = torch.mm(F.normalize(edge_features, dim=1),
                                F.normalize(edge_features, dim=1).t())
    graph_similarity = torch.mm(F.normalize(graph_features, dim=1),
                                F.normalize(graph_features, dim=1).t())

    node_label_equal = (node_labels.unsqueeze(1) == node_labels.unsqueeze(0))
    edge_label_equal = (edge_labels.unsqueeze(1) == edge_labels.unsqueeze(0))
    graph_label_equal = (graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))

    ns_exp = torch.exp(node_similarity).masked_fill(node_label_equal, 0)
    es_exp = torch.exp(edge_similarity).masked_fill(edge_label_equal, 0)
    gs_exp = torch.exp(graph_similarity).masked_fill(graph_label_equal, 0)

    nf_log_prob = node_similarity - torch.log(ns_exp.sum(dim=1, keepdim=True))
    ef_log_prob = edge_similarity - torch.log(es_exp.sum(dim=1, keepdim=True))
    gf_log_prob = graph_similarity - torch.log(gs_exp.sum(dim=1, keepdim=True))

    # loss
    node_loss = - (nf_log_prob * node_label_equal.float()).mean()
    edge_loss = - (ef_log_prob * edge_label_equal.float()).mean()
    graph_loss = - (gf_log_prob * graph_label_equal.float()).mean()

    total_loss = node_loss + edge_loss + graph_loss

    return total_loss



def train_model(args):

    dataset = torch.load(f'./pretrain/dataset/{args.dataset_name}.pt')

    with open(args.params, 'r') as f:
        params = json.load(f)

    ########################
    model = DICE(params['model']['encoder']['dice'])
    model.optimizer = torch.optim.Adam(model.parameters(), lr=params['pretraining']['train']['lr'])
    ########################

    # Wandb Initialization
    if args.wandb_log:
        config = {}
        for key, value in params['pretraining']['train'].items():
            config[key] = value
        config["dataset"] = dataset['name']

        wandb.init(
            project=params['project_name'],
            name="dice_pretraining",
            config=config
        )

    # Dataset
    train_data = dataset['train_data']
    val_data = dataset['val_data']
    print("train_data size : ", len(train_data))
    print("val_data size : ", len(val_data))

    train_dataloader = GraphDataLoader(train_data, batch_size=params['pretraining']['train']['batch_size'], shuffle=True)
    val_dataloader = GraphDataLoader(val_data, batch_size=params['pretraining']['train']['batch_size'], shuffle=False)

    # Model
    model = model.to(params['pretraining']['train']["device"])
    model = model.train()

    # Training
    for epoch in range(int(params['pretraining']['train']["epochs"])):

        train_losses = []
        for train_batch in train_dataloader:
            # Sending to Device
            train_batch = send_to_device(train_batch, params['pretraining']['train']["device"])
            ############################
            model.optimizer.zero_grad()
            nf, ef, gf = model(train_batch)
            train_loss = contrastive_learning_loss(train_batch, nf, ef, gf)
            train_loss.backward()
            model.optimizer.step()
            ############################
            train_losses.append(train_loss.item())

        val_losses = []
        for val_batch in val_dataloader:
            # Sending to Device
            val_batch = send_to_device(val_batch, params['pretraining']['train']["device"])
            ############################
            with torch.no_grad():
                nf, ef, gf = model(val_batch)
                val_loss = contrastive_learning_loss(val_batch, nf, ef, gf)
            ############################
            val_losses.append(val_loss.item())

        info = {}
        if args.wandb_log:
            info['dice_pretraining_train_loss'] = sum(train_losses)/len(train_losses)
            info['dice_pretraining_val_loss'] = sum(val_losses)/len(val_losses)
            wandb.log(info)

        print()
        print(f"Epoch: {epoch}/{params['pretraining']['train']['epochs']}\n\
                Train Loss: {sum(train_losses)/len(train_losses)}\n\
                Validation Loss: {sum(val_losses)/len(val_losses)}")

        if epoch % (int(params['pretraining']['train']['epochs']) // 5) == 0:
            model.save(f"./pretrain/encoder/saved_models"\
                       f"/{params['project_name']}_pretrained_model_{epoch}"\
                       f"_{params['model']['encoder']['dice']['gnn_type']}.pt")


    if args.wandb_log: wandb.finish()

    model.save(f"./pretrain/encoder/saved_models/"\
               f"/{params['project_name']}_pretrained_model"\
               f"_{params['model']['encoder']['dice']['gnn_type']}.pt")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DICE Pre-Training.")
    parser.add_argument("--dataset_name", default="pretraining_dataset_wo_device_params", type=str, help="Name of the dataset directory")
    parser.add_argument("--params", default="./params.json", type=str, help="Path to the params file")
    parser.add_argument("--wandb_log", default=False, type=bool, help="Log to wandb")
    args = parser.parse_args()
    train_model(args)