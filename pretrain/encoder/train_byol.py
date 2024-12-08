import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader
from model import send_to_device, Encoder




def contrastive_learning_loss(batch, nf_online, ef_online, gf_online,
                                     nf_target, ef_target, gf_target):

    # Compute Similarity
    node_similarity  = torch.mm(F.normalize(nf_online, dim=1),
                                F.normalize(nf_target, dim=1).t())
    edge_similarity  = torch.mm(F.normalize(ef_online, dim=1),
                                F.normalize(ef_target, dim=1).t())
    graph_similarity = torch.mm(F.normalize(gf_online, dim=1),
                                F.normalize(gf_target, dim=1).t())
    ns_exp, es_exp, gs_exp = torch.exp(node_similarity), torch.exp(edge_similarity), torch.exp(graph_similarity)

    nf_log_prob = node_similarity - torch.log(ns_exp.sum(dim=1, keepdim=True))
    ef_log_prob = edge_similarity - torch.log(es_exp.sum(dim=1, keepdim=True))
    gf_log_prob = graph_similarity - torch.log(gs_exp.sum(dim=1, keepdim=True))

    # For node level
    node_labels = batch['node_y'].squeeze()
    node_label_equal = (node_labels.unsqueeze(1) == node_labels.unsqueeze(0))
    node_loss = - (nf_log_prob * node_label_equal.float()).sum() / node_label_equal.float().sum()

    # For edge level
    edge_labels = batch['edge_y'].squeeze()
    edge_label_equal = (edge_labels.unsqueeze(1) == edge_labels.unsqueeze(0))
    edge_loss = - (ef_log_prob * edge_label_equal.float()).sum() / edge_label_equal.float().sum()

    # For graph level
    graph_labels = batch['circuit'].squeeze()
    graph_label_equal = (graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))
    graph_loss = - (gf_log_prob * graph_label_equal.float()).sum() / graph_label_equal.float().sum()

    # loss
    total_loss = node_loss + edge_loss + graph_loss

    return total_loss




def train_model(args):

    dataset = torch.load(f'../dataset/{args.dataset_name}.pt')

    with open(args.params, 'r') as f:
        params = json.load(f)

    ########################
    online_model = Encoder(params['model'])
    target_model = Encoder(params['model'])
    ########################

    # Wandb Initialization
    if args.wandb_log:
        config = {}
        for key, value in params['train'].items():
            config[key] = value
        config["dataset"] = dataset['name']

        wandb.init(
            project="Contrastive Pre-Training",
            config=config
        )

    # Dataset
    train_data = dataset['train_data']
    val_data = dataset['val_data']
    print("train_data size : ", len(train_data))
    print("val_data size : ", len(val_data))

    train_dataloader = GraphDataLoader(train_data, batch_size=params['train']['batch_size'], shuffle=True)
    val_dataloader = GraphDataLoader(val_data, batch_size=params['train']['batch_size'], shuffle=False)

    # Model
    online_model = online_model.to(params['train']["device"])
    online_model = online_model.train()
    target_model = target_model.to(params['train']["device"])
    target_model = target_model.train()


    # Training
    for epoch in range(int(params['train']["epochs"])):

        train_losses = []
        for train_batch in train_dataloader:
            # Sending to Device
            train_batch = send_to_device(train_batch, params['train']["device"])
            ############################
            online_model.optimizer.zero_grad()

            nf_online, ef_online, gf_online, train_online_info =\
                online_model(train_batch)
            nf_target, ef_target, gf_target, train_target_info =\
                target_model(train_batch)
            train_loss = contrastive_learning_loss(
                train_batch,
                nf_online, ef_online, gf_online,
                nf_target.detach(), ef_target.detach(), gf_target.detach()
            )
            train_loss.backward()

            online_model.optimizer.step()
            ############################
            train_losses.append(train_loss.item())

        val_losses = []
        for val_batch in val_dataloader:
            # Sending to Device
            val_batch = send_to_device(val_batch, params['train']["device"])
            ############################
            with torch.no_grad():
                nf_online, ef_online, gf_online, val_online_info =\
                    online_model(val_batch)
                nf_target, ef_target, gf_target, val_target_info =\
                    target_model(val_batch)
                val_loss = contrastive_learning_loss(
                    val_batch,
                    nf_online, ef_online, gf_online,
                    nf_target.detach(), ef_target.detach(), gf_target.detach()
                )
            ############################
            val_losses.append(val_loss.item())

        for param_online, param_target in zip(online_model.parameters(), target_model.parameters()):
            param_target.data = param_online.data * params['train']['tau']\
                              + param_target.data * (1.0 - params['train']['tau'])



        info = {}
        if args.wandb_log:
            info['train_loss'] = sum(train_losses)/len(train_losses)
            info['val_loss'] = sum(val_losses)/len(val_losses)
            wandb.log(info)

        print()
        print(f"Epoch: {epoch}/{params['train']['epochs']}\n\
                Train Loss: {sum(train_losses)/len(train_losses)}\n\
                Validation Loss: {sum(val_losses)/len(val_losses)}")

    if args.wandb_log: wandb.finish()

    online_model.save(f"./saved_models/online_{params['model']['project_name']}.pt")
    target_model.save(f"./saved_models/target_{params['model']['project_name']}.pt")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Contrastive Pre-Training.")
    parser.add_argument("--dataset_name", default="pretraining_dataset_wo_device_params", type=str, help="Name of the dataset directory")
    parser.add_argument("--params", default="./params.json", type=str, help="Path to the params file")
    parser.add_argument("--wandb_log", default=False, type=bool, help="Log to wandb")
    args = parser.parse_args()
    train_model(args)