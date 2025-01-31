import os
import sys
import json
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.amp import autocast, GradScaler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed, sample_min_number_of_indices
from dataloader import GraphDataLoader
from model import DICE


def contrastive_learning_loss(gf, gf_labels, tau_p, tau, tau_n):
    # -----------------------------------
    # Preprocess
    # -----------------------------------
    # negative label -i just indicate that they are negative augmentation pairs of graphs with label i

    graph_indices = sample_min_number_of_indices(gf_labels)
    graph_labels, graph_features = gf_labels[graph_indices], gf[graph_indices]

                    # (same label & both pos label)
    positive_mask =   (graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))\
                    & ((graph_labels > 0).unsqueeze(1) & (graph_labels > 0).unsqueeze(0))

                    # (same label & negative label) |
                    # (diff label & both pos label) | (diff label & one neg or both neg label)
                    #                                   |-|-|-|-|-|-|-|-|-|-------> [This includes negative data augmentation relation]
    nonequal_mask =  ((graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))\
                        & ~((graph_labels > 0).unsqueeze(1) & (graph_labels > 0).unsqueeze(0)))\
                    | ~(graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))

                    # negative data augementation relation
    negative_mask = (graph_labels.unsqueeze(1) == -graph_labels.unsqueeze(0))

    # -----------------------------------
    # Graph-level Contrastive Loss
    # -----------------------------------
    gf_n = F.normalize(graph_features, dim=-1)
    gf_cosine_similarity = torch.mm(gf_n, gf_n.t())

    gs_exp = torch.exp(gf_cosine_similarity)

    gf_log_value_matrix = \
        gf_cosine_similarity/tau - torch.log(
            (gs_exp**(1/(tau_p))).masked_fill(~(positive_mask), 0).sum(dim=1, keepdim=True)  # sum pos pairs
            + (gs_exp**(1/tau_n)).masked_fill(~(nonequal_mask & ~negative_mask), 0).sum(dim=1, keepdim=True)  # sum neg pairs
        )

    gf_loss = - (gf_log_value_matrix * positive_mask.float()).sum()
    gf_loss /= positive_mask.float().sum()

    return gf_loss


def train_model(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    ### Dataset
    train_dataset = torch.load(f'./pretrain/dataset/{args.dataset_name}_train_pda.pt')
    val_dataset = torch.load(f'./pretrain/dataset/{args.dataset_name}_val_pda.pt')
    print()
    print("Dataset Loaded")
    print()

    with open('./params.json', 'r') as f:
        params = json.load(f)
    tau_p = f"{params['pretraining']['train']['tau_p']}".replace(".", "")
    tau = f"{params['pretraining']['train']['tau']}".replace(".", "")
    tau_n = f"{params['pretraining']['train']['tau_n']}".replace(".", "")

    ########################
    model = DICE(params['model']['encoder']['dice'], gnn_depth=args.gnn_depth)
    model.apply(init_weights)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=params['pretraining']['train']['lr'])
    model = model.to(params['pretraining']['train']["device"])
    ########################
    print()
    print("Model Initialized")
    print()

    ### Wandb Initialization
    if args.wandb_log:
        config = {}
        for key, value in params['pretraining']['train'].items():
            config[key] = value

        wandb.init(
            project=params['project_name'],
            name=(
                f"dice_pretraining_{params['model']['encoder']['dice']['gnn_type']}"
                f"_depth{args.gnn_depth}_taup{tau_p}tau{tau}taun{tau_n}_pda_seed{args.seed}"
            ),
            config=config
        )

    ### Dataloader
    # training dataloader
    train_data = []
    for circuit_name, pos_neg_train_data in train_dataset.items():
        pos_train_data = pos_neg_train_data['pos']
        train_data += pos_train_data
    random.shuffle(train_data)
    train_dataloader = GraphDataLoader(
        train_data,
        batch_size=params['pretraining']['train']['batch_size'],
        shuffle=True
    )

    # validation dataloader
    val_data = []
    for circuit_name, pos_neg_val_data in val_dataset.items():
        pos_val_data = pos_neg_val_data['pos']
        val_data += pos_val_data
    random.shuffle(val_data)
    val_dataloader = GraphDataLoader(
        val_data,
        batch_size=params['pretraining']['train']['batch_size'],
        shuffle=False
    )

    print()
    print("Dataloader Initialized")
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    print()


    ### Training
    scaler = GradScaler()
    for epoch in range(int(params['pretraining']['train']["epochs"])):

        # train
        model.train()
        train_losses = []
        for train_batch in train_dataloader:
            train_batch = send_to_device(train_batch, params['pretraining']['train']["device"])
            ############################
            model.optimizer.zero_grad()
            with autocast(device_type=params['pretraining']['train']["device"]):
                _, _, gf = model(train_batch)
                train_loss = contrastive_learning_loss(
                    gf,
                    train_batch['circuit'],
                    params['pretraining']['train']['tau_p'],
                    params['pretraining']['train']['tau'],
                    params['pretraining']['train']['tau_n']
                )
            scaler.scale(train_loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
            ############################
            train_losses.append(train_loss.item())

        # validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = send_to_device(val_batch, params['pretraining']['train']["device"])
                ############################
                with autocast(device_type=params['pretraining']['train']["device"]):
                    _, _, gf = model(val_batch)
                    val_loss = contrastive_learning_loss(
                        gf,
                        val_batch['circuit'],
                        params['pretraining']['train']['tau_p'],
                        params['pretraining']['train']['tau'],
                        params['pretraining']['train']['tau_n']
                    )
                ############################
                val_losses.append(val_loss.item())

        info = {}
        if args.wandb_log:
            info['dice_pretraining_train_loss'] = sum(train_losses) / len(train_losses)
            info['dice_pretraining_val_loss'] = sum(val_losses) / len(val_losses)
            wandb.log(info)

        print()
        print(
            f"Epoch: {epoch}/{params['pretraining']['train']['epochs']}\n"
            f"Train Loss: {sum(train_losses)/len(train_losses)}\n"
            f"Validation Loss: {sum(val_losses)/len(val_losses)}"
        )

        model.save(
            f"./pretrain/encoder/saved_models"
            f"/{params['project_name']}_pretrained_model"
            f"_{params['model']['encoder']['dice']['gnn_type']}"
            f"_depth{args.gnn_depth}_taup{tau_p}tau{tau}taun{tau_n}_pda_epoch{epoch}.pt"
        )

    if args.wandb_log:
        wandb.finish()

    model.save(
        f"./pretrain/encoder/{params['project_name']}_pretrained_model"
        f"_{params['model']['encoder']['dice']['gnn_type']}"
        f"_depth{args.gnn_depth}_taup{tau_p}tau{tau}taun{tau_n}_pda.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICE Pre-Training.")
    parser.add_argument(
        "--dataset_name",
        default="pretraining_dataset_wo_device_params",
        type=str,
        help="Name of the dataset directory"
    )
    parser.add_argument("--gnn_depth", default=2, type=int, help="Depth of GNN")
    parser.add_argument("--wandb_log", default=0, type=int, help="Log to wandb")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    train_model(args)
