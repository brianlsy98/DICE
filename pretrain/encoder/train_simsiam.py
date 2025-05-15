import os
import sys
import json
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.cuda.amp import autocast, GradScaler


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed, sample_min_number_of_indices
from dataloader import GraphDataLoader
from model import DICE


def simsiam_loss(gf, gf_labels, predictor):
    # -----------------------------------
    # Preprocess
    # -----------------------------------
    graph_indices = sample_min_number_of_indices(gf_labels)
    graph_labels, graph_features = gf_labels[graph_indices], gf[graph_indices]

    # -----------------------------------
    # SimSiam Loss
    # -----------------------------------    
    # projection, prediction
    z, p = graph_features, predictor(graph_features)            # shape: [N, D]
    # normalized z*p matrix
    zp_normalized_matrix = torch.mm(F.normalize(z.detach(), dim=-1),
                                    F.normalize(p, dim=-1).t()) # shape: [N, N]
    # positive mask
    positive_mask = (graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))
                                                                # shape: [N, N]
    # loss
    loss = -zp_normalized_matrix[positive_mask].mean()
    
    return loss



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

    
    ########################
    # simsiam predictor
    simsiam_predictor = nn.Sequential(
        nn.Linear(params['model']['encoder']['dice']['gf_out_dim'], params['model']['encoder']['dice']['hidden_dim']),
        nn.BatchNorm1d(params['model']['encoder']['dice']['hidden_dim']),
        nn.ReLU(),
        nn.Linear(params['model']['encoder']['dice']['hidden_dim'], params['model']['encoder']['dice']['hidden_dim']),
        nn.BatchNorm1d(params['model']['encoder']['dice']['hidden_dim']),
        nn.ReLU(),
        nn.Linear(params['model']['encoder']['dice']['hidden_dim'], params['model']['encoder']['dice']['gf_out_dim'])
    ).to(params['pretraining']['train']["device"])
    simsiam_predictor.apply(init_weights)
    # DICE
    model = DICE(params['model']['encoder']['dice'], gnn_depth=args.gnn_depth)
    model.apply(init_weights)
    update_params = list(model.parameters()) + list(simsiam_predictor.parameters())
    model.optimizer = torch.optim.Adam(update_params, lr=0.01*params['pretraining']['train']['lr'])
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
                f"_depth{args.gnn_depth}_simsiam_seed{args.seed}"
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
            with autocast():
                _, _, gf = model(train_batch)
                train_loss = simsiam_loss(
                    gf,
                    train_batch['circuit'],
                    simsiam_predictor
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
                with autocast():
                    _, _, gf = model(val_batch)
                    val_loss = simsiam_loss(
                        gf,
                        val_batch['circuit'],
                        simsiam_predictor
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

        if epoch % 40 == 0:
            model.save(
                f"./pretrain/encoder/saved_models"
                f"/{params['project_name']}_pretrained_model"
                f"_{params['model']['encoder']['dice']['gnn_type']}"
                f"_depth{args.gnn_depth}_simsiam_epoch{epoch}.pt"
            )

    if args.wandb_log:
        wandb.finish()

    model.save(
        f"./pretrain/encoder/{params['project_name']}_pretrained_model"
        f"_{params['model']['encoder']['dice']['gnn_type']}"
        f"_depth{args.gnn_depth}_simsiam.pt"
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
