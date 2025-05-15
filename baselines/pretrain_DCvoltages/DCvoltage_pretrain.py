import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from torch.cuda.amp import autocast, GradScaler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed
from dataloader import GraphDataLoader

from baseline_models import DCvoltagePrediction


def dcvoltage_mse_loss(out, batch):
    voltage_node_indices = torch.where(batch['node_y'] <= 2)[0]

    return F.mse_loss(out[voltage_node_indices],
                      batch['device_params'][voltage_node_indices])



def train(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    ### Dataset
    train_data = torch.load(f'./baselines/pretrain_DCvoltages/dcvoltage_prediction_train.pt')
    val_data = torch.load(f'./baselines/pretrain_DCvoltages/dcvoltage_prediction_val.pt')
    print()
    print("Dataset Loaded")
    print("train_data:", len(train_data))
    print()
    
    with open("./params.json", 'r') as f:
        params = json.load(f)

    ########################
    lr = 5e-6
    batch_size = 1024
    epochs = 200
    model = DCvoltagePrediction(args, params['baseline_model']['DCvoltage_pretrain_model'])
    model.apply(init_weights)
    model.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )
    model = model.to('cuda')
    ########################
    print()
    print("Model Initialized")
    print("training parameter num:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print()
    model_name = f"DCvoltagePredictPretrain_seed{args.seed}"


    ### Wandb Initialization
    if args.wandb_log:
        config = {"lr": lr}
        for key, value in params['baseline_model']['DCvoltage_pretrain_model'].items():
            config[key] = value

        wandb.init(
            project=params['project_name'],
            name=model_name,
            config=config
        )


    ### Dataloader
    # training dataloader
    train_dataloader = GraphDataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    # validation dataloader
    val_dataloader = GraphDataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )

    print()
    print("Dataloader Initialized")
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    print()

    # Training
    scaler = GradScaler()
    for epoch in range(epochs):

        model.train()
        train_losses = []
        for train_batch in train_dataloader:
            train_batch = send_to_device(train_batch, 'cuda')
            ############################
            model.optimizer.zero_grad()
            with autocast():
                out = model(train_batch)
                train_loss = dcvoltage_mse_loss(out.squeeze(), train_batch)
            scaler.scale(train_loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
            ############################
            train_losses.append(train_loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = send_to_device(val_batch, 'cuda')
                ############################
                with autocast():
                    out = model(val_batch)
                    val_loss = dcvoltage_mse_loss(out.squeeze(), val_batch)
                ############################
                val_losses.append(val_loss.item())
        
        info = {}
        if args.wandb_log:
            info[f'dcvoltage_train_loss'] = sum(train_losses) / len(train_losses)
            info[f'dcvoltage_val_loss'] = sum(val_losses) / len(val_losses)
            wandb.log(info)

        print()
        print(
            f"Epoch: {epoch}/{epochs}\n"
            f"    Train Loss: {sum(train_losses)/len(train_losses)}\n"
            f"    Validation Loss: {sum(val_losses)/len(val_losses)}"
        )

        if epoch % (epochs // 5) == 0:
            model.save(f"./baselines/pretrain_DCvoltages/saved_models/{model_name}_epoch{epoch}.pt")

    if args.wandb_log: wandb.finish()

    model.save(f"./baselines/pretrain_DCvoltages/{model_name}.pt")
    model.save_gnn_backbone(f"./baselines/pretrain_DCvoltages/{model_name}_gnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_log", default=0, type=int, help="Log to wandb")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    train(args)