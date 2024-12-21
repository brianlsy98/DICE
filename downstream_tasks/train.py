import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import send_to_device, calculate_downstream_loss
from dataloader import GraphDataLoader

from downstream_model import DownstreamModel


def train(args):
    dataset = torch.load(
        f'./downstream_tasks/{args.task_name}/{args.subtask_name}/{args.subtask_name}_dataset.pt'
    )

    with open(args.params, 'r') as f:
        params = json.load(f)

    ########################
    # model type 0 : GE_MLP + DICE  (DICE is frozen)
    # model type 1 : GE_MLP
    # model type 2 : GE_MLP + GE_GNN_wo_dp
    # model type 3 : GE_MLP + GE_GNN_w_dp
    model = DownstreamModel(
        args.subtask_name, 
        args.model_type, 
        params['model']
    )
    model.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # only update trainable params
        lr=params['downstream_tasks']['train']['lr']
    )
    ########################

    # Wandb Initialization
    if args.wandb_log:
        config = {}
        for key, value in params['downstream_tasks']['train'].items():
            config[key] = value
        config["dataset"] = dataset['name']

        wandb.init(
            project=params['project_name'],
            name=f"model_{args.model_type}",
            config=config
        )

    # Dataset
    train_data = dataset['train_data']
    val_data = dataset['val_data']
    print("train_data size : ", len(train_data))
    print("val_data size : ", len(val_data))

    train_dataloader = GraphDataLoader(
        train_data,
        batch_size=params['downstream_tasks']['train']['batch_size'],
        shuffle=True
    )
    val_dataloader = GraphDataLoader(
        val_data,
        batch_size=params['downstream_tasks']['train']['batch_size'],
        shuffle=False
    )

    # Model
    device = params['downstream_tasks']['train']['device']
    model = model.to(device)
    model = model.train()

    # Training
    for epoch in range(int(params['downstream_tasks']['train']['epochs'])):

        train_losses = []
        for train_batch in train_dataloader:
            # Sending to Device
            train_batch = send_to_device(train_batch, params['downstream_tasks']['train']["device"])
            model.optimizer.zero_grad()
            out = model(train_batch)
            train_loss = calculate_downstream_loss(out, train_batch, args.subtask_name)
            train_loss.backward()
            model.optimizer.step()
            train_losses.append(train_loss.item())

        val_losses = []
        for val_batch in val_dataloader:
            # Sending to Device
            val_batch = send_to_device(val_batch, params['downstream_tasks']['train']["device"])
            with torch.no_grad():
                out = model(val_batch)
                val_loss = calculate_downstream_loss(out, val_batch, args.subtask_name)
            val_losses.append(val_loss.item())

        info = {}
        if args.wandb_log:
            info[f'{args.subtask_name}_train_loss'] =\
                sum(train_losses) / len(train_losses)
            info[f'{args.subtask_name}_val_loss'] =\
                sum(val_losses) / len(val_losses)
            wandb.log(info)

        print()
        print(
            f"Epoch: {epoch}/{params['downstream_tasks']['train']['epochs']}\n"
            f"    Train Loss: {sum(train_losses)/len(train_losses)}\n"
            f"    Validation Loss: {sum(val_losses)/len(val_losses)}"
        )

        if epoch % (int(params['downstream_tasks']['train']['epochs']) // 5) == 0:
            model.save(
                f"./downstream_tasks/{args.task_name}/{args.subtask_name}/saved_models/"
                f"{args.subtask_name}_model_{args.model_type}_epoch_{epoch}.pt"
            )

    if args.wandb_log:
        wandb.finish()

    model.save(
        f"./downstream_tasks/{args.task_name}/{args.subtask_name}/saved_models/"
        f"{args.subtask_name}_model_{args.model_type}.pt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="digital")
    parser.add_argument("--subtask_name", type=str, default="delay_prediction")
    parser.add_argument("--params", default="./params.json", type=str, help="Path to the params file")
    parser.add_argument("--model_type", default=2, type=int, help="Model Type")
    parser.add_argument("--wandb_log", default=False, type=bool, help="Log to wandb")
    args = parser.parse_args()
    train(args)