from math import e
import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/data/torch_datasets'))
sys.path.append(parent_dir)

from dataloader import HeteroGraphDataLoader
from model import *



def train_model(model, train_params, dataset, wandb_log=False):

    # Wandb Initialization
    if wandb_log:
        config = {}
        for key, value in train_params.items():
            config[key] = value
        config["dataset"] = dataset['name']

        wandb.init(
            project="DC Voltage Prediction",
            config=config
        )

    # Dataset
    train_data = dataset['train_data']
    print("train_data size : ", len(train_data))
    val_data = dataset['val_data']
    print("val_data size : ", len(val_data))
    train_dataloader = HeteroGraphDataLoader(train_data, batch_size=train_params["batch_size"], shuffle=True)
    val_dataloader = HeteroGraphDataLoader(val_data, batch_size=train_params["batch_size"], shuffle=False)

    # Model
    model = model.to(train_params["device"])
    model = model.train()

    # Criterion
    if train_params["loss"] == "mse":
        criterion = nn.MSELoss()
    

    # Training
    for epoch in range(int(train_params["epochs"])):

        train_losses = []        
        for train_batch in train_dataloader:
            # Sending to Device
            train_batch = send_to_device(train_batch, train_params["device"])
            ############################
            model.optimizer.zero_grad()

            output, e_info, d_info = model(train_batch)

            train_loss = criterion(output, train_batch['output']['dc_voltages'])
            train_loss.backward()

            model.optimizer.step()
            ############################
            train_losses.append(train_loss.item())

        val_losses = []
        for val_batch in val_dataloader:
            # Sending to Device
            val_batch = send_to_device(val_batch, train_params["device"])
            ############################
            with torch.no_grad():
                output, e_info, d_info = model(val_batch)
                val_loss = criterion(output, val_batch['output']['dc_voltages'])
            ############################
            val_losses.append(val_loss.item())

        info = {}
        if wandb_log:
            info['train_loss'] = sum(train_losses)/len(train_losses)
            info['val_loss'] = sum(val_losses)/len(val_losses)
            wandb.log(info)


        print()
        print(f"Epoch: {epoch}/{train_params['epochs']}\n\
                Train Loss: {sum(train_losses)/len(train_losses)}\n\
                Validation Loss: {sum(val_losses)/len(val_losses)}")


    if wandb_log: wandb.finish()
    
    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Voltage Prediction Training.")
    parser.add_argument("--dataset_dir", default="./dataset/data/torch_datasets",
                        type=str, help="Name of the netlist template directory")
    args = parser.parse_args()

    datasets = os.listdir(args.dataset_dir)
    print(datasets)

    for dataset in datasets:
        if dataset == "dataloader.py" or dataset == "__pycache__":
            continue
        
        # Load Dataset
        data_set = torch.load(f'{args.dataset_dir}/{dataset}')

        # Load Parameters
        with open(f"./model_params.json", 'r') as f:
            model_params = json.load(f)
        with open(f"./train_params.json", 'r') as f:
            train_params = json.load(f)

        ########################
        model = PretrainModel(model_params)
        model = train_model(model, train_params, data_set, wandb_log=True)
        model.save(f"./saved_models")
        ########################