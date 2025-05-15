import os
import sys
import json
import argparse

from torcheval.metrics.functional import r2_score

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

def dcvoltage_r2_score(out, batch):
    voltage_node_indices = torch.where(batch['node_y'] <= 2)[0]

    return r2_score(out[voltage_node_indices],
                    batch['device_params'][voltage_node_indices])


def test(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ### Dataset
    test_data = torch.load(f'./baselines/pretrain_DCvoltages/dcvoltage_prediction_test.pt')
    print()
    print("Dataset Loaded")
    print("test_data:", len(test_data))
    print()
    
    with open("./params.json", 'r') as f:
        params = json.load(f)

    ########################
    seed_num = 10
    batch_size = 1024
    model = DCvoltagePrediction(args, params['baseline_model']['DCvoltage_pretrain_model'])
    model.load(f"./baselines/pretrain_DCvoltages/DCvoltagePredictPretrain_seed{args.seed}.pt")
    model = model.to('cuda')
    model.eval()
    ########################
    print()
    print("Model Initialized")
    print("model parameter num:", sum(p.numel() for p in model.parameters()))
    print()
    model_name = f"DCvoltagePredictPretrain_seed{args.seed}"


    ### Dataloader
    # test dataloader
    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True
    )

    print()
    print("Dataloader Initialized")
    print("test_data:", len(test_data))
    print()

    # Training
    scaler = GradScaler()
    for seed in range(seed_num):
        set_seed(seed)

        test_losses = []
        with torch.no_grad():
            for test_batch in test_dataloader:
                test_batch = send_to_device(test_batch, 'cuda')
                ############################
                with autocast():
                    out = model(test_batch)
                    test_loss = dcvoltage_mse_loss(out.squeeze(), test_batch)
                    r2 = dcvoltage_r2_score(out.squeeze(), test_batch)
                ############################
                test_losses.append(test_loss.item())
        print()
        print(
            f"seed: {seed}/{seed_num}\n"
            f"    Test Loss: {sum(test_losses)/len(test_losses)}\n"
            f"    Test R2: {r2}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")
    args = parser.parse_args()
    test(args)