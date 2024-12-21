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



def test(args):

    dataset = torch.load(f'./downstream_tasks/{args.task_name}/{args.subtask_name}/{args.subtask_name}_dataset.pt')

    with open(args.params, 'r') as f:
        params = json.load(f)

    ########################
    # model type 0 : GE_MLP + DICE
    # model type 1 : GE_MLP
    # model type 2 : GE_MLP + GE_GNN_wo_dp
    # model type 3 : GE_MLP + GE_GNN_w_dp
    model = DownstreamModel(
        args.subtask_name, 
        args.model_type, 
        params['model']
    )
    model.load(f"./downstream_tasks/{args.task_name}/{args.subtask_name}"\
               f"/saved_models/{args.subtask_name}_model_{args.model_type}.pt")
    ########################

    # Dataset
    test_data = dataset['test_data']
    print("test_data size : ", len(test_data))

    test_dataloader = GraphDataLoader(test_data, batch_size=params['downstream_tasks']['test']['batch_size'], shuffle=True)

    # Model
    model = model.to(params['downstream_tasks']['test']['device'])
    model = model.eval()

    # Test
    for iteration in range(int(params['downstream_tasks']['test']['num_iterations'])):

        test_losses = []
        for test_batch in test_dataloader:
            # Sending to Device
            test_batch = send_to_device(test_batch, params['downstream_tasks']['test']["device"])
            ############################
            out = model(test_batch)
            test_loss = calculate_downstream_loss(out, test_batch, args.subtask_name)
            test_loss.backward()
            ############################
            test_losses.append(test_loss.item())

        print()
        print(f"iter: {iteration}/{params['downstream_tasks']['test']['num_iterations']}\n"\
              f"Test Loss: {sum(test_losses)/len(test_losses)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="digital")
    parser.add_argument("--subtask_name", type=str, default="delay_prediction")
    parser.add_argument("--params", default="./params.json", type=str, help="Path to the params file")
    parser.add_argument("--model_type", default=1, type=int, help="Model Type")
    args = parser.parse_args()
    test(args)