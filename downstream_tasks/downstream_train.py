import os
import sys
import json
from math import isnan, isinf
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from torch.amp import autocast, GradScaler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed
from dataloader import GraphDataLoader

from downstream_model import DownstreamModel
from downstream_test import get_test_info


import itertools

def calculate_downstream_loss(out, batch, task_name):

    if task_name == "circuit_similarity_prediction":
        permutations = list(itertools.permutations(range(batch['batch'].max().item() + 1), 2))  # (N, 2)
        N = len(permutations)
        indices = torch.tensor(permutations).to(out.device)

        target_label = batch['labels'][0].view(1, 1, -1)  # (1, 1, label_num)
        labels = batch['labels'][indices,:].view(N, 2, -1)  # (N, 2, label_num)

        comparison = torch.sum(target_label * labels, dim=-1)  # (N, 2)

        left_is_less = (comparison[:,0] < comparison[:,1]).long()
        equal = (comparison[:,0] == comparison[:,1]).long()
        left_is_greater = (comparison[:,0] > comparison[:,1]).long()

        target = equal + 2 * left_is_greater    # (N, )

        # out: (N, 3)
        loss = F.cross_entropy(out, target)

        return loss


    elif task_name == "delay_prediction":
        rise_delay = batch['minus_log_rise_delay']
        fall_delay = batch['minus_log_fall_delay']
        delays = torch.stack([rise_delay, fall_delay], dim=1)
        # print(torch.cat([torch.exp(-out), torch.exp(-delays)], dim=1))
        ##### remind that the output is minus log value of the delays
        if torch.isnan(delays).any():
            raise ValueError("NaN in delays")
        elif torch.isinf(delays).any():
            raise ValueError("Inf in delays")

        return F.mse_loss(out, delays)

    elif task_name == "opamp_metric_prediction":
        power = batch['power']
        voutdc_minus_vindc = batch['voutdc_minus_vindc']
        cmrr_dc = batch['cmrr_dc']
        gain_dc = batch['gain_dc']
        vddpsrr_dc = batch['vddpsrr_dc']
        metrics = torch.stack([power, voutdc_minus_vindc, cmrr_dc, gain_dc, vddpsrr_dc], dim=1)
        # power: *1e3, voutdc_minus_vindc: *10, cmrr_dc: /10, gain_dc: /10, vddpsrr_dc: /10
        return F.mse_loss(out, metrics)

    else: raise ValueError("Invalid Subtask Name")



def train(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    ### Dataset
    train_data = torch.load(f'./downstream_tasks/{args.task_name}/{args.task_name}_train.pt')
    val_data = torch.load(f'./downstream_tasks/{args.task_name}/{args.task_name}_val.pt')
    print()
    print("Dataset Loaded")
    print("train_data:", len(train_data))
    print()
    
    with open("./params.json", 'r') as f:
        params = json.load(f)
    taup = args.taup.replace(".", ""); tau = args.tau.replace(".",""); tau_n = args.taun.replace(".", "")

    ########################
    model = DownstreamModel(args, params['model'])
    model.apply(init_weights)
    if args.dice_depth > 0:
        if not args.pda:
            model.load_dice(f"./pretrain/encoder/DICE_pretrained_model"
                            f"_{params['model']['encoder']['dice']['gnn_type']}"
                            f"_depth{args.dice_depth}_taup{taup}tau{tau}taun{tau_n}.pt")
        elif args.pda:
            model.load_dice(f"./pretrain/encoder/DICE_pretrained_model"
                            f"_{params['model']['encoder']['dice']['gnn_type']}"
                            f"_depth{args.dice_depth}_taup{taup}tau{tau}taun{tau_n}_pda.pt")
    model.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # only update trainable params
        lr=params['downstream_tasks'][f'{args.task_name}']['train']['lr']
    )
    model = model.to(params['downstream_tasks'][f'{args.task_name}']['train']['device'])
    ########################
    print()
    print("Model Initialized")
    print("training parameter num:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print()
    if args.dice_depth == 0:
        model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                    f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                    f"_sGNN{args.s_gnn_depth}_seed{args.seed}"
    else:
        if not args.pda:
            model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                        f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                        f"_sGNN{args.s_gnn_depth}_taup{taup}tau{tau}taun{tau_n}_seed{args.seed}"
        elif args.pda:
            model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                        f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                        f"_sGNN{args.s_gnn_depth}_taup{taup}tau{tau}taun{tau_n}_pda_seed{args.seed}"


    ### Wandb Initialization
    if args.wandb_log:
        config = {}
        for key, value in params['downstream_tasks'][f'{args.task_name}']['train'].items():
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
        batch_size=params['downstream_tasks'][f'{args.task_name}']['train']['batch_size'],
        shuffle=True
    )
    # validation dataloader
    val_dataloader = GraphDataLoader(
        val_data,
        batch_size=params['downstream_tasks'][f'{args.task_name}']['train']['batch_size'],
        shuffle=False
    )
    if args.task_name == "circuit_similarity_prediction":
        # acc dataloader
        acc_dataloader = GraphDataLoader(
            val_data[:54],
            batch_size=3,
            shuffle=True
        )

    print()
    print("Dataloader Initialized")
    print("train_data:", len(train_data))
    print("val_data:", len(val_data))
    print()
    min_val_loss = float('inf')

    # Training
    scaler = GradScaler()
    for epoch in range(int(params['downstream_tasks'][f'{args.task_name}']['train']['epochs'])):

        model.train(); model.encoder.dice.eval()
        train_losses = []
        for train_batch in train_dataloader:
            train_batch = send_to_device(train_batch, params['downstream_tasks'][f'{args.task_name}']['train']["device"])
            ############################
            model.optimizer.zero_grad()
            with autocast(device_type=params['downstream_tasks'][f'{args.task_name}']['train']['device']):
                out = model(train_batch)
                train_loss = calculate_downstream_loss(out, train_batch, args.task_name)
            scaler.scale(train_loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
            ############################
            train_losses.append(train_loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_batch = send_to_device(val_batch, params['downstream_tasks'][f'{args.task_name}']['train']['device'])
                ############################
                with autocast(device_type=params['downstream_tasks'][f'{args.task_name}']['train']["device"]):
                    out = model(val_batch)
                    val_loss = calculate_downstream_loss(out, val_batch, args.task_name)
                ############################
                val_losses.append(val_loss.item())

        if args.task_name == "circuit_similarity_prediction":
            accuracies = []
            with torch.no_grad():
                for acc_batch in acc_dataloader:
                    acc_batch = send_to_device(acc_batch, params['downstream_tasks'][f'{args.task_name}']['train']['device'])
                    ############################
                    with autocast(device_type=params['downstream_tasks'][f'{args.task_name}']['train']["device"]):
                        out = model(acc_batch)
                        accuracies.append(get_test_info(out, acc_batch, args.task_name))
                    ############################


        info = {}
        if args.wandb_log:
            info[f'{args.task_name}_train_loss'] = sum(train_losses) / len(train_losses)
            info[f'{args.task_name}_val_loss'] = sum(val_losses) / len(val_losses)
            if args.task_name == "circuit_similarity_prediction":
                info[f'{args.task_name}_accuracy'] = torch.cat(accuracies, dim=0).mean().item()
            
            wandb.log(info)

        print()
        print(
            f"Epoch: {epoch}/{params['downstream_tasks'][f'{args.task_name}']['train']['epochs']}\n"
            f"    Train Loss: {sum(train_losses)/len(train_losses)}\n"
            f"    Validation Loss: {sum(val_losses)/len(val_losses)}"
        )
        if args.task_name == "circuit_similarity_prediction":
            print(f"    Validation Accuracy: {torch.cat(accuracies, dim=0).mean().item()}")


        if args.task_name == "delay_prediction" or args.task_name == "opamp_metric_prediction":
            if not isnan(sum(val_losses)/len(val_losses)) and not isinf(sum(val_losses)/len(val_losses))\
                and not isnan(sum(train_losses)/len(train_losses)) and not isinf(sum(train_losses)/len(train_losses)):
                if sum(val_losses)/len(val_losses) < min_val_loss:
                    min_val_loss = sum(val_losses)/len(val_losses)
                    model.save(f"./downstream_tasks/{args.task_name}/saved_models/{model_name}.pt")

        if epoch % (int(params['downstream_tasks'][f'{args.task_name}']['train']['epochs']) // 5) == 0:
            model.save(f"./downstream_tasks/{args.task_name}/saved_models/{model_name}_epoch{epoch}.pt")

    if args.wandb_log: wandb.finish()
    if args.task_name == "circuit_similarity_prediction":
        model.save(f"./downstream_tasks/{args.task_name}/saved_models/{model_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="circuit_similarity_prediction")
    parser.add_argument("--dice_depth", type=int, default=0, help="depth for DICE")
    parser.add_argument("--p_gnn_depth", type=int, default=0, help="depth for Newly training parallel GNN")
    parser.add_argument("--s_gnn_depth", type=int, default=0, help="depth for Newly training series GNN")
    parser.add_argument("--taup", type=str, default="0.2", help="tau_p value for DICE")
    parser.add_argument("--tau", type=str, default="0.05", help="tau value for DICE")
    parser.add_argument("--taun", type=str, default="0.05", help="tau_n value for DICE")
    parser.add_argument("--pda", type=int, default=0, help="Use only Positive Data Augmentation")
    parser.add_argument("--wandb_log", default=0, type=int, help="Log to wandb")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    train(args)