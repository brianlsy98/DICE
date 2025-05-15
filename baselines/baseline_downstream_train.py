import os
import sys
import json
import argparse
from math import isnan, isinf

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from torch.cuda.amp import autocast, GradScaler

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream_tasks'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed
from dataloader import GraphDataLoader

from downstream_train import calculate_downstream_loss
from downstream_test import get_test_info, process_info_seed

from baseline_models import BaselineModel



def train(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)

    with open("./params.json", 'r') as f:
        params = json.load(f)


    ### Dataset
    if args.baseline_name == "DeepGen_u"\
    or args.baseline_name == "DeepGen_p":
        train_data = torch.load(f'./baselines/{args.task_name}/{args.task_name}_DeepGen_train.pt')
        val_data = torch.load(f'./baselines/{args.task_name}/{args.task_name}_DeepGen_val.pt')
    elif args.baseline_name == "DICE":
        train_data = torch.load(f'./downstream_tasks/{args.task_name}/{args.task_name}_train.pt')
        val_data = torch.load(f'./downstream_tasks/{args.task_name}/{args.task_name}_val.pt')
    elif args.baseline_name == "ParaGraph":
        train_data = torch.load(f'./baselines/{args.task_name}/{args.task_name}_ParaGraph_train.pt')
        val_data = torch.load(f'./baselines/{args.task_name}/{args.task_name}_ParaGraph_val.pt')
    print()
    print("Dataset Loaded")
    print("train_data:", len(train_data))
    print()
    

    ########################
    model = BaselineModel(args, params['baseline_model'])
    model.apply(init_weights)
    if args.baseline_name == "DICE":
        model.encoder.encoder.dice.load_state_dict(torch.load("./pretrain/encoder/DICE_pretrained_model_GIN_depth2_taup02tau005taun005.pt"))
        for param in model.encoder.encoder.dice.parameters(): param.requires_grad = False    # freeze DICE
    elif args.baseline_name == "DeepGen_p":
        model.encoder.encoder.gnn_backbone.load_state_dict(torch.load("./baselines/pretrain_DCvoltages/DCvoltagePredictPretrain_seed0_gnn.pt"))
        for param in model.encoder.encoder.gnn_backbone.parameters(): param.requires_grad = False    # freeze
    
    lr = params['downstream_tasks'][f'{args.task_name}']['train']['lr']

    model.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),  # only update trainable params
        lr=lr
    )
    model = model.to(params['downstream_tasks'][f'{args.task_name}']['train']['device'])
    ########################
    print()
    print("Model Initialized")
    print("training parameter num:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print()
    if args.baseline_name == "DICE":
        model_name = f"{args.task_name}_{args.baseline_name}_taup02tau005taun005_seed{args.seed}"
    else:
        model_name = f"{args.task_name}_{args.baseline_name}_seed{args.seed}"


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

    if args.task_name != "circuit_similarity_prediction":
        batch_size = params['downstream_tasks'][f'{args.task_name}']['train']['batch_size']//2
        epochs = int(params['downstream_tasks'][f'{args.task_name}']['train']['epochs']//3)
    else:
        batch_size = params['downstream_tasks'][f'{args.task_name}']['train']['batch_size']
        epochs = int(params['downstream_tasks'][f'{args.task_name}']['train']['epochs'])

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
    for epoch in range(epochs):

        model.train()
        if args.baseline_name == "DICE": model.encoder.encoder.dice.eval()
        elif args.baseline_name == "DeepGen_p": model.encoder.encoder.gnn_backbone.eval()
        train_losses = []
        for train_batch in train_dataloader:
            train_batch = send_to_device(train_batch, params['downstream_tasks'][f'{args.task_name}']['train']["device"])
            ############################
            model.optimizer.zero_grad()
            with autocast():
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
                with autocast():
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
                    with autocast():
                        out = model(acc_batch)
                        accuracies.append(get_test_info(out, acc_batch, args.task_name))
                    ############################
        elif args.task_name == "delay_prediction" or args.task_name == "opamp_metric_prediction":
            test_infos = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch = send_to_device(val_batch, params['downstream_tasks'][f'{args.task_name}']['train']['device'])
                    ############################
                    with autocast():
                        out = model(val_batch)
                        test_infos.append(get_test_info(out, val_batch, args.task_name))
                    ############################
            test_infos = torch.cat(test_infos, dim=0)
            r2_score = process_info_seed(args, test_infos, 0, params)



        info = {}
        if args.wandb_log:
            info[f'{args.task_name}_train_loss'] = sum(train_losses) / len(train_losses)
            info[f'{args.task_name}_val_loss'] = sum(val_losses) / len(val_losses)
            if args.task_name == "circuit_similarity_prediction":
                info[f'{args.task_name}_accuracy'] = torch.cat(accuracies, dim=0).mean().item()
            elif args.task_name == "delay_prediction":
                info[f'{args.task_name}_rise_d_r2'] = r2_score[0].item()
                info[f'{args.task_name}_fall_d_r2'] = r2_score[1].item()
            elif args.task_name == "opamp_metric_prediction":
                info[f'{args.task_name}_power_r2'] = r2_score[0].item()
                info[f'{args.task_name}_voffset_r2'] = r2_score[1].item()
                info[f'{args.task_name}_cmrr_r2'] = r2_score[2].item()
                info[f'{args.task_name}_gain_r2'] = r2_score[3].item()
                info[f'{args.task_name}_psrr_r2'] = r2_score[4].item()

            wandb.log(info)

        print()
        print(
            f"Epoch: {epoch}/{epochs}\n"
            f"    Train Loss: {sum(train_losses)/len(train_losses)}\n"
            f"    Validation Loss: {sum(val_losses)/len(val_losses)}"
        )
        if args.task_name == "circuit_similarity_prediction":
            print(f"    Validation Accuracy: {torch.cat(accuracies, dim=0).mean().item()}")
        elif args.task_name == "delay_prediction":
            print(f"    Rise Delay R2 Score: {r2_score[0].item()}")
            print(f"    Fall Delay R2 Score: {r2_score[1].item()}")
        elif args.task_name == "opamp_metric_prediction":
            print(f"    Power R2 Score: {r2_score[0].item()}")
            print(f"    Voffset R2 Score: {r2_score[1].item()}")
            print(f"    CMRR R2 Score: {r2_score[2].item()}")
            print(f"    Gain R2 Score: {r2_score[3].item()}")
            print(f"    PSRR R2 Score: {r2_score[4].item()}")


        if args.task_name == "delay_prediction" or args.task_name == "opamp_metric_prediction":
            if not isnan(sum(val_losses)/len(val_losses)) and not isinf(sum(val_losses)/len(val_losses))\
                and not isnan(sum(train_losses)/len(train_losses)) and not isinf(sum(train_losses)/len(train_losses)):
                if sum(val_losses)/len(val_losses) < min_val_loss:
                    min_val_loss = sum(val_losses)/len(val_losses)
                    model.save(f"./baselines/{args.task_name}/saved_models/{model_name}.pt")

        if epoch % (epochs // 5) == 0:
            model.save(f"./baselines/{args.task_name}/saved_models/{model_name}_epoch{epoch}.pt")

    if args.wandb_log: wandb.finish()
    if args.task_name == "circuit_similarity_prediction":
        model.save(f"./baselines/{args.task_name}/saved_models/{model_name}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="circuit_similarity_prediction")
    parser.add_argument("--baseline_name", type=str, default="DICE")
    parser.add_argument("--print_info", default=0, type=int, help="Print info")
    parser.add_argument("--wandb_log", default=0, type=int, help="Log to wandb")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    train(args)