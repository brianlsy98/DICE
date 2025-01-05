import os
import sys
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torcheval.metrics.functional import r2_score

from torch.amp import autocast

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed
from dataloader import GraphDataLoader

from downstream_model import DownstreamModel



def get_test_info(out, batch, task_name):

    if task_name == "circuit_similarity_prediction":
        # out: (batch_size, 1)
        # batch['labels']: (batch_size, label_num)
        labels = batch['labels'].float()  # (batch_size, label_num)
        batch_size = labels.size(0)

        sim = torch.mm(labels[1:], labels[0].unsqueeze(-1))   # (batch_size-1, 1)
        sim = torch.softmax(sim, dim=0).squeeze(-1)           # (batch_size-1,)

        out_squeezed = out.squeeze(-1)  # (batch_size-1,)

        correct_pairs = 0
        total_pairs = 0

        for i in range(batch_size-1):
            for j in range(i+1, batch_size-1):

                if sim[i] == sim[j] and abs(out_squeezed[i] - out_squeezed[j]) < 0.1/(batch_size-1): correct_pairs += 1
                elif sim[i] > sim[j] and out_squeezed[i] > out_squeezed[j]: correct_pairs += 1
                elif sim[i] < sim[j] and out_squeezed[i] < out_squeezed[j]: correct_pairs += 1

                total_pairs += 1
        
        accuracy = correct_pairs / total_pairs
        return torch.tensor([accuracy], dtype=torch.float16, device=out.device)

    elif task_name == "circuit_label_prediction":
        # out : (batch_size, label_num), batch['labels'] : (batch_size, label_num)
        prediction = torch.round(out)
        accuracies = (prediction == batch['labels']).float()
        return accuracies
    
    elif task_name == "delay_prediction":
        rise_delay = batch['minus_log_rise_delay']
        fall_delay = batch['minus_log_fall_delay']
        delays = torch.stack([rise_delay, fall_delay], dim=1)
        # print(torch.cat([torch.exp(-out), torch.exp(-delays)], dim=1))
        ##### remind that the output is minus log value of the delays
        return torch.cat([torch.exp(-out), torch.exp(-delays)], dim=1)

    elif task_name == "opamp_metric_prediction":
        power = batch['power']
        voutdc_minus_vindc = batch['voutdc_minus_vindc']
        cmrr_dc = batch['cmrr_dc']
        gain_dc = batch['gain_dc']
        vddpsrr_dc = batch['vddpsrr_dc']
        metrics = torch.stack([power, voutdc_minus_vindc, cmrr_dc, gain_dc, vddpsrr_dc], dim=1)
        # power: *1e3, voutdc_minus_vindc: *10, cmrr_dc: /10, gain_dc: /10, vddpsrr_dc: /10
        dividing_tensor = torch.tensor([1e3, 10, 0.1, 0.1, 0.1], device=metrics.device)
        return torch.cat([out/dividing_tensor, metrics/dividing_tensor], dim=1)

    else: raise ValueError("Invalid Subtask Name")



def process_info_seed(test_infos, seed, params, print_info=1):

    if args.task_name == "circuit_label_prediction":
        analog_acc = 100*test_infos[:, 0].mean().item()
        digital_acc = 100*test_infos[:, 1].mean().item()
        amplifiers_acc = 100*test_infos[:, 2].mean().item()
        logic_gates_acc = 100*test_infos[:, 3].mean().item()
        if print_info:
            print()
            print(f"Seed: {seed}/{params['downstream_tasks'][f'{args.task_name}']['test']['seeds']}")
            print(f"Label (Analog) Accuracy     : {analog_acc:.2f}%")
            print(f"Label (Digital) Accuracy    : {digital_acc:.2f}%")
            print(f"Label (Amplifiers) Accuracy : {amplifiers_acc:.2f}%")
            print(f"Label (Logic Gates) Accuracy: {logic_gates_acc:.2f}%")
        return torch.tensor([analog_acc, digital_acc, amplifiers_acc, logic_gates_acc])

    elif args.task_name == "circuit_similarity_prediction":
        acc = 100*test_infos.mean().item()
        if print_info:
            print()
            print(f"Seed: {seed}/{params['downstream_tasks'][f'{args.task_name}']['test']['seeds']}")
            print(f"Similarity Prediction Accuracy: {acc:.2f}%")
        return torch.tensor([acc])

    elif args.task_name == "delay_prediction":
        r2_score_rise_delay = r2_score(test_infos[:, 0], test_infos[:, 2])
        r2_score_fall_delay = r2_score(test_infos[:, 1], test_infos[:, 3])
        if print_info:
            print()
            print(f"Seed: {seed}/{params['downstream_tasks'][f'{args.task_name}']['test']['seeds']}")
            print(f"Rise Delay R2 Score: {r2_score_rise_delay:.2f}")
            print(f"Fall Delay R2 Score: {r2_score_fall_delay:.2f}")
        return torch.tensor([r2_score_rise_delay, r2_score_fall_delay])

    elif args.task_name == "opamp_metric_prediction":
        r2_score_power = r2_score(test_infos[:, 0], test_infos[:, 5])
        r2_score_voutdc_minus_vindc = r2_score(test_infos[:, 1], test_infos[:, 6])
        r2_score_cmrr_dc = r2_score(test_infos[:, 2], test_infos[:, 7])
        r2_score_gain_dc = r2_score(test_infos[:, 3], test_infos[:, 8])
        r2_score_vddpsrr_dc = r2_score(test_infos[:, 4], test_infos[:, 9])
        if print_info:
            print()
            print(f"Seed: {seed}/{params['downstream_tasks'][f'{args.task_name}']['test']['seeds']}")
            print(f"Power R2 Score: {r2_score_power:.2f}")
            print(f"Voutdc-Vindc R2 Score: {r2_score_voutdc_minus_vindc:.2f}")
            print(f"CMRR R2 Score: {r2_score_cmrr_dc:.2f}")
            print(f"Gain R2 Score: {r2_score_gain_dc:.2f}")
            print(f"Vddpsrr R2 Score: {r2_score_vddpsrr_dc:.2f}")
        return torch.tensor([r2_score_power, r2_score_voutdc_minus_vindc, r2_score_cmrr_dc, r2_score_gain_dc, r2_score_vddpsrr_dc])

    else: raise ValueError("Invalid Subtask Name")



def print_info(infos):

    if args.task_name == "circuit_label_prediction":
        mean, std = infos.mean(dim=0), infos.std(dim=0)
        print("############# Label Prediction #############")
        print(f"Label (Analog) Accuracy     : {mean[0]:.2f}% ± {std[0]:.2f}")
        print(f"Label (Digital) Accuracy    : {mean[1]:.2f}% ± {std[1]:.2f}")
        print(f"Label (Amplifiers) Accuracy : {mean[2]:.2f}% ± {std[2]:.2f}")
        print(f"Label (Logic Gates) Accuracy: {mean[3]:.2f}% ± {std[3]:.2f}")
        print("############################################")

    elif args.task_name == "circuit_similarity_prediction":
        mean, std = infos.mean(dim=0), infos.std(dim=0)
        print("############# Similarity Prediction #############")
        print(f"Similarity Prediction Accuracy: {mean[0]:.2f}% ± {std[0]:.2f}")
        print("#################################################")

    elif args.task_name == "delay_prediction":
        mean, std = infos.mean(dim=0), infos.std(dim=0)
        print("############# Delay Prediction #############")
        print(f"Rise Delay R2 Score: {mean[0]:.2f} ± {std[0]:.2f}")
        print(f"Fall Delay R2 Score: {mean[1]:.2f} ± {std[1]:.2f}")
        print("############################################")

    elif args.task_name == "opamp_metric_prediction":
        mean, std = infos.mean(dim=0), infos.std(dim=0)
        print("############# Opamp Metric Prediction #############")
        print(f"Power R2 Score: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Voutdc-Vindc R2 Score: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"CMRR R2 Score: {mean[2]:.4f} ± {std[2]:.4f}")
        print(f"Gain R2 Score: {mean[3]:.4f} ± {std[3]:.4f}")
        print(f"Vddpsrr R2 Score: {mean[4]:.4f} ± {std[4]:.4f}")
        print("###################################################")
    
    else: raise ValueError("Invalid Subtask Name")



def test(args):

    with open("./params.json", 'r') as f:
        params = json.load(f)
    tau = args.tau.replace(".", ""); tau_tn = args.tautn.replace(".", "")

    ### Dataset
    test_data = torch.load(f'./downstream_tasks/{args.task_name}/{args.task_name}_test.pt')
    print()
    print("Dataset Loaded")
    print("test_data:", len(test_data))
    print()
    ### Dataloader
    # test dataloader
    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=params['downstream_tasks'][f'{args.task_name}']['test']['batch_size'],
        shuffle=True
    )
    print()
    print("Dataloader Initialized")
    print("test_data:", len(test_data))
    print()

    ### Model
    ########################
    model = DownstreamModel(args, params['model'])
    if args.dice_depth == 0:
        model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                        f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                        f"_sGNN{args.s_gnn_depth}_seed{args.trained_model_seed}"
    else:
        model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                        f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                        f"_sGNN{args.s_gnn_depth}_tau{tau}_tautn{tau_tn}_seed{args.trained_model_seed}"
    model.apply(init_weights)
    model.load(f"./downstream_tasks/{args.task_name}/{model_name}.pt")
    model = model.to(params['downstream_tasks'][f'{args.task_name}']['test']['device'])
    model.eval()
    ########################
    print()
    print("Model Initialized")
    print("Model Name:", model_name)
    print("parameter num:", sum(p.numel() for p in model.parameters()))
    print(f"Seeds: {params['downstream_tasks'][f'{args.task_name}']['test']['seeds']}")

    infos = []
    ### Test
    for s in range(params['downstream_tasks'][f'{args.task_name}']['test']['seeds']):

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        set_seed(s, False)

        test_infos = []
        with torch.no_grad():
            for test_batch in test_dataloader:
                test_batch = send_to_device(test_batch, params['downstream_tasks'][f'{args.task_name}']['test']['device'])
                if args.task_name == "circuit_similarity_prediction" and test_batch['batch'].max() < 2: continue
                ############################
                with autocast(device_type=params['downstream_tasks'][f'{args.task_name}']['test']["device"]):
                    out = model(test_batch)
                    test_info = get_test_info(out, test_batch, args.task_name)
                ############################
                test_infos.append(test_info)
        test_infos = torch.concat(test_infos, dim=0)
        infos.append(process_info_seed(test_infos, s, params, print_info=args.print_info))

    infos = torch.stack(infos, dim=0)
    print_info(infos)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="circuit_similarity_prediction")
    parser.add_argument("--dice_depth", type=int, default=0, help="depth for DICE")
    parser.add_argument("--p_gnn_depth", type=int, default=0, help="depth for parallel GNN")
    parser.add_argument("--s_gnn_depth", type=int, default=0, help="depth for series GNN")
    parser.add_argument("--tau", type=str, default="0.05", help="tau value for DICE")
    parser.add_argument("--tautn", type=str, default="0.05", help="tau_tn value for DICE")
    parser.add_argument("--trained_model_seed", type=int, default=0, help="Seed for trained model")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--print_info", type=int, default=1, help="Print information")
    args = parser.parse_args()
    test(args)