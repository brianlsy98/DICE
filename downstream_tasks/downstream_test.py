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

import itertools

def get_test_info(out, batch, task_name):

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

        accuracy = torch.mean((torch.argmax(out, dim=1) == target).float()).unsqueeze(0)

        return accuracy


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



def process_info_seed(args, test_infos, seed, params):

    if args.task_name == "circuit_similarity_prediction":
        acc = 100*test_infos.mean().item()
        if args.print_info:
            print()
            print(f"Similarity Prediction Accuracy: {acc:.2f}%")
        return torch.tensor([acc])

    elif args.task_name == "delay_prediction":
        r2_score_rise_delay = r2_score(test_infos[:, 0], test_infos[:, 2])
        r2_score_fall_delay = r2_score(test_infos[:, 1], test_infos[:, 3])
        if args.print_info:
            print()
            print(f"Rise Delay R2 Score: {r2_score_rise_delay:.4f}")
            print(f"Fall Delay R2 Score: {r2_score_fall_delay:.4f}")
        return torch.tensor([r2_score_rise_delay, r2_score_fall_delay])

    elif args.task_name == "opamp_metric_prediction":
        r2_score_power = r2_score(test_infos[:, 0], test_infos[:, 5])
        r2_score_voutdc_minus_vindc = r2_score(test_infos[:, 1], test_infos[:, 6])
        r2_score_cmrr_dc = r2_score(test_infos[:, 2], test_infos[:, 7])
        r2_score_gain_dc = r2_score(test_infos[:, 3], test_infos[:, 8])
        r2_score_vddpsrr_dc = r2_score(test_infos[:, 4], test_infos[:, 9])
        if args.print_info:
            print()
            print(f"Power R2 Score: {r2_score_power:.4f}")
            print(f"Voutdc-Vindc R2 Score: {r2_score_voutdc_minus_vindc:.4f}")
            print(f"CMRR R2 Score: {r2_score_cmrr_dc:.4f}")
            print(f"Gain R2 Score: {r2_score_gain_dc:.4f}")
            print(f"Vddpsrr R2 Score: {r2_score_vddpsrr_dc:.4f}")
        return torch.tensor([r2_score_power, r2_score_voutdc_minus_vindc, r2_score_cmrr_dc, r2_score_gain_dc, r2_score_vddpsrr_dc])

    else: raise ValueError("Invalid Subtask Name")



def print_info(args, infos):

    if args.task_name == "circuit_similarity_prediction":
        mean, std = infos.mean(dim=0), infos.std(dim=0)
        print("############# Similarity Prediction #############")
        print(f"Similarity Prediction Accuracy: {mean[0]:.2f}% ± {std[0]:.2f}")
        print("#################################################")

    elif args.task_name == "delay_prediction":
        mean, std = infos.mean(dim=0), infos.std(dim=0)
        print("############# Delay Prediction #############")
        print(f"Rise Delay R2 Score: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Fall Delay R2 Score: {mean[1]:.4f} ± {std[1]:.4f}")
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
    taup=args.taup.replace(".", ""); tau=args.tau.replace(".", ""); taun = args.taun.replace(".", "")

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
        if args.cl_type == 'nda':
            model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                        f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                        f"_sGNN{args.s_gnn_depth}_taup{taup}tau{tau}taun{taun}_seed{args.trained_model_seed}"
        elif args.cl_type == 'pda':
            model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                            f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                            f"_sGNN{args.s_gnn_depth}_tau{tau}_pda_seed{args.trained_model_seed}"
        elif args.cl_type == 'simsiam':
            model_name = f"{args.task_name}_{params['model']['encoder']['dice']['gnn_type']}"\
                            f"_DICE{args.dice_depth}_pGNN{args.p_gnn_depth}"\
                            f"_sGNN{args.s_gnn_depth}_simsiam_seed{args.trained_model_seed}"
    model.apply(init_weights)
    model.load(f"./downstream_tasks/{args.task_name}/saved_models/{model_name}.pt")
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
    for s in params['downstream_tasks'][f'{args.task_name}']['test']['seeds']:

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
        infos.append(process_info_seed(args, test_infos, s, params))

    infos = torch.stack(infos, dim=0)
    print_info(args, infos)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="circuit_similarity_prediction")
    parser.add_argument("--dice_depth", type=int, default=2, help="depth for DICE")
    parser.add_argument("--p_gnn_depth", type=int, default=0, help="depth for parallel GNN")
    parser.add_argument("--s_gnn_depth", type=int, default=2, help="depth for series GNN")
    parser.add_argument("--taup", type=str, default="0.2", help="taup value for DICE")
    parser.add_argument("--tau", type=str, default="0.05", help="tau value for DICE")
    parser.add_argument("--taun", type=str, default="0.05", help="taun value for DICE")
    parser.add_argument("--cl_type", default='nda', choices=['nda', 'pda', 'simsiam'], help="NT-Xent vs SimSiam vs Ours")
    parser.add_argument("--trained_model_seed", type=int, default=0, help="Seed for trained model")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--print_info", type=int, default=1, help="Print information")
    args = parser.parse_args()
    test(args)