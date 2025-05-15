import os
import sys
import json
import random
import argparse
from tqdm import tqdm

import torch

from torch.amp import autocast

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream_tasks'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed
from dataloader import GraphDataLoader

from downstream_test import get_test_info, process_info_seed, print_info

from baseline_models import BaselineModel



def test(args):

    with open("./params.json", 'r') as f:
        params = json.load(f)

        ### Dataset
        if args.baseline_name == "DeepGen_u"\
        or args.baseline_name == "DeepGen_p":
            test_data = torch.load(f'./baselines/{args.task_name}/{args.task_name}_DeepGen_test.pt')
        elif args.baseline_name == "DICE":
            test_data = torch.load(f'./downstream_tasks/{args.task_name}/{args.task_name}_test.pt')
        else:
            test_data = torch.load(f'./baselines/{args.task_name}/{args.task_name}_ParaGraph_test.pt')
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

    total_info = []
    for trained_seed in args.trained_model_seeds:

        ### Model
        ########################
        model = BaselineModel(args, params['baseline_model'])
        if args.baseline_name == "DICE":
            model_name = f"{args.task_name}_{args.baseline_name}_taup02tau005taun005_seed{trained_seed}"
        else:
            model_name = f"{args.task_name}_{args.baseline_name}_seed{trained_seed}"
        model.apply(init_weights)
        model.load(f"./baselines/{args.task_name}/saved_models/{model_name}.pt")
        model = model.to(params['downstream_tasks'][f'{args.task_name}']['test']['device'])
        model.eval()
        ########################
        print()
        print("Model Initialized")
        print("Model Name:", model_name)
        print("parameter num:", sum(p.numel() for p in model.parameters()))

        infos = []
        ### Test
        for s in params['downstream_tasks'][f'{args.task_name}']['test']['seeds']:

                os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                set_seed(s, False)
                
                test_infos = []
                with torch.no_grad():
                    for test_batch in test_dataloader:
                        test_batch = send_to_device(test_batch, params['downstream_tasks'][f'{args.task_name}']['test']['device'])
                        # if args.task_name == "circuit_similarity_prediction" and test_batch['batch'].max() < 2: continue
                        ############################
                        with autocast(device_type=params['downstream_tasks'][f'{args.task_name}']['test']["device"]):
                            out = model(test_batch)
                            test_info = get_test_info(out, test_batch, args.task_name)
                        ############################
                        test_infos.append(test_info)
                test_infos = torch.concat(test_infos, dim=0)
                infos.append(process_info_seed(args, test_infos, s, params))
                # print(infos)
        infos = torch.stack(infos, dim=0)
        total_info.append(infos)
        print_info(args, infos)
        print()
    total_info = torch.concat(total_info, dim=0)
    print()
    print(" TOTAL ")
    print_info(args, total_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="circuit_similarity_prediction")
    parser.add_argument("--baseline_name", type=str, default="DICE")
    parser.add_argument("--trained_model_seeds", nargs='+', type=int, help="Seeds for trained model")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--print_info", type=int, default=0, help="Print information")
    args = parser.parse_args()
    test(args)