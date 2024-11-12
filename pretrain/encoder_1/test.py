import os
import sys
import json
import argparse
import numpy as np

import torch



parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/run_files'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/data/torch_datasets'))
sys.path.append(parent_dir)

from dataloader import HeteroGraphDataLoader


from model import *
from utils import *


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

        test_data = data_set['val_data']
        print("test_data size : ", len(test_data))

        # Load Parameters
        with open(f"./model_params.json", 'r') as f:
            model_params = json.load(f)
        with open(f"./test_params.json", 'r') as f:
            test_params = json.load(f)

        ########################
        model = PretrainModel(model_params)
        model = model.to(test_params["device"])
        model = model.eval()
        ########################
        test_dataloader = HeteroGraphDataLoader(test_data, batch_size=1, shuffle=True)

        total_node_count = 0
        counts = [0, 0, 0, 0, 0, 0]

        for test_batch in test_dataloader:

            test_batch = send_to_device(test_batch, test_params["device"])

            with torch.no_grad():
                output, e_info, f_info = model(test_batch)

            x = output - test_batch['output']['dc_voltages']

            total_node_count += len(x)

            for ele in x:
                if torch.abs(ele) < 0.1:
                    counts[0] += 1
                if torch.abs(ele) < 0.3:
                    counts[1] += 1
                if torch.abs(ele) < 0.5:
                    counts[2] += 1
                if torch.abs(ele) < 0.7:
                    counts[3] += 1
                if torch.abs(ele) < 0.9:
                    counts[4] += 1
                if torch.abs(ele) < 2.0:
                    counts[5] += 1


        print(f"Total Node Count: {total_node_count}")
        print(f"within 0.1 : {100*counts[0]/total_node_count:.2f}%")
        print(f"within 0.3 : {100*counts[1]/total_node_count:.2f}%")
        print(f"within 0.5 : {100*counts[2]/total_node_count:.2f}%")
        print(f"within 0.7 : {100*counts[3]/total_node_count:.2f}%")
        print(f"within 0.9 : {100*counts[4]/total_node_count:.2f}%")
        print(f"within 2.0 : {100*counts[5]/total_node_count:.2f}%")
