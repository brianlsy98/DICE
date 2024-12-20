import os
import sys
import copy
import json
import subprocess
from itertools import product
import re
import argparse
import random

from tqdm import tqdm

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pretrain/dataset'))
sys.path.append(parent_dir)

from utils import parse_netlist
from dataloader import *



def generate_dataset(args):
    """
    Run ngspice in batch mode to simulate the circuit.
    """
    # Circuit Dictionary
    with open("../circuits.json", 'r') as f:
        circuit_dictionary = json.loads(f.read())
    print()
    print(circuit_dictionary)

    subtask_dir = f"./{args.task_name}/{args.subtask_name}"

    task_circuits = os.listdir(f"{subtask_dir}/netlists")

    graph_list = []

    print()
    print(f"Generating dataset for {args.subtask_name}...")

    for circuit in task_circuits:
        circuit_dir = f"{subtask_dir}/netlists/{circuit}"
        print(f"Converting Graphs for {circuit}...")
        
        with open(f"{subtask_dir}/sim_template.cir", 'r') as f:
            sim_content = f.read()
        sim_content = sim_content.replace("'netlist_name'", circuit)
        with open("./sim.cir", 'w') as f:
            f.write(sim_content)

        # info from netlist template file
        circuit_name, node_names, nf, node_labels,\
        edge_indices, ef, edge_labels = parse_netlist(f'{circuit_dir}/netlist_template.cir')

        # Read the parameter file
        with open(f"{circuit_dir}/param.json", 'r') as f:
            param = json.load(f)
        param_keys = list(param.keys())
        param_pairs= list(product(*(param[key] for key in param_keys)))

        # For every possible parameter combination
        for param_values in tqdm(param_pairs, desc=f'Sweeping over parameters'):

            # Read the netlist template file
            with open(f"{circuit_dir}/netlist_template.cir", 'r') as f:
                netlist_content = f.read()
            
            ## netlist template file -> parameter selected netlist file
            param_pair = tuple((k, v) for k, v in zip(param_keys, param_values))
            for key, value in param_pair:
                netlist_content = netlist_content.replace(f"'{key}'", str(value))
            with open("./netlist.cir", 'w') as f:
                f.write(netlist_content)

            ## Run ngspice
            result = subprocess.run(['ngspice', '-b', './sim.cir'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True)

            ## Parse ngspice results
            measurements = {}
            # Regex to capture lines like: parameter_name = value
            pattern = re.compile(r'^\s*(\w+)\s*=\s*(\S+)')
            
            for line in result.stdout.splitlines():
                match = pattern.match(line)
                if match:
                    key, val_str = match.groups()
                    if val_str == "failed":
                        measurements[key] = None
                    else:
                        # Try to convert to float if possible
                        try:
                            measurements[key] = float(val_str)
                        except ValueError:
                            measurements[key] = val_str

            ## Make Graph
            graph = GraphData()
            graph.set_node_attributes(torch.from_numpy(nf).float())
            graph.set_node_labels(torch.from_numpy(node_labels).long())
            graph.set_edge_attributes(torch.from_numpy(edge_indices).long(),
                                    torch.from_numpy(ef).float())
            graph.set_edge_labels(torch.from_numpy(edge_labels).long())

            if circuit_name not in list(circuit_dictionary.keys()):
                circuit_dictionary[circuit_name] = len(circuit_dictionary)
                with open("../circuits.json", 'w') as f:
                    f.write(json.dumps(circuit_dictionary, indent=4))
            graph.set_graph_attributes(circuit=circuit_dictionary[circuit_name])


            # set device parameters : if device -> -log(parameter_value), else -> 1
            # mos : W/L, res : resistance, cap : capacitance, ind : inductance
            graph = add_device_params(graph, node_names, param_pair)

            # set graph level simulation results : rise_delay, fall_delay
            graph = add_simulation_results(graph, measurements, args.subtask_name)
            graph_list.append(graph)

            # remove netlist.cir, sim.cir
            os.remove(f"./netlist.cir")
        os.remove(f"./sim.cir")

    # Save the dataset
    torch_data = {}
    path = f"./{args.task_name}/{args.subtask_name}"

    torch_data['name'] = args.subtask_name

    random.shuffle(graph_list)
    torch_data['train_data'] = graph_list[:int(len(graph_list)*args.train_ratio)]
    torch_data['val_data'] = graph_list[int(len(graph_list)*args.train_ratio):int(len(graph_list)*(args.train_ratio+args.val_ratio))]
    torch_data['test_data'] = graph_list[int(len(graph_list)*(args.train_ratio+args.val_ratio)):]

    os.makedirs(path, exist_ok=True)
    torch.save(torch_data, f"{path}/{args.subtask_name}_dataset.pt")

    # remove bsim4v5.out
    os.remove(f"./bsim4v5.out")

    print()
    print(circuit_dictionary)
    print()



def add_device_params(graph, node_names, param_pair):
    # node_name  : ('device1_name', 'device2_name', ...)
    # param_pair : (('param_name', param_value), ...)
    device_params = torch.ones_like(graph.node_y).float()
    for param_name, param_value in param_pair:
        if param_name[0] == "M":
            device_node_index = node_names.index(param_name[:-1])
            if param_name[-1] == "W":
                if device_params[device_node_index] == 1:
                    device_params[device_node_index] *= param_value
                else:
                    device_params[device_node_index] *= param_value
                    device_params[device_node_index] = -torch.log(device_params[device_node_index])
            elif param_name[-1] == "L":
                if device_params[device_node_index] == 1:
                    device_params[device_node_index] /= param_value
                else:
                    device_params[device_node_index] /= param_value
                    device_params[device_node_index] = -torch.log(device_params[device_node_index])
        elif param_name[0] == "V":
            voltage_node_index = node_names.index(param_name)
            device_params[voltage_node_index] = torch.exp(torch.tensor(param_value, dtype=torch.float32))
        else:
            device_node_index = node_names.index(param_name)
            device_params[device_node_index] = -torch.log(torch.tensor(param_value, dtype=torch.float32))

    graph.set_device_params(device_params)

    return graph



def add_simulation_results(graph, measurements, subtask_name):
    if subtask_name == "inv_delay":
        rise_delay = measurements['t_out_rise_edge'] - measurements['t_in_rise_edge']
        fall_delay = measurements['t_out_fall_edge'] - measurements['t_in_fall_edge']
        graph.set_graph_attributes(rise_delay=rise_delay, fall_delay=fall_delay)

    return graph



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="digital")
    parser.add_argument("--subtask_name", type=str, default="delay_prediction")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    generate_dataset(args)