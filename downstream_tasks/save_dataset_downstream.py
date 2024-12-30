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

from utils import parse_netlist, data_augmentation
from dataloader import GraphData



def incomplete_simulation(graph_attrs, task_name):
    if task_name == "delay_prediction":
        if graph_attrs['minus_log_rise_delay'] is None or graph_attrs['minus_log_fall_delay'] is None:
            return True
        elif torch.isnan(graph_attrs['minus_log_rise_delay']) or torch.isnan(graph_attrs['minus_log_fall_delay']):
            return True
        elif torch.isinf(graph_attrs['minus_log_rise_delay']) or torch.isinf(graph_attrs['minus_log_fall_delay']):
            return True
        else: return False
    else:
        raise ValueError("Invalid Subtask Name")



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

    if subtask_name == "delay_prediction":
        rise_delay = measurements['t_out_rise_edge'] - measurements['t_in_rise_edge']
        fall_delay = measurements['t_out_fall_edge'] - measurements['t_in_fall_edge']
        minus_log_rise_delay = -torch.log(torch.tensor(rise_delay, dtype=torch.float32))
        minus_log_fall_delay = -torch.log(torch.tensor(fall_delay, dtype=torch.float32))
        graph.set_graph_attributes(minus_log_rise_delay=minus_log_rise_delay,
                                   minus_log_fall_delay=minus_log_fall_delay)

    return graph



###############################################################################################################
# Generate Dataset for Downstream Task: Delay Prediction
def gen_data_delay_prediction(args, circuit_dictionary):
    print()
    print(f"Generating dataset for delay_prediction...")

    graph_list = []

    task_dir = f"./downstream_tasks/delay_prediction"
    task_circuits = os.listdir(f"{task_dir}/device_parameters")

    for circuit in task_circuits:
        if circuit in os.listdir("./circuits/no_pretrain"):
            circuit_dir = f"./circuits/no_pretrain/{circuit}"
        elif circuit in os.listdir("./circuits/pretrain"):
            circuit_dir = f"./circuits/pretrain/{circuit}"
        else: raise ValueError("Invalid Circuit")

        device_param_dir = f"{task_dir}/device_parameters/{circuit}"
        print(f"Converting Graphs for {circuit}...")
        
        with open(f"{task_dir}/sim_template.cir", 'r') as f:
            sim_content = f.read()
        sim_content = sim_content.replace("'netlist_name'", circuit)
        with open(f"{task_dir}/sim.cir", 'w') as f:
            f.write(sim_content)

        # info from netlist template file
        circuit_name, node_names, nf, node_labels,\
        edge_indices, ef, edge_labels = parse_netlist(f'{circuit_dir}/netlist.cir')

        # Read the parameter file
        with open(f"{device_param_dir}/param.json", 'r') as f:
            param = json.load(f)
        param_keys = list(param.keys())
        param_pairs= list(product(*(param[key] for key in param_keys)))

        # For every possible parameter combination
        for param_values in tqdm(param_pairs, desc=f'Sweeping over parameters'):

            # Read the netlist template file
            with open(f"{circuit_dir}/netlist.cir", 'r') as f:
                netlist_content = f.read()
            
            ## netlist template file -> parameter selected netlist file
            param_pair = tuple((k, v) for k, v in zip(param_keys, param_values))
            for key, value in param_pair:       
                netlist_content = netlist_content.replace(f"'{key}'", str(value))
            with open(f"{task_dir}/netlist.cir", 'w') as f:
                f.write(netlist_content)

            ## Run ngspice
            result = subprocess.run(['ngspice', '-b', f'{task_dir}/sim.cir'], 
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
                circuit_dictionary[circuit_name] = len(circuit_dictionary)+1
            graph.set_graph_attributes(circuit=torch.tensor(circuit_dictionary[circuit_name], dtype=torch.long))


            # set device parameters : if device -> -log(parameter_value), else -> 1
            # mos : W/L, res : resistance, cap : capacitance, ind : inductance
            graph = add_device_params(graph, node_names, param_pair)

            # set graph level simulation results : rise_delay, fall_delay
            graph = add_simulation_results(graph, measurements, args.task_name)

            # remove netlist.cir, sim.cir
            os.remove(f"{task_dir}/netlist.cir")

            # go for another loop if simulation is not complete
            if incomplete_simulation(graph.graph_attrs, args.task_name): continue

            graph_list.append(graph)

        # remove simulation files
        os.remove(f"{task_dir}/sim.cir")
        os.remove(f"./bsim4v5.out")

    return graph_list, circuit_dictionary
###############################################################################################################



###############################################################################################################
# Generate Dataset for Downstream Task: Circuit Similarity Prediction
def gen_data_circuit_similarity_prediction(args, circuit_dictionary):
    print()
    print(f"Generating dataset for circuit_similarity_prediction...")
    graph_list = []

    # Circuit Labels
    task_dir = f"./downstream_tasks/circuit_similarity_prediction"
    with open(f"{task_dir}/circuit_labels.json", 'r') as f:
        circuit_labels = json.loads(f.read())
    print()
    print(circuit_labels)
    print()

    # Task Circuits
    task_circuits = os.listdir("./circuits/no_pretrain")\
                  + os.listdir("./circuits/pretrain")

    for circuit in tqdm(task_circuits):
        if circuit in os.listdir("./circuits/no_pretrain"):
            circuit_dir = f"./circuits/no_pretrain/{circuit}"
        elif circuit in os.listdir("./circuits/pretrain"):
            circuit_dir = f"./circuits/pretrain/{circuit}"

        # info from netlist template file
        circuit_name, node_names, nf, node_labels,\
        edge_indices, ef, edge_labels = parse_netlist(f'{circuit_dir}/netlist.cir')

        ## Make Graph
        graph = GraphData()
        graph.set_node_attributes(torch.from_numpy(nf).float())
        graph.set_node_labels(torch.from_numpy(node_labels).long())
        graph.set_edge_attributes(torch.from_numpy(edge_indices).long(),
                                torch.from_numpy(ef).float())
        graph.set_edge_labels(torch.from_numpy(edge_labels).long())

        if circuit_name not in list(circuit_dictionary.keys()):
            circuit_dictionary[circuit_name] = len(circuit_dictionary)+1
        graph.set_graph_attributes(circuit=torch.tensor(circuit_dictionary[circuit_name], dtype=torch.long))

        labels = []
        for label in circuit_labels.keys():
            if circuit_name in circuit_labels[label]:
                labels.append(1)
            else:
                labels.append(0)
        graph.set_graph_attributes(labels=torch.tensor(labels, dtype=torch.long))

        graphs = [graph]

        # Data Augmentation
        for _ in range(999):
            g = random.choice(graphs)
            new_graph = data_augmentation(g, "pos", 1)
            graphs.extend(new_graph)

        graph_list.extend(graphs)

    return graph_list, circuit_dictionary
###############################################################################################################



###############################################################################################################
def generate_dataset(args):

    # Circuit Dictionary
    with open("./circuits/circuits.json", 'r') as f:
        circuit_dictionary = json.loads(f.read())
    print()
    print(circuit_dictionary)

    ### Generate Dataset
    if args.task_name == "delay_prediction":
        graph_list, circuit_dictionary = gen_data_delay_prediction(args, circuit_dictionary)
    elif args.task_name == "circuit_similarity_prediction":
        graph_list, circuit_dictionary = gen_data_circuit_similarity_prediction(args, circuit_dictionary)

    # Save the dataset
    random.shuffle(graph_list)
    train_dataset = graph_list[:int(len(graph_list)*args.train_ratio)]
    val_dataset = graph_list[int(len(graph_list)*args.train_ratio):int(len(graph_list)*(args.train_ratio+args.val_ratio))]
    test_dataset = graph_list[int(len(graph_list)*(args.train_ratio+args.val_ratio)):]

    path = f"./downstream_tasks/{args.task_name}"
    os.makedirs(path, exist_ok=True)
    torch.save(train_dataset, f"{path}/{args.task_name}_train.pt")
    torch.save(val_dataset, f"{path}/{args.task_name}_val.pt")
    torch.save(test_dataset, f"{path}/{args.task_name}_test.pt")


    with open("./circuits/circuits.json", 'w') as f:
        f.write(json.dumps(circuit_dictionary, indent=4))
    print()
    print(circuit_dictionary)
    print()
###############################################################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="delay_prediction")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    generate_dataset(args)