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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils_baseline import parse_netlist_baseline
from dataloader import GraphData


def incomplete_simulation(graph):
    if torch.isnan(graph.x).any() or torch.isnan(graph.edge_attr).any():
        return True
    elif torch.isinf(graph.x).any() or torch.isinf(graph.edge_attr).any():
        return True
    else:
        return False




def add_device_params(graph, node_names, edge_names, param_pair, baseline_name):
    # node_name  : ('device1_name', 'device2_name', ...)
    # param_pair : (('param_name', param_value), ...)


    #####################################
    ########## DeepGen_pretrain #########
    #####################################
    if baseline_name == "DeepGen":

        device_params = torch.ones_like(graph.node_y).float()
        for param_name, param_value in param_pair:
            if param_name[0] == "M":
                for mosfet_nodes in ['d', 'g', 's', 'b']:
                    device_node_index = node_names.index(f"{param_name[:-1]}_{mosfet_nodes}")
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
                if param_name in node_names:
                    voltage_node_index = node_names.index(param_name)
                else: 
                    voltage_node_index = node_names.index(param_name.lower())
                device_params[voltage_node_index] = torch.exp(torch.tensor(param_value, dtype=torch.float32))
            else:
                for i in range(1, 3):
                    device_node_index = node_names.index(f"{param_name}_{i}")
                    device_params[device_node_index] = -torch.log(torch.tensor(param_value, dtype=torch.float32))

        graph.set_device_params(device_params)


    return graph




def add_simulation_results(graph, node_names, measurements):

    for node_name, node_voltage in measurements.items():
        if node_name == '0': node_index = node_names.index('gnd')
        else: node_index = node_names.index(node_name)
        graph.device_params[node_index] = torch.exp(torch.tensor(node_voltage, dtype=torch.float32))
    return graph


###############################################################################################################
# Generate Dataset for Downstream Task: Delay Prediction, Opamp Metric Prediction
def gen_data_simresult_prediction(args, circuit_dictionary):

    graph_list = []

    task_dir = f"./baselines/pretrain_DCvoltages"
    task_circuits = os.listdir(f"{task_dir}/device_parameters")

    for circuit in task_circuits:
        if circuit in os.listdir(f"./circuits/pretrain"):
            circuit_dir = f"./circuits/pretrain/{circuit}"
        elif circuit in os.listdir(f"./circuits/downstreamtrain_nopretrain"):
            circuit_dir = f"./circuits/downstreamtrain_nopretrain/{circuit}"
        device_param_dir = f"{task_dir}/device_parameters/{circuit}"
        print(f"Converting Graphs for {circuit}...")
        
        # info from netlist template file
        circuit_name, node_names, edge_names, nf, node_labels,\
        edge_indices, ef, edge_labels = parse_netlist_baseline(
            f'{circuit_dir}/netlist.cir', "DeepGen"
        )

        # Read the parameter file
        with open(f"{device_param_dir}/param.json", 'r') as f:
            param = json.load(f)
        param_keys = list(param.keys())
        param_pairs= list(product(*(param[key] for key in param_keys)))

        # For every possible parameter combination
        for param_values in tqdm(param_pairs, desc=f'Sweeping over parameters'):

            # Read the netlist template file
            with open(f"{device_param_dir}/sim_template.cir", 'r') as f:
                sim_content = f.read()
            
            ## sim template file -> parameter selected netlist file
            param_pair = tuple((k, v) for k, v in zip(param_keys, param_values))
            for key, value in param_pair:
                sim_content = sim_content.replace(f"'{key}'", str(value))
            with open(f"{task_dir}/sim.cir", 'w') as f:
                f.write(sim_content)

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
            # print(measurements)
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
            graph = add_device_params(graph, node_names, edge_names, param_pair, "DeepGen")

            # set graph level simulation results : -log(rise_delay), -log(fall_delay), op_amp power, etc.
            graph = add_simulation_results(graph, node_names, measurements)

            # remove netlist.cir, sim.cir
            if os.path.exists(f"{task_dir}/netlist.cir"):
                os.remove(f"{task_dir}/netlist.cir")

            # go for another loop if simulation is not complete
            if incomplete_simulation(graph):
                continue

            graph_list.append(graph)

        # remove simulation files
        os.remove(f"{task_dir}/sim.cir")
        os.remove(f"./bsim4v5.out")

    random.shuffle(graph_list)
    graph_list_train = graph_list[:int(len(graph_list)*args.train_ratio)]
    graph_list_val = graph_list[int(len(graph_list)*args.train_ratio):int(len(graph_list)*(args.train_ratio+args.val_ratio))]
    graph_list_test = graph_list[int(len(graph_list)*(args.train_ratio+args.val_ratio)):]

    return graph_list_train, graph_list_val, graph_list_test, circuit_dictionary
###############################################################################################################



###############################################################################################################
def generate_dataset(args):

    # Circuit Dictionary
    with open("./circuits/circuits.json", 'r') as f:
        circuit_dictionary = json.loads(f.read())
    print()
    print(circuit_dictionary)

    ### Generate Dataset
    print()
    print(f"Generating dataset for DC voltage prediction...")
    graph_list_train, graph_list_val, graph_list_test, circuit_dictionary = gen_data_simresult_prediction(args, circuit_dictionary)


    # Save the dataset
    random.shuffle(graph_list_train); random.shuffle(graph_list_val); random.shuffle(graph_list_test)
    path = f"./baselines/pretrain_DCvoltages"
    os.makedirs(path, exist_ok=True)
    torch.save(graph_list_train, f"{path}/dcvoltage_prediction_train.pt")
    torch.save(graph_list_val, f"{path}/dcvoltage_prediction_val.pt")
    torch.save(graph_list_test, f"{path}/dcvoltage_prediction_test.pt")


    with open("./circuits/circuits.json", 'w') as f:
        f.write(json.dumps(circuit_dictionary, indent=4))
    print()
    print(circuit_dictionary)
    print()
###############################################################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    generate_dataset(args)