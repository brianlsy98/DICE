import os
import sys
import argparse
import random
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import parse_netlist, data_augmentation
from dataloader import GraphData



def save_dataset(args):

    circuit_dictionary = {}

    ############################################################################################################
    init_graphs = dict()

    # Dataset before Augmentation
    for circuit in os.listdir("./circuits/pretrain"):

        netlist = f"./circuits/pretrain/{circuit}/netlist.cir"
        circuit_name, node_names, nf, node_labels, edge_indices, ef, edge_labels = parse_netlist(netlist)

        graph = GraphData()
        graph.set_node_attributes(torch.from_numpy(nf).float())
        graph.set_node_labels(torch.from_numpy(node_labels).long())
        graph.set_edge_attributes(torch.from_numpy(edge_indices).long(),
                                  torch.from_numpy(ef).float())
        graph.set_edge_labels(torch.from_numpy(edge_labels).long())

        if circuit_name not in list(circuit_dictionary.keys()):
            circuit_dictionary[circuit_name] = len(circuit_dictionary)+1

        graph.set_graph_attributes(circuit=circuit_dictionary[circuit_name])

        init_graphs[circuit_name] = graph


    # Data Augmentation
    train_dataset, val_dataset, test_dataset = {}, {}, {}

    for circuit_name, init_graph in tqdm(init_graphs.items()):
        
        circuit_i_train_graphs = {"pos": [init_graph],
                                  "neg": data_augmentation(init_graph, "neg", 1)}
        circuit_i_val_graphs   = {"pos": [init_graph],
                                  "neg": data_augmentation(init_graph, "neg", 1)}
        circuit_i_test_graphs  = {"pos": [init_graph],
                                  "neg": data_augmentation(init_graph, "neg", 1)}

        # Train Graph Data Augmentation
        while len(circuit_i_train_graphs["pos"]) < args.datasize_per_circuit//2*args.train_ratio:
            pos_graph = random.choice(circuit_i_train_graphs["pos"])
            new_pos_graphs = data_augmentation(pos_graph, "pos", 1)
            new_neg_graphs = data_augmentation(pos_graph, "neg", 1)
            circuit_i_train_graphs["pos"].extend(new_pos_graphs)
            circuit_i_train_graphs["neg"].extend(new_neg_graphs)
        train_dataset[circuit_name] = circuit_i_train_graphs
        print(f"Training Set Data Augmentation ({circuit_name}): size {args.datasize_per_circuit*args.train_ratio}")
    ############################################################################################################


    ############################################################################################################
    # Add unseen circuits to test dataset
    unseen_graphs = dict()

    # Dataset before Augmentation
    for circuit in os.listdir("./circuits/downstreamtrain_nopretrain")+os.listdir("./circuits/untrained"):
        if circuit in os.listdir("./circuits/downstreamtrain_nopretrain"):
            netlist = f"./circuits/downstreamtrain_nopretrain/{circuit}/netlist.cir"
        elif circuit in os.listdir("./circuits/untrained"):
            netlist = f"./circuits/untrained/{circuit}/netlist.cir"
        circuit_name, node_names, nf, node_labels, edge_indices, ef, edge_labels = parse_netlist(netlist)

        graph = GraphData()
        graph.set_node_attributes(torch.from_numpy(nf).float())
        graph.set_node_labels(torch.from_numpy(node_labels).long())
        graph.set_edge_attributes(torch.from_numpy(edge_indices).long(),
                                  torch.from_numpy(ef).float())
        graph.set_edge_labels(torch.from_numpy(edge_labels).long())

        if circuit_name not in list(circuit_dictionary.keys()):
            circuit_dictionary[circuit_name] = len(circuit_dictionary)+1

        graph.set_graph_attributes(circuit=circuit_dictionary[circuit_name])

        unseen_graphs[circuit_name] = graph

    # Data Augmentation
    for circuit_name, unseen_graph in tqdm(unseen_graphs.items()):
        circuit_i_val_graphs  = {"pos": [unseen_graph],
                                 "neg": data_augmentation(unseen_graph, "neg", 1)}
        circuit_i_test_graphs = {"pos": [unseen_graph],
                                 "neg": data_augmentation(unseen_graph, "neg", 1)}

        # Validation Graph Data Augmentation
        while len(circuit_i_val_graphs["pos"]) < args.datasize_per_circuit//2*args.val_ratio:
            pos_graph = random.choice(circuit_i_val_graphs["pos"])
            new_pos_graphs = data_augmentation(pos_graph, "pos", 1)
            new_neg_graphs = data_augmentation(pos_graph, "neg", 1)
            circuit_i_val_graphs["pos"].extend(new_pos_graphs)
            circuit_i_val_graphs["neg"].extend(new_neg_graphs)
        val_dataset[circuit_name] = circuit_i_val_graphs
        print(f"Validation Set Data Augmentation ({circuit_name}): size {args.datasize_per_circuit*args.val_ratio}")

        # Test Graph Data Augmentation
        while len(circuit_i_test_graphs["pos"]) < args.datasize_per_circuit//2*args.test_ratio:
            pos_graph = random.choice(circuit_i_test_graphs["pos"])
            new_pos_graphs = data_augmentation(pos_graph, "pos", 1)
            new_neg_graphs = data_augmentation(pos_graph, "neg", 1)
            circuit_i_test_graphs["pos"].extend(new_pos_graphs)
            circuit_i_test_graphs["neg"].extend(new_neg_graphs)
        test_dataset[circuit_name] = circuit_i_test_graphs
        print(f"Test Set Data Augmentation ({circuit_name}): size {args.datasize_per_circuit*args.test_ratio}")
    ############################################################################################################

    # Save the circuit dictionary
    with open("./circuits/circuits.json", 'w') as f:
        f.write(json.dumps(circuit_dictionary, indent=4))
    print()
    print(circuit_dictionary)
    print()

    # Save the dataset
    path = "./pretrain/dataset"
    os.makedirs(path, exist_ok=True)
    torch.save(train_dataset, f"{path}/{args.dataset_name}_train.pt")
    torch.save(val_dataset, f"{path}/{args.dataset_name}_val.pt")
    torch.save(test_dataset, f"{path}/{args.dataset_name}_test.pt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate circuit and simulation files from templates.")
    parser.add_argument("--dataset_name", default="pretraining_dataset_wo_device_params", type=str, help="Name of the dataset.")
    parser.add_argument("--datasize_per_circuit", default=5000, type=int, help="Number of data for each circuit (for augmentation).")
    parser.add_argument("--train_ratio", default=0.8, type=float, help="Percentage of data to be used for training.")
    parser.add_argument("--val_ratio", default=0.1, type=float, help="Percentage of data to be used for validation.")
    parser.add_argument("--test_ratio", default=0.1, type=float, help="Percentage of data to be used for testing.")
    args = parser.parse_args()
    save_dataset(args)