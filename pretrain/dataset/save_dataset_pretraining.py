import os
import sys
import argparse
import copy
import random
import json

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import parse_netlist
from dataloader import GraphData

'''
node_types = {'gnd'            : 0,
              'vdd'            : 1,
              'voltage_net'    : 2,
              'current_source' : 3,
              'nmos'           : 4,
              'pmos'           : 5,
              'resistor'       : 6,
              'capacitor'      : 7,
              'inductor'       : 8}
edge_types = {'current_net'    : 0,
              'v2ng'           : 1,
              'v2pg'           : 2,
              'v2nb'           : 3,
              'v2pb'           : 4}
'''

def add_parallel_device(graph, device, device_index):
    new_graph = copy.deepcopy(graph)

    ### add new node
    new_graph.x = torch.cat([new_graph.x, new_graph.x[device_index].unsqueeze(0)], dim=0)
    new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([device])], dim=0)
    new_device_index = len(new_graph.x) - 1

    ### add new edges
    # current flowing edges
    index_1, index_2 = torch.where(  ((new_graph.edge_index[1] == device_index) == True)\
                                    & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0])).all(dim=1))[0]
    net_index_1, _ = new_graph.edge_index[:, index_1]
    net_index_2, _ = new_graph.edge_index[:, index_2]
    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                      torch.tensor([[new_device_index, net_index_1],
                                                    [net_index_1, new_device_index]]),
                                      torch.tensor([[new_device_index, net_index_2],
                                                    [net_index_2, new_device_index]])], dim=1)
    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                     torch.tensor([[1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0]])], dim=0)
    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0, 0, 0, 0])], dim=0)

    # mosfet related edges
    if device == 4 or device == 5:
        mos_mask = (new_graph.edge_index[1] == device_index)
        if device == 4:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 1, 0, 0, 0])).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 1, 0])).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 1, 0, 0, 0],
                                                           [0, 0, 0, 1, 0]])], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([1, 3])], dim=0)
        elif device == 5:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 1, 0, 0])).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 0, 1])).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 1]])], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([2, 4])], dim=0)
        gate_node_index = new_graph.edge_index[0, gate_index]
        bulk_node_index = new_graph.edge_index[0, bulk_index]
        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                          torch.tensor([[gate_node_index, bulk_node_index],
                                                        [new_device_index, new_device_index]])], dim=1)
    
    return new_graph


def add_series_device(graph, device, device_index):
    new_graph = copy.deepcopy(graph)

    ### add new nodes
    new_graph.x = torch.cat([new_graph.x, new_graph.x[device_index].unsqueeze(0)], dim=0)
    new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([device])], dim=0)
    new_graph.x = torch.cat([new_graph.x, torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0]])], dim=0)
    new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([2])], dim=0)
    new_device_index, new_net_index = len(new_graph.x) - 2, len(new_graph.x) - 1

    ### delete previous edges
    valid_indices = torch.where(  ((new_graph.edge_index[1] == device_index) == True)\
                                & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0])).all(dim=1))[0]
    deleting_edge_index1 = random.choice(valid_indices)
    old_net_index, _ = new_graph.edge_index[:, deleting_edge_index1]
    deleting_edge_index2 = torch.where(  (new_graph.edge_index[0] == device_index)\
                                       & (new_graph.edge_index[1] == old_net_index)\
                                       & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0])).all(dim=1))[0]
    edge_deleting_mask = torch.ones(len(new_graph.edge_attr), dtype=torch.bool)
    edge_deleting_mask[deleting_edge_index1] = False
    edge_deleting_mask[deleting_edge_index2] = False
    new_graph.edge_index = new_graph.edge_index[:, edge_deleting_mask]
    new_graph.edge_attr = new_graph.edge_attr[edge_deleting_mask, :]
    new_graph.edge_y = new_graph.edge_y[edge_deleting_mask]


    ### add new edges
    # current flowing edges
    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                      torch.tensor([[new_net_index, device_index],
                                                    [device_index, new_net_index]]),
                                      torch.tensor([[new_net_index, new_device_index],
                                                    [new_device_index, new_net_index]]),
                                      torch.tensor([[new_device_index, old_net_index],
                                                    [old_net_index, new_device_index]])], dim=1)
    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                     torch.tensor([[1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0]])], dim=0)
    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0, 0, 0, 0, 0, 0])], dim=0)



    # mosfet related edges
    if device == 4 or device == 5:
        mos_mask = (new_graph.edge_index[1] == device_index)
        if device == 4:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 1, 0, 0, 0])).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 1, 0])).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 1, 0, 0, 0],
                                                           [0, 0, 0, 1, 0]])], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([1, 3])], dim=0)
        elif device == 5:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 1, 0, 0])).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 0, 1])).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 1]])], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([2, 4])], dim=0)
        gate_node_index = new_graph.edge_index[0, gate_index]
        bulk_node_index = new_graph.edge_index[0, bulk_index]
        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                          torch.tensor([[gate_node_index, bulk_node_index],
                                                        [new_device_index, new_device_index]])], dim=1)
    
    if new_graph.x.size(0) < new_graph.edge_index[0].max():
        print("Error: edge_index[0].max() is larger than the number of nodes.")
    if new_graph.x.size(0) < new_graph.edge_index[1].max():
        print("Error: edge_index[1].max() is larger than the number of nodes.")

    return new_graph



def data_augmentation(graph):

    new_graph = copy.deepcopy(graph)

    # current_source, nmos, pmos, resistor, capacitor, inductor
    valid_indices = torch.where(  (new_graph.node_y >= 3)\
                                & (new_graph.node_y <= 8) )[0]
    device_index = random.choice(valid_indices)
    device = new_graph.node_y[device_index]

    # add device in parallel / series
    parallel_or_series = random.choice(["parallel", "series"])
    if parallel_or_series == "parallel":
        new_graph = add_parallel_device(new_graph, device, device_index)
    elif parallel_or_series == "series":
        new_graph = add_series_device(new_graph, device, device_index)

    return new_graph



def save_dataset(args):

    os.remove("./circuits.json")
    circuit_dictionary = {}

    graphs = []

    # Dataset before Augmentation
    for circuit in os.listdir(args.netlist_dir):

        netlist = f"{args.netlist_dir}/{circuit}/netlist.cir"
        circuit_name, node_names, nf, node_labels, edge_indices, ef, edge_labels = parse_netlist(netlist)

        graph = GraphData()
        graph.set_node_attributes(torch.from_numpy(nf).float())
        graph.set_node_labels(torch.from_numpy(node_labels).long())
        graph.set_edge_attributes(torch.from_numpy(edge_indices).long(),
                                  torch.from_numpy(ef).float())
        graph.set_edge_labels(torch.from_numpy(edge_labels).long())

        if circuit_name not in list(circuit_dictionary.keys()):
            circuit_dictionary[circuit_name] = len(circuit_dictionary)
        graph.set_graph_attributes(circuit=circuit_dictionary[circuit_name])

        graphs.append(graph)

    with open("./circuits.json", 'w') as f:
        f.write(json.dumps(circuit_dictionary, indent=4))
    print()
    print(circuit_dictionary)
    print()

    random.shuffle(graphs)
    train_graphs = copy.deepcopy(graphs)
    random.shuffle(graphs)
    val_graphs = copy.deepcopy(graphs)
    random.shuffle(graphs)
    test_graphs = copy.deepcopy(graphs)

    # Train Graph Data Augmentation
    while len(train_graphs) < args.dataset_size*args.train_ratio:
        graph = random.choice(train_graphs)
        new_graph = data_augmentation(graph)
        train_graphs.append(new_graph)
        
        if len(train_graphs) % 1000 == 0:
            print(f"Training Set Data Augmentation: {len(train_graphs)}/{int(args.dataset_size*args.train_ratio)}")

    # Validation Graph Data Augmentation
    while len(val_graphs) < args.dataset_size*args.val_ratio:
        graph = random.choice(val_graphs)
        new_graph = data_augmentation(graph)
        val_graphs.append(new_graph)
        
        if len(val_graphs) % 1000 == 0:
            print(f"Validation Set Data Augmentation: {len(val_graphs)}/{int(args.dataset_size*args.val_ratio)}")

    # Test Graph Data Augmentation
    while len(test_graphs) < args.dataset_size*args.test_ratio:
        graph = random.choice(test_graphs)
        new_graph = data_augmentation(graph)
        test_graphs.append(new_graph)
        
        if len(test_graphs) % 1000 == 0:
            print(f"Test Set Data Augmentation: {len(test_graphs)}/{int(args.dataset_size*args.test_ratio)}")

    # Save the dataset
    torch_data = {}
    path = "./pretrain/dataset"

    torch_data['name'] = args.dataset_name
    torch_data['train_data'] = train_graphs
    torch_data['val_data'] = val_graphs
    torch_data['test_data'] = test_graphs

    os.makedirs(path, exist_ok=True)
    torch.save(torch_data, f"{path}/{args.dataset_name}.pt")







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate circuit and simulation files from templates.")
    parser.add_argument("--dataset_name", default="pretraining_dataset_wo_device_params", type=str, help="Name of the dataset.")
    parser.add_argument("--dataset_size", default=100000, type=int, help="Number of circuits in the dataset (for augmentation).")
    parser.add_argument("--netlist_dir", default="./pretrain/dataset/netlist_templates", type=str, help="Name of the netlist template directory.")
    parser.add_argument("--train_ratio", default=0.7, type=float, help="Percentage of data to be used for training.")
    parser.add_argument("--val_ratio", default=0.05, type=float, help="Percentage of data to be used for validation.")
    parser.add_argument("--test_ratio", default=0.25, type=float, help="Percentage of data to be used for testing.")
    args = parser.parse_args()
    save_dataset(args)