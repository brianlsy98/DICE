from math import e
import os
import sys
import re
import argparse


import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/torch_datasets'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)


from utils import *
from dataloader import *







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate circuit and simulation files from templates.")
    parser.add_argument("--netlist_dir", default="./data/netlists",
                        type=str, help="Name of the netlist template directory")
    parser.add_argument("--sim_result_dir", default="./data/sims/txt",
                        type=str, help="Name of the simulation template directory")
    parser.add_argument("--train_ratio", default=0.8,
                        type=float, help="Percentage of data to be used for training")
    parser.add_argument("--val_ratio", default=0.1,
                        type=float, help="Percentage of data to be used for validation")
    parser.add_argument("--test_ratio", default=0.1,
                        type=float, help="Percentage of data to be used for testing")
    args = parser.parse_args()
    
    ########################################
    # Preparing the dataset
    ########################################
    sim_netlist_types = os.listdir(args.netlist_dir)
    edge_types = ['nmos', 'pmos', 'R', 'L', 'C']           # order is important
    ########################################


    # Collect Graphs
    graphs = []
    for sim_netlist_type in sim_netlist_types:
        
        sim_netlists = os.listdir(f"{args.netlist_dir}/{sim_netlist_type}")

        for sim_netlist in sim_netlists:

            sim_netlist_file = f"{args.netlist_dir}/{sim_netlist_type}/{sim_netlist}"
            sim_result_filename = sim_netlist[8:-4]+".txt"
            sim_result_file = f"{args.sim_result_dir}/{sim_netlist_type}/{sim_result_filename}"

            ## Parse sim_netlist_file and get
            ### Graph Structure, Edge Feature Values
            node_names, input_node_names, output_node_names, edge_features, graph_attributes =\
                parse_sim_netlist(sim_netlist_file)
            # node_mask = torch.ones(len(node_names), dtype=torch.bool)
            # out_node_indices = [node_names.index(n) for n in output_node_names]
            # node_mask[out_node_indices] = False

            ## Parse sim_result_file and get
            ### Node Feature Values
            node_features = parse_sim_result(sim_result_file)

            node_names = {'input': input_node_names, 'output': output_node_names}

            # print()
            # print(node_names)
            # print(input_node_names)
            # print(output_node_names)
            # print(node_features)
            # print(edge_features)
            # print(graph_attributes)
            # print(node_mask)
            # print()


            ## Make Graphs
            # i : index for different node voltage combinations
            for i in range(len(list(node_features.values())[0])):

                # node features
                in_nf_i, out_nf_i = [], []
                for in_node_name in input_node_names:
                    in_nf_i.append(node_features[in_node_name][i])
                for out_node_name in output_node_names:
                    out_nf_i.append(node_features[out_node_name][i])
                nf_i = {'input': torch.tensor(in_nf_i, dtype=torch.float),
                        'output': torch.tensor(out_nf_i, dtype=torch.float)}

                # edge features
                ef_i = {}
                for k, v in edge_features.items():
                    src_node_name, edge_type, dst_node_name = k
                    src_node_type = 'input' if src_node_name in input_node_names else 'output'
                    dst_node_type = 'input' if dst_node_name in input_node_names else 'output'

                    key = (src_node_type, edge_type, dst_node_type)
                    key_inv = (dst_node_type, edge_type, src_node_type)
                    if key not in ef_i.keys():
                        ef_i[key] = {'edge_index': [], 'edge_attr': []}
                    if key_inv not in ef_i.keys():
                        ef_i[key_inv] = {'edge_index': [], 'edge_attr': []}

                    # edge index (bi-directional)
                    ef_i[key]['edge_index'].append([node_names[src_node_type].index(src_node_name),
                                                    node_names[dst_node_type].index(dst_node_name)])
                    ef_i[key_inv]['edge_index'].append([node_names[dst_node_type].index(dst_node_name),
                                                        node_names[src_node_type].index(src_node_name)])

                    # edge attribute
                    # j : index for each edge feature list
                    edge_attribute = []
                    for feat_v in v:
                        if type(feat_v) == str:
                            n_type = 'input' if feat_v in input_node_names else 'output'
                            n_type_index = list(node_names.keys()).index(n_type)
                            n_index = node_names[n_type].index(feat_v)
                            edge_attribute.append(n_type_index)
                            edge_attribute.append(n_index)
                        else:
                            edge_attribute.append(feat_v)
                    ef_i[key]['edge_attr'].append(edge_attribute)
                    ef_i[key_inv]['edge_attr'].append(edge_attribute)


                # print(out_n_index, nf_i, ef_i, graph_attributes)
                # make graph given
                ## node names, nf_i, ef_i
                # print()
                # print(nf_i)
                # print(ef_i)
                # print()


                graph = HeteroGraphData()
                # add node
                for node_type, dc_v in nf_i.items():
                    graph.add_node(node_type, dc_voltages=dc_v)
                # add edge
                for (src_type, edge_type, dst_type), edge_attr in ef_i.items():                    
                    graph.add_edge((src_type, edge_type, dst_type),
                                    edge_index=torch.tensor(edge_attr['edge_index'], dtype=torch.float).T,
                                    edge_attr=torch.tensor(edge_attr['edge_attr'], dtype=torch.float))
                # add graph attribute
                if graph_attributes['TECH'] == './templates/mosfet_model/45nm_bulk.txt':
                    tech = 0
                else: tech = 1
                if graph_attributes['TEMP'] == '27':
                    temp = 0
                else: temp = 1
                graph.set_graph_attributes(tech=tech, temp=temp)

                # print()
                # print(graph.node_dict)
                # print(graph.edge_dict)
                # print(graph.graph_attrs)
                # print()

                graphs.append(graph)
                #### Each graph has one set of node voltages

    
    # Save the dataset
    torch_data = {}
    dataset_name = "dc"
    path = "./data/torch_datasets"

    torch_data['name'] = dataset_name
    torch_data['all_data'] = graphs

    import random
    random.shuffle(graphs)
    train_size, val_size, test_size = int(len(graphs)*args.train_ratio),\
                                      int(len(graphs)*args.val_ratio),\
                                      int(len(graphs)*args.test_ratio)
    torch_data['train_data'] = graphs[:train_size]
    torch_data['val_data'] = graphs[train_size:train_size+val_size]
    torch_data['test_data'] = graphs[train_size+val_size:]

    os.makedirs(path, exist_ok=True)
    torch.save(torch_data, f"{path}/{dataset_name}.pt")
