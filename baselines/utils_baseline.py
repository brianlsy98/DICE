import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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



###### Data Generation Utils #######
def parse_netlist_baseline(filename, baseline_name):

    node_names = []
    node_features = []
    node_labels = []
    node_dict = {}  # Maps node names to indices

    edge_names = []
    edge_indices = [[], []]
    edge_features = []
    edge_labels = []



    #####################################
    ############# ParaGraph #############
    #####################################
    if baseline_name == "ParaGraph":
        node_types = {'voltage_net'    : 0,
                      'current_source' : 1,
                      'nmos'           : 2,
                      'pmos'           : 3,
                      'resistor'       : 4,
                      'capacitor'      : 5,
                      'inductor'       : 6}
        edge_types = {'current_net'    : 0,
                      'net2trangate'   : 1,
                      'trangate2net'   : 2,
                      'net2trandrain'  : 3,
                      'trandrain2net'  : 4}

        def add_node(name, type_index):
            if name not in node_dict and name != 'gnd' and name != 'vdd':
                idx = len(node_names)
                node_dict[name] = idx
                node_names.append(name)
                feature = np.zeros(len(node_types))
                feature[type_index] = 1
                node_labels.append(type_index)
                node_features.append(feature)
                return node_dict[name]
            elif name in node_dict and name != 'gnd' and name != 'vdd':
                return node_dict[name]
            elif name == 'gnd' or name == 'vdd':
                return None

        def add_edge(src_idx, tgt_idx, type_index):
            if src_idx is not None and tgt_idx is not None:
                edge_indices[0].append(src_idx)
                edge_indices[1].append(tgt_idx)
                feature = np.zeros(len(edge_types))
                feature[type_index] = 1
                edge_labels.append(type_index)
                edge_features.append(feature)

        circuit_name = ''
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split()
                if not parts:
                    continue

                # Circuit Name
                if line.startswith('.subckt'):
                    circuit_name = parts[1]

                # MOSFETs
                elif line.startswith('MN') or line.startswith('MP'):
                    mosfet_name, drain, gate, source, body = parts[:5]
                    device_type = 'nmos' if line.startswith('MN') else 'pmos'

                    # Add nodes
                    mosfet_idx = add_node(mosfet_name, node_types[device_type])
                    drain_idx = add_node(drain, node_types['voltage_net'])
                    gate_idx = add_node(gate, node_types['voltage_net'])
                    source_idx = add_node(source, node_types['voltage_net'])
                    body_idx = add_node(body, node_types['voltage_net'])

                    # Add edges
                    add_edge(drain_idx, mosfet_idx, edge_types['net2trandrain'])
                    add_edge(mosfet_idx, drain_idx, edge_types['trandrain2net'])
                    add_edge(source_idx, mosfet_idx, edge_types['net2trandrain'])
                    add_edge(mosfet_idx, source_idx, edge_types['trandrain2net'])
                    add_edge(gate_idx, mosfet_idx, edge_types['net2trangate'])
                    add_edge(mosfet_idx, gate_idx, edge_types['trangate2net'])

                # Passive Components
                elif line.startswith(('I', 'R', 'C', 'L')):
                    comp_name, node1, node2 = parts[:3]
                    comp_type = {'I': 'current_source', 'R': 'resistor',
                                 'C': 'capacitor', 'L': 'inductor'}[line[0]]
                    # Add nodes
                    comp_idx = add_node(comp_name, node_types[comp_type])
                    node1_idx = add_node(node1, node_types['voltage_net'])
                    node2_idx = add_node(node2, node_types['voltage_net'])

                    # Add edges (bidirectional)
                    add_edge(node1_idx, comp_idx, edge_types['current_net'])
                    add_edge(node2_idx, comp_idx, edge_types['current_net'])
                    add_edge(comp_idx, node1_idx, edge_types['current_net'])
                    add_edge(comp_idx, node2_idx, edge_types['current_net'])


    #####################################
    ############## DeepGen ##############
    #####################################
    elif baseline_name == "DeepGen":
        node_types = {'gnd'            : 0,
                      'vin'            : 1,
                      'voltage_net'    : 2,
                      'current_source' : 3,
                      'nmos_gate'      : 4,
                      'nmos_drain'     : 5,
                      'nmos_source'    : 6,
                      'nmos_bulk'      : 7,
                      'pmos_gate'      : 8,
                      'pmos_drain'     : 9,
                      'pmos_source'    : 10,
                      'pmos_bulk'      : 11,
                      'resistor'       : 12,
                      'capacitor'      : 13,
                      'inductor'       : 14}
        
        def add_node(name, type_index):
            if name not in node_dict:
                idx = len(node_names)
                node_dict[name] = idx
                node_names.append(name)
                feature = np.zeros(len(node_types))
                feature[type_index] = 1
                node_labels.append(type_index)
                node_features.append(feature)
                return node_dict[name]
            elif name in node_dict:
                return node_dict[name]

        def add_edge(src_idx, tgt_idx):
            edge_indices[0].append(src_idx)
            edge_indices[1].append(tgt_idx)
            feature = [1]
            edge_labels.append(0)
            edge_features.append(feature)

        circuit_name = ''
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split()
                if not parts:
                    continue

                # Circuit Name
                if line.startswith('.subckt'):
                    circuit_name = parts[1]
                    input_names = parts[2:]

                # MOSFETs
                elif line.startswith('MN') or line.startswith('MP'):
                    mosfet_name, drain, gate, source, body = parts[:5]
                    device_type = 'nmos' if line.startswith('MN') else 'pmos'

                    # Add nodes
                    ## voltage nets
                    if drain == "gnd": net_d = add_node(drain, node_types['gnd'])
                    elif drain in input_names: net_d = add_node(drain, node_types['vin'])
                    else: net_d = add_node(drain, node_types['voltage_net'])
                    if gate == "gnd": net_g = add_node(gate, node_types['gnd'])
                    elif gate in input_names: net_g = add_node(gate, node_types['vin'])
                    else: net_g = add_node(gate, node_types['voltage_net'])
                    if source == "gnd": net_s = add_node(source, node_types['gnd'])
                    elif source in input_names: net_s = add_node(source, node_types['vin'])
                    else: net_s = add_node(source, node_types['voltage_net'])
                    if body == "gnd": net_b = add_node(body, node_types['gnd'])
                    elif body in input_names: net_b = add_node(body, node_types['vin'])
                    else: net_b = add_node(body, node_types['voltage_net'])
                    ## mosfet
                    if device_type == 'nmos':
                        drain_idx = add_node(f"{mosfet_name}_d", node_types['nmos_drain'])
                        gate_idx = add_node(f"{mosfet_name}_g", node_types['nmos_gate'])
                        source_idx = add_node(f"{mosfet_name}_s", node_types['nmos_source'])
                        body_idx = add_node(f"{mosfet_name}_b", node_types['nmos_bulk'])
                    elif device_type == 'pmos':
                        drain_idx = add_node(f"{mosfet_name}_d", node_types['pmos_drain'])
                        gate_idx = add_node(f"{mosfet_name}_g", node_types['pmos_gate'])
                        source_idx = add_node(f"{mosfet_name}_s", node_types['pmos_source'])
                        body_idx = add_node(f"{mosfet_name}_b", node_types['pmos_bulk'])

                    # Add edges
                    ## Within Mosfet
                    add_edge(drain_idx, gate_idx); add_edge(gate_idx, drain_idx)
                    add_edge(drain_idx, source_idx); add_edge(source_idx, drain_idx)
                    add_edge(drain_idx, body_idx); add_edge(body_idx, drain_idx)
                    add_edge(gate_idx, source_idx); add_edge(source_idx, gate_idx)
                    add_edge(gate_idx, body_idx); add_edge(body_idx, gate_idx)
                    add_edge(source_idx, body_idx); add_edge(body_idx, source_idx)
                    ## Between Mosfet and Voltage Nets
                    add_edge(drain_idx, net_d); add_edge(net_d, drain_idx)
                    add_edge(gate_idx, net_g); add_edge(net_g, gate_idx)
                    add_edge(source_idx, net_s); add_edge(net_s, source_idx)
                    add_edge(body_idx, net_b); add_edge(net_b, body_idx)


                # Passive Components
                elif line.startswith(('I', 'R', 'C', 'L')):
                    comp_name, node1, node2 = parts[:3]
                    comp_type = {'I': 'current_source', 'R': 'resistor',
                                 'C': 'capacitor', 'L': 'inductor'}[line[0]]
                    # Add nodes
                    ## voltage nets
                    if node1 == "gnd": net1 = add_node(node1, node_types['gnd'])
                    elif node1 in input_names: net1 = add_node(node1, node_types['vin'])
                    else: net1 = add_node(node1, node_types['voltage_net'])
                    if node2 == "gnd": net2 = add_node(node2, node_types['gnd'])
                    elif node2 in input_names: net2 = add_node(node2, node_types['vin'])
                    else: net2 = add_node(node2, node_types['voltage_net'])
                    ## passive component
                    comp_idx_1 = add_node(f"{comp_name}_1", node_types[comp_type])
                    comp_idx_2 = add_node(f"{comp_name}_2", node_types[comp_type])
                    
                    # Add edges
                    ## Within Passive Component
                    add_edge(comp_idx_1, comp_idx_2); add_edge(comp_idx_2, comp_idx_1)
                    ## Between Passive Component and Voltage Nets
                    add_edge(comp_idx_1, net1); add_edge(net1, comp_idx_1)
                    add_edge(comp_idx_2, net2); add_edge(net2, comp_idx_2)

                elif line.startswith('V'):
                    voltage_name, node1, node2 = parts[:3]
                    # Add nodes
                    ## voltage nets
                    if node1 == "gnd": net1 = add_node(node1, node_types['gnd'])
                    elif node1 in input_names: net1 = add_node(node1, node_types['vin'])
                    else: net1 = add_node(node1, node_types['voltage_net'])
                    if node2 == "gnd": net2 = add_node(node2, node_types['gnd'])
                    elif node2 in input_names: net2 = add_node(node2, node_types['vin'])
                    else: net2 = add_node(node2, node_types['voltage_net'])
                    ## voltage source
                    voltage_idx = add_node(voltage_name, node_types['voltage_net'])

                    # Add edges
                    add_edge(voltage_idx, net1); add_edge(net1, voltage_idx)
                    add_edge(voltage_idx, net2); add_edge(net2, voltage_idx)



    # Convert to NumPy arrays
    node_features = np.array(node_features)
    node_labels = np.array(node_labels)
    edge_indices = np.array(edge_indices)
    edge_features = np.array(edge_features)
    edge_labels = np.array(edge_labels)

    return circuit_name, node_names, edge_names, node_features, node_labels,\
            edge_indices, edge_features, edge_labels






def data_augmentation_baseline(graph, pos_or_neg="pos", sample_size=2, baseline_name="ParaGraph"):
    new_graph = copy.deepcopy(graph)
    new_graphs = []


    #####################################
    ############# ParaGraph #############
    #####################################
    if baseline_name == "ParaGraph":

        def add_parallel_device(graph, device, device_index):
            new_graph = copy.deepcopy(graph)

            ### add new node
            new_graph.x = torch.cat([new_graph.x, F.one_hot(torch.tensor([device], dtype=torch.long), 7).float()], dim=0)
            new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([device], dtype=torch.long)], dim=0)
            new_device_index = len(new_graph.x) - 1


            ### add new edges
            if device != 2 and device != 3:
                indices = torch.where(  ((new_graph.edge_index[1] == device_index) == True)\
                                                & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0], dtype=torch.long)).all(dim=1))[0]
                if len(indices) == 2:
                    net_index_1, _ = new_graph.edge_index[:, indices[0]]
                    net_index_2, _ = new_graph.edge_index[:, indices[1]]
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                    torch.tensor([[new_device_index, net_index_1],
                                                                    [net_index_1, new_device_index]], dtype=torch.long),
                                                    torch.tensor([[new_device_index, net_index_2],
                                                                    [net_index_2, new_device_index]], dtype=torch.long)], dim=1)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                    torch.tensor([[1, 0, 0, 0, 0],
                                                                  [1, 0, 0, 0, 0],
                                                                  [1, 0, 0, 0, 0],
                                                                  [1, 0, 0, 0, 0]], dtype=torch.float32)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0, 0, 0, 0], dtype=torch.long)], dim=0)
                
                elif len(indices) == 1:
                    net_index, _ = new_graph.edge_index[:, indices[0]]
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                    torch.tensor([[new_device_index, net_index],
                                                                    [net_index, new_device_index]], dtype=torch.long)], dim=1)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                    torch.tensor([[1, 0, 0, 0, 0],
                                                                  [1, 0, 0, 0, 0]], dtype=torch.float32)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0, 0], dtype=torch.long)], dim=0)


            # mosfet related edges
            elif device == 2 or device == 3:
                # net2transistor
                mos_net2tran_mask = (new_graph.edge_index[1] == device_index)
                net2trangate_index = torch.where(  (mos_net2tran_mask == True)\
                                        & (new_graph.edge_attr == torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)).all(dim=1))[0]
                net2trandrain_index = torch.where(  (mos_net2tran_mask == True)\
                                        & (new_graph.edge_attr == torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)).all(dim=1))[0]
                
                if len(net2trangate_index) == 1 and len(net2trandrain_index) == 1:
                    gate_node_index = new_graph.edge_index[0, net2trangate_index]
                    drain_node_index = new_graph.edge_index[0, net2trandrain_index]
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                    torch.tensor([[gate_node_index, drain_node_index],
                                                                [new_device_index, new_device_index]], dtype=torch.long)], dim=1)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                    torch.tensor([[0, 1, 0, 0, 0],
                                                                  [0, 0, 0, 1, 0]], dtype=torch.float32)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([1, 3], dtype=torch.long)], dim=0)
                    
                elif len(net2trangate_index) == 0 or len(net2trandrain_index) == 2:
                    if len(net2trangate_index) != 0:
                        gate_node_index = new_graph.edge_index[0, net2trangate_index]
                        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                        torch.tensor([[gate_node_index],
                                                                      [new_device_index]], dtype=torch.long)], dim=1)
                        new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                        torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)], dim=0)
                        new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([1], dtype=torch.long)], dim=0)
                    if len(net2trandrain_index) == 2:
                        drain_node_index_1, drain_node_index_2 = new_graph.edge_index[0, net2trandrain_index]
                        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                        torch.tensor([[drain_node_index_1, drain_node_index_2],
                                                                      [new_device_index, new_device_index]], dtype=torch.long)], dim=1)
                        new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                        torch.tensor([[0, 0, 0, 1, 0],
                                                                      [0, 0, 0, 1, 0]], dtype=torch.float32)], dim=0)
                        new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([3, 3], dtype=torch.long)], dim=0)

                
                # transistor2net
                mos_tran2net_mask = (new_graph.edge_index[0] == device_index)
                trangate2net_index = torch.where(  (mos_tran2net_mask == True)\
                                        & (new_graph.edge_attr == torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)).all(dim=1))[0]
                trandrain2net_index = torch.where(  (mos_tran2net_mask == True)\
                                        & (new_graph.edge_attr == torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)).all(dim=1))[0]

                if len(trangate2net_index) == 1 and len(trandrain2net_index) == 1:
                    gate_node_index = new_graph.edge_index[1, trangate2net_index]
                    drain_node_index = new_graph.edge_index[1, trandrain2net_index]
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                    torch.tensor([[new_device_index, new_device_index],
                                                                [gate_node_index, drain_node_index]], dtype=torch.long)], dim=1)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                    torch.tensor([[0, 0, 1, 0, 0],
                                                                  [0, 0, 0, 0, 1]], dtype=torch.float32)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([2, 4], dtype=torch.long)], dim=0)

                elif len(trangate2net_index) == 0 or len(trandrain2net_index) == 2:
                    if len(trangate2net_index) != 0:
                        gate_node_index = new_graph.edge_index[1, trangate2net_index]
                        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                        torch.tensor([[new_device_index],
                                                                      [gate_node_index]], dtype=torch.long)], dim=1)
                        new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                        torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32)], dim=0)
                        new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([2], dtype=torch.long)], dim=0)
                    if len(trandrain2net_index) == 2:
                        drain_node_index_1, drain_node_index_2 = new_graph.edge_index[1, trandrain2net_index]
                        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                        torch.tensor([[new_device_index, new_device_index],
                                                                      [drain_node_index_1, drain_node_index_2]], dtype=torch.long)], dim=1)
                        new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                        torch.tensor([[0, 0, 0, 0, 1],
                                                                      [0, 0, 0, 0, 1]], dtype=torch.float32)], dim=0)
                        new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([4, 4], dtype=torch.long)], dim=0)



            return new_graph

        
        # current source, nmos, pmos, resistor, capacitor, inductor
        valid_indices = torch.where(  (new_graph.node_y >= 1)\
                                    & (new_graph.node_y <= 6) )[0]
        device_indices = random.sample(valid_indices.tolist(), sample_size)
        devices = new_graph.node_y[device_indices]

        if pos_or_neg == "pos": # only do pos data augmentation
            # add the exact same device in parallel
            for device_index, device in zip(device_indices, devices):
                temp_graph = add_parallel_device(new_graph, device, device_index)
                new_graphs.append(temp_graph)



    #####################################
    ############## DeepGen ##############
    #####################################
    elif baseline_name == "DeepGen":

        def add_parallel_edge(graph, node, node_index):
            new_graph = copy.deepcopy(graph)

            ### add new node
            new_graph.x = torch.cat([new_graph.x,
                                     F.one_hot(torch.tensor([node], dtype=torch.long), 15).float()], dim=0)
            new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([node], dtype=torch.long)], dim=0)
            new_node_index = len(new_graph.x) - 1

            ### add new edge
            indices = torch.where((new_graph.edge_index[1] == node_index) == True)[0]
            for index in indices:
                if new_graph.edge_index[0][index] == node_index:
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                      torch.tensor([[new_node_index],
                                                                    [new_graph.edge_index[1][index]]], dtype=torch.long)], dim=1)
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                      torch.tensor([[new_graph.edge_index[1][index]],
                                                                    [new_node_index]], dtype=torch.long)], dim=1)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                     torch.tensor([1], dtype=torch.long).unsqueeze(-1)], dim=0)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                     torch.tensor([1], dtype=torch.long).unsqueeze(-1)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0], dtype=torch.long)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0], dtype=torch.long)], dim=0)
                elif new_graph.edge_index[1][index] == node_index:
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                      torch.tensor([[new_graph.edge_index[0][index]],
                                                                    [new_node_index]], dtype=torch.long)], dim=1)
                    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                                      torch.tensor([[new_node_index],
                                                                    [new_graph.edge_index[0][index]]], dtype=torch.long)], dim=1)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                     torch.tensor([1], dtype=torch.long).unsqueeze(-1)], dim=0)
                    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                                     torch.tensor([1], dtype=torch.long).unsqueeze(-1)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0], dtype=torch.long)], dim=0)
                    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0], dtype=torch.long)], dim=0)

            return new_graph

        # gnd, vin, voltage_net, current_source, nmos_gate, nmos_drain, nmos_source, nmos_bulk, pmos_gate, pmos_drain, pmos_source, pmos_bulk, resistor, capacitor, inductor
        valid_indices = torch.where(  (new_graph.node_y >= 3)\
                                    & (new_graph.node_y <= 14) )[0]
        node_indices = random.sample(valid_indices.tolist(), sample_size)
        nodes = new_graph.node_y[node_indices]

        if pos_or_neg == "pos": # only do pos data augmentation
            # add the exact edge in parallel
            for node_index, node in zip(node_indices, nodes):
                temp_graph = add_parallel_edge(new_graph, node, node_index)
                new_graphs.append(temp_graph)


    return new_graphs
####################################