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
def parse_netlist(filename):

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

    node_names = []
    node_features = []
    node_labels = []
    node_dict = {}  # Maps node names to indices

    edge_indices = [[], []]
    edge_features = []
    edge_labels = []

    def add_node(name, type_index):
        if name not in node_dict:
            idx = len(node_names)
            node_dict[name] = idx
            node_names.append(name)
            if name == 'gnd':
                feature = np.zeros(len(node_types))
                feature[0] = 1
                node_labels.append(0)
            elif name == 'vdd':
                feature = np.zeros(len(node_types))
                feature[1] = 1
                node_labels.append(1)
            else:
                feature = np.zeros(len(node_types))
                feature[type_index] = 1
                node_labels.append(type_index)
            node_features.append(feature)
        return node_dict[name]

    def add_edge(src_idx, tgt_idx, type_index):
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
                add_edge(drain_idx, mosfet_idx, edge_types['current_net'])
                add_edge(mosfet_idx, drain_idx, edge_types['current_net'])
                add_edge(source_idx, mosfet_idx, edge_types['current_net'])
                add_edge(mosfet_idx, source_idx, edge_types['current_net'])

                if device_type == 'nmos':
                    add_edge(gate_idx, mosfet_idx, edge_types['v2ng'])
                    add_edge(body_idx, mosfet_idx, edge_types['v2nb'])
                else:
                    add_edge(gate_idx, mosfet_idx, edge_types['v2pg'])
                    add_edge(body_idx, mosfet_idx, edge_types['v2pb'])

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

    # Convert to NumPy arrays
    node_features = np.array(node_features)
    node_labels = np.array(node_labels)
    edge_indices = np.array(edge_indices)
    edge_features = np.array(edge_features)
    edge_labels = np.array(edge_labels)

    return circuit_name, node_names, node_features, node_labels,\
            edge_indices, edge_features, edge_labels



def add_parallel_device(graph, device, target_device, target_device_index):
    new_graph = copy.deepcopy(graph)

    ### add new node
    new_graph.x = torch.cat([new_graph.x, F.one_hot(torch.tensor([device], dtype=torch.long), 9).float()], dim=0)
    new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([device], dtype=torch.long)], dim=0)
    new_device_index = len(new_graph.x) - 1

    ### add new edges
    # current flowing edges
    index_1, index_2 = torch.where(  ((new_graph.edge_index[1] == target_device_index) == True)\
                                    & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0], dtype=torch.long)).all(dim=1))[0]
    net_index_1, _ = new_graph.edge_index[:, index_1]
    net_index_2, _ = new_graph.edge_index[:, index_2]
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

    # mosfet related edges
    if target_device == 4 or target_device == 5:
        mos_mask = (new_graph.edge_index[1] == target_device_index)
        if target_device == 4:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 1, 0, 0, 0],
                                                           [0, 0, 0, 1, 0]], dtype=torch.float32)], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([1, 3], dtype=torch.long)], dim=0)
        elif target_device == 5:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 1]], dtype=torch.float32)], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([2, 4], dtype=torch.long)], dim=0)
        gate_node_index = new_graph.edge_index[0, gate_index]
        bulk_node_index = new_graph.edge_index[0, bulk_index]
        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                          torch.tensor([[gate_node_index, bulk_node_index],
                                                        [new_device_index, new_device_index]], dtype=torch.long)], dim=1)
    
    return new_graph



def add_series_device(graph, device, target_device, target_device_index):
    new_graph = copy.deepcopy(graph)

    ### add new nodes
    new_graph.x = torch.cat([new_graph.x, F.one_hot(torch.tensor([device], dtype=torch.long), 9).float()], dim=0)
    new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([device], dtype=torch.long)], dim=0)
    new_graph.x = torch.cat([new_graph.x, torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)], dim=0)
    new_graph.node_y = torch.cat([new_graph.node_y, torch.tensor([2], dtype=torch.long)], dim=0)
    new_device_index, new_net_index = len(new_graph.x) - 2, len(new_graph.x) - 1

    ### delete previous edges
    valid_indices = torch.where(  ((new_graph.edge_index[1] == target_device_index) == True)\
                                & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)).all(dim=1))[0]
    deleting_edge_index1 = random.choice(valid_indices)
    old_net_index, _ = new_graph.edge_index[:, deleting_edge_index1]
    deleting_edge_index2 = torch.where(  (new_graph.edge_index[0] == target_device_index)\
                                       & (new_graph.edge_index[1] == old_net_index)\
                                       & (new_graph.edge_attr == torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)).all(dim=1))[0]
    edge_deleting_mask = torch.ones(len(new_graph.edge_attr), dtype=torch.bool)
    edge_deleting_mask[deleting_edge_index1] = False
    edge_deleting_mask[deleting_edge_index2] = False
    new_graph.edge_index = new_graph.edge_index[:, edge_deleting_mask]
    new_graph.edge_attr = new_graph.edge_attr[edge_deleting_mask, :]
    new_graph.edge_y = new_graph.edge_y[edge_deleting_mask]


    ### add new edges
    # current flowing edges
    new_graph.edge_index = torch.cat([new_graph.edge_index,
                                      torch.tensor([[new_net_index, target_device_index],
                                                    [target_device_index, new_net_index]], dtype=torch.long),
                                      torch.tensor([[new_net_index, new_device_index],
                                                    [new_device_index, new_net_index]], dtype=torch.long),
                                      torch.tensor([[new_device_index, old_net_index],
                                                    [old_net_index, new_device_index]], dtype=torch.long)], dim=1)
    new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                     torch.tensor([[1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0],
                                                   [1, 0, 0, 0, 0]], dtype=torch.float32)], dim=0)
    new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long)], dim=0)



    # mosfet related edges
    if target_device == 4 or target_device == 5:
        mos_mask = (new_graph.edge_index[1] == target_device_index)
        if target_device == 4:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32)).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 1, 0, 0, 0],
                                                           [0, 0, 0, 1, 0]], dtype=torch.float32)], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([1, 3], dtype=torch.long)], dim=0)
        elif target_device == 5:
            gate_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32)).all(dim=1))[0]
            bulk_index = torch.where(  (mos_mask == True)\
                                     & (new_graph.edge_attr == torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)).all(dim=1))[0]
            new_graph.edge_attr = torch.cat([new_graph.edge_attr,
                                             torch.tensor([[0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 1]], dtype=torch.float32)], dim=0)
            new_graph.edge_y = torch.cat([new_graph.edge_y, torch.tensor([2, 4], dtype=torch.long)], dim=0)
        gate_node_index = new_graph.edge_index[0, gate_index]
        bulk_node_index = new_graph.edge_index[0, bulk_index]
        new_graph.edge_index = torch.cat([new_graph.edge_index,
                                          torch.tensor([[gate_node_index, bulk_node_index],
                                                        [new_device_index, new_device_index]], dtype=torch.long)], dim=1)
    
    if new_graph.x.size(0) < new_graph.edge_index[0].max():
        print("Error: edge_index[0].max() is larger than the number of nodes.")
    if new_graph.x.size(0) < new_graph.edge_index[1].max():
        print("Error: edge_index[1].max() is larger than the number of nodes.")

    return new_graph



def change_device(graph, device, target_device_index):
    new_graph = copy.deepcopy(graph)

    ### change the device
    new_graph.x[target_device_index] = F.one_hot(torch.tensor(device, dtype=torch.long), 9).float()
    new_graph.node_y[target_device_index] = device

    # we do not input mosfets in this function

    return new_graph



def data_augmentation(graph, pos_or_neg="pos", sample_size=2):
    new_graph = copy.deepcopy(graph)
    new_graphs = []

    # current_source, nmos, pmos, resistor, capacitor, inductor
    valid_indices = torch.where(  (new_graph.node_y >= 3)\
                                & (new_graph.node_y <= 8) )[0]
    device_indices = random.sample(valid_indices.tolist(), sample_size)
    devices = new_graph.node_y[device_indices]

    if pos_or_neg == "pos":
        # add the exact same device in parallel or series
        for device_index, device in zip(device_indices, devices):
            parallel_or_series = random.choice(["parallel", "series"])
            if parallel_or_series == "parallel":
                temp_graph = add_parallel_device(new_graph, device, device, device_index)
            elif parallel_or_series == "series":
                temp_graph = add_series_device(new_graph, device, device, device_index)
            new_graphs.append(temp_graph)

    elif pos_or_neg == "neg":
        # current_source : change to resistor & add capacitor parallel & add inductor parallel
        # nmos : add pmos series & parallel
        # pmos : add nmos series & parallel
        # resistor : change to capcitor & add inductor series
        # capacitor : add inductor series & parallel
        # inductor : add capacitor series & parallel
        for device_index, device in zip(device_indices, devices):
            if device == 3:     # current_source
                temp_graph = change_device(new_graph, 6, device_index)
                temp_graph = add_parallel_device(temp_graph, 7, device, device_index)
                temp_graph = add_parallel_device(temp_graph, 8, device, device_index)
            elif device == 4:   # nmos
                temp_graph = add_parallel_device(new_graph, 5, device, device_index)
                temp_graph = add_series_device(temp_graph, 5, device, device_index)
            elif device == 5:   # pmos
                temp_graph = add_parallel_device(new_graph, 4, device, device_index)
                temp_graph = add_series_device(temp_graph, 4, device, device_index)
            elif device == 6:   # resistor
                temp_graph = change_device(new_graph, 7, device_index)
                temp_graph = add_series_device(temp_graph, 8, device, device_index)
            elif device == 7:   # capacitor
                temp_graph = add_parallel_device(new_graph, 8, device, device_index)
                temp_graph = add_series_device(temp_graph, 8, device, device_index)
            elif device == 8:   # inductor
                temp_graph = add_parallel_device(new_graph, 7, device, device_index)
                temp_graph = add_series_device(temp_graph, 7, device, device_index)

            temp_graph.set_graph_attributes(circuit=-graph.graph_attrs['circuit'])
            new_graphs.append(temp_graph)

    return new_graphs
####################################





########### Model Utils ############
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    if isinstance(layer, nn.BatchNorm1d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)



def send_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = send_to_device(batch[key], device)
    return batch



def build_layer(input_dim, hidden_dim, output_dim, num_layers,
                    activation='gelu', bias=True, dropout=0.2):
    """
    Builds a simple feedforward block with BatchNorm, activation,
    and an optional Dropout after each activation.

    Args:
        input_dim (int): Dimensionality of input features.
        hidden_dim (int): Dimensionality of hidden layers.
        output_dim (int): Dimensionality of output layer.
        num_layers (int): Number of linear layers in the block.
        activation (str): Activation to use: 'gelu'|'relu'|'sigmoid'|'tanh'.
        bias (bool): Whether Linear layers should have a bias term.
        dropout (float): Probability of dropout. 0.0 disables dropout.
    
    Returns:
        nn.Sequential: The resulting feedforward network.
    """
    act_map = {'gelu': nn.GELU, 'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU}
    activation_fn = act_map.get(activation, nn.GELU)    # Default : GELU

    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim, bias=bias))
    else:
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        for _ in range(num_layers - 2):
            layers.extend([nn.BatchNorm1d(hidden_dim), activation_fn(),
                           nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim, bias=bias)])
        layers.extend([nn.BatchNorm1d(hidden_dim), activation_fn(),
                       nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim, bias=bias)])

        return nn.Sequential(*layers)
####################################





########### Train Utils ############
def set_seed(seed, gpu_fix=True):
    """
    Set the seed for reproducibility in Python, NumPy, and PyTorch.
    Note: Full reproducibility is not guaranteed across different computing architectures.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations.
    if gpu_fix:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def sample_min_number_of_indices(labels):
        unique_labels, counts = torch.unique(labels, return_counts=True)
        min_count = counts.min().item()

        indices = []
        for lbl in unique_labels:
            lbl_indices = torch.where(labels == lbl)[0]
            perm = torch.randperm(len(lbl_indices))
            lbl_indices = lbl_indices[perm[:min_count]]
            indices.append(lbl_indices)
        indices = torch.cat(indices, dim=0)
        indices = indices[torch.randperm(len(indices))]

        return indices
####################################