import numpy as np


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