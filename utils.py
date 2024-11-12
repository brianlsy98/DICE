import re
import numpy as np
import torch



def convert_to_number(value_str):
    """
    Convert a string with unit suffix to a numeric value.
    Supports: 'p' (pico), 'n' (nano), 'u' (micro), 'm' (milli), 'k' (kilo), 'M' (mega)
    """
    # Define the conversion factors for each unit suffix
    unit_multipliers = {
        'p': 1e-12,  # pico
        'n': 1e-9,   # nano
        'u': 1e-6,   # micro
        'm': 1e-3,   # milli
        'k': 1e3,    # kilo
        'M': 1e6     # mega
    }
    
    # Extract the numeric part and the unit suffix
    if value_str[-1] in unit_multipliers:
        number = float(value_str[:-1])  # Get the numeric part
        multiplier = unit_multipliers[value_str[-1]]  # Get the corresponding multiplier
        return number * multiplier
    else:
        # If there's no unit suffix, return the number directly
        return float(value_str)



def parse_sim_netlist(filename):
    input_nodes = set()     # To store input nodes
    nodes = set()           # To store in/out nodes
    edges = {}              # To store edges with edge types and features (W, L)
    graph_level_attr = {}

    # Read the netlist file
    with open(filename, 'r') as file:
        for line in file:
            
            line = line.strip()
            parts = line.split()

            ####### Graph Level Attributes #######
            if line.startswith('.include'):
                including_filename = parts[1]
                
                # subcircuit
                if including_filename[-4:] == ".cir":
                    ns, es, glas = parse_sim_netlist(including_filename)
                    nodes.update(ns)
                    for e in es: edges.append(e)
                    for k, v in glas: graph_level_attr[k] = v
                # technology file
                else:
                    graph_level_attr['TECH'] = including_filename
                    
            elif line.startswith('.option'):
                option_parts = parts[1].split("=")
                graph_level_attr[option_parts[0]] = option_parts[1]                
            ####### Graph Level Attributes #######


            ####### Graph Structure & Edges #######
            ## MOSFET
            elif line.startswith('M'):
                mosfet_name = parts[0]
                drain_node, gate_node, source_node, body_node = parts[1:5]
                mosfet_type = parts[5]
                params = ' '.join(parts[5:])  # Get the rest of the line containing W, L
                w_match = re.search(r'W=(\d+\.?\d*[munp]?)', params)
                l_match = re.search(r'L=(\d+\.?\d*[munp]?)', params)
                W = convert_to_number(w_match.group(1)) if w_match else None  # Extract width (W)
                L = convert_to_number(l_match.group(1)) if l_match else None  # Extract length (L)
                # Node Update
                nodes.update([drain_node, gate_node, source_node, body_node])
                # Edge Update
                edges[(drain_node, mosfet_type, source_node)] = [gate_node, body_node, W/L]
            ## Resistor
            elif line.startswith('R'):
                resistor_name, node_1, node_2, value = parts[:4]
                nodes.update([node_1, node_2])
                edges[(node_1, 'R', node_2)] = [convert_to_number(value)]
            ## Inductor
            elif line.startswith('L'):
                inductor_name, node_1, node_2, value = parts[:4]
                nodes.update([node_1, node_2])
                edges[(node_1, 'L', node_2)] = [convert_to_number(value)]
            ## Capacitor
            elif line.startswith('C'):
                capacitor_name, node_1, node_2, value = parts[:4]
                nodes.update([node_1, node_2])
                edges[(node_1, 'C', node_2)] = [convert_to_number(value)]
            ####### Graph Structure & Edges #######


            ####### Input Nodes #######
            elif line.startswith('V'):
                input_nodes.update([parts[1]])
            ####### Input Nodes #######


    return list(nodes), list(input_nodes), list(nodes-input_nodes), edges, graph_level_attr



def parse_sim_result(filename):

    data_dict = {}

    with open(filename, 'r') as file:
        # Read the header line and extract variable names
        header_line = file.readline().strip()
        column_names = header_line.split()
        
        # Use regex to extract variable names inside v()
        variable_names = [re.search(r'v\((.*?)\)', col).group(1) for col in column_names]
        
        # Initialize the dictionary with variable names
        for var in variable_names:
            data_dict[var] = []
        
        # Read and parse the data lines
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # Split the line into values and convert to floats
            values = [float(val) for val in line.strip().split()]
            
            # Assign values to the corresponding variable in the dictionary
            for var, val in zip(variable_names, values):
                data_dict[var].append(val)

    return data_dict


