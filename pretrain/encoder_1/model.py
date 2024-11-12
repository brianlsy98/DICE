import os
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax


def send_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = send_to_device(batch[key], device)
    return batch



class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.node_types = ['input', 'output']
        self.rel_types = ['nmos', 'pmos', 'R', 'L', 'C']
        self.hidden_dim = params['hidden_dim']

        def build_ef1_layers(input_dim, output_dim, bias=True):
            layers = []
            layers.append(nn.Linear(input_dim, params['hidden_dim'], bias=True))
            layers.append(nn.GELU())
            layers.append(nn.Linear(params['hidden_dim'], params['hidden_dim'], bias=bias))
            layers.append(nn.GELU())
            layers.append(nn.Linear(params['hidden_dim'], output_dim, bias=bias))
            return nn.Sequential(*layers)
        
        def build_ef2_layers(input_dim, output_dim, bias=True):
            layers = []
            layers.append(nn.Linear(input_dim, params['hidden_dim'], bias=True))
            layers.append(nn.GELU())
            for _ in range(params['layer_num'] - 2):
                layers.append(nn.Linear(params['hidden_dim'], params['hidden_dim'], bias=True))
                layers.append(nn.GELU())
            layers.append(nn.Linear(params['hidden_dim'], output_dim, bias=bias))
            return nn.Sequential(*layers)

        # Layers
        self.nf1_input = nn.Linear(1, params['hidden_dim'], bias=True)
        self.ef1_embed = nn.ModuleDict({
            'nmos': build_ef1_layers(1, params['hidden_dim'], bias=True),
            'pmos': build_ef1_layers(1, params['hidden_dim'], bias=True),
            'R': build_ef1_layers(1, params['hidden_dim'], bias=True),
            'L': build_ef1_layers(1, params['hidden_dim'], bias=True),
            'C': build_ef1_layers(1, params['hidden_dim'], bias=True),
        })
        self.ef2_embed = nn.ModuleDict({
            'nmos': build_ef2_layers(5*params['hidden_dim'], params['hidden_dim'], bias=True),
            'pmos': build_ef2_layers(5*params['hidden_dim'], params['hidden_dim'], bias=True),
            'R': build_ef2_layers(3*params['hidden_dim'], params['hidden_dim'], bias=True),
            'L': build_ef2_layers(3*params['hidden_dim'], params['hidden_dim'], bias=True),
            'C': build_ef2_layers(3*params['hidden_dim'], params['hidden_dim'], bias=True),
        })
        self.ef2_self_embed = build_ef2_layers(params['hidden_dim'], params['hidden_dim'], bias=True)
        self.score = nn.Linear(params['hidden_dim'], params['hidden_dim'], bias=True)


        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["lr"])



    def forward(self, batch):

        all_device_params = []
        all_vg_vb_i_mos = []         # nmos / pmos

        all_src_n_types = []
        all_rel_types = []
        all_dst_n_types = []

        all_src_nodes, all_src_node_voltages = [], []
        all_dst_nodes, all_dst_node_voltages = [], []

        ### Initial Data Preparation
        ## Compute Edge Features 1
        for edge_type in batch['edge_types']:
            src_type, rel_type, dst_type = edge_type

            # Get edge index and edge attributes
            edge_index = batch[edge_type]['edge_index']                 # Shape: [2, num_edges]
            edge_attr = batch[edge_type]['edge_attr']                   # Shape: [num_edges, edge_attr_dim]
            
            # Get Edge feature 1 (device_params)
            device_params = edge_attr[:, -1].unsqueeze(-1)              # Shape: [num_edges, 1]
            # device_params = self.ef1_embed[rel_type](device_params)   # Shape: [num_edges, 1]
            if rel_type == 'nmos' or rel_type == 'pmos':
                vg_vb_i_mos = edge_attr[:, :-1]                         # Shape: [num_edges, 4]
            else: vg_vb_i_mos = torch.zeros((edge_attr.size(0),4), device=edge_attr.device)
                                                                        # Shape: [num_edges, 4]

            # Get node indices
            src_indices = edge_index[0].long()                          # Shape: [num_edges]
            dst_indices = edge_index[1].long()                          # Shape: [num_edges]
            # Get node voltage values
            src_node_voltages = batch[src_type]['dc_voltages'][src_indices]     # Shape: [num_edges]
            dst_node_voltages = batch[dst_type]['dc_voltages'][dst_indices]     # Shape: [num_edges]

            # for tensor of [total_num_edges]
            all_device_params.append(device_params)
            all_vg_vb_i_mos.append(vg_vb_i_mos)
            all_src_n_types.append(torch.ones_like(src_indices) * self.node_types.index(src_type))
            all_rel_types.append(torch.ones_like(src_indices) * self.rel_types.index(rel_type))
            all_dst_n_types.append(torch.ones_like(dst_indices) * self.node_types.index(dst_type))

            all_src_nodes.append(src_indices)
            all_src_node_voltages.append(src_node_voltages)
            all_dst_nodes.append(dst_indices)
            all_dst_node_voltages.append(dst_node_voltages)

        # Concatenate across all edge types
        all_device_params = torch.cat(all_device_params, dim=0)             # Shape: [total_num_edges, 1]
        all_vg_vb_i_mos = torch.cat(all_vg_vb_i_mos, dim=0)                 # Shape: [total_num_edges, 4]
        all_src_n_types = torch.cat(all_src_n_types, dim=0)                 # Shape: [total_num_edges]
        all_rel_types = torch.cat(all_rel_types, dim=0)                     # Shape: [total_num_edges]
        all_dst_n_types = torch.cat(all_dst_n_types, dim=0)                 # Shape: [total_num_edges]
        all_src_nodes = torch.cat(all_src_nodes, dim=0)                     # Shape: [total_num_edges]
        all_src_node_voltages = torch.cat(all_src_node_voltages, dim=0)     # Shape: [total_num_edges]
        all_dst_nodes = torch.cat(all_dst_nodes, dim=0)                     # Shape: [total_num_edges]
        all_dst_node_voltages = torch.cat(all_dst_node_voltages, dim=0)     # Shape: [total_num_edges]

        total_num_edges = all_device_params.size(0)
        input_num_nodes = batch['input']['dc_voltages'].size(0)
        output_num_nodes = batch['output']['dc_voltages'].size(0)
        device = all_device_params.device


        # masks for each node types
        src_node_input_indices = torch.where(all_src_n_types == 0)
        src_node_output_indices = torch.where(all_src_n_types == 1)
        dst_node_input_indices = torch.where(all_dst_n_types == 0)
        dst_node_output_indices = torch.where(all_dst_n_types == 1)
        # masks for each edge types
        nmos_indices = torch.where(all_rel_types == 0)
        pmos_indices = torch.where(all_rel_types == 1)
        R_indices = torch.where(all_rel_types == 2)
        L_indices = torch.where(all_rel_types == 3)
        C_indices = torch.where(all_rel_types == 4)

        
        ## Compute Edge Features 1
        ef1_nmos = self.ef1_embed['nmos'](all_device_params[nmos_indices])    # Shape: [nmos_num_edges, hidden_dim]
        ef1_pmos = self.ef1_embed['pmos'](all_device_params[pmos_indices])    # Shape: [pmos_num_edges, hidden_dim]
        ef1_R = self.ef1_embed['R'](all_device_params[R_indices])             # Shape: [R_num_edges, hidden_dim]
        ef1_L = self.ef1_embed['L'](all_device_params[L_indices])             # Shape: [L_num_edges, hidden_dim]
        ef1_C = self.ef1_embed['C'](all_device_params[C_indices])             # Shape: [C_num_edges, hidden_dim]

        ef1 = torch.zeros((total_num_edges, self.hidden_dim), device=device) # Shape: [total_num_edges, hidden_dim]
        ef1[nmos_indices] = ef1_nmos
        ef1[pmos_indices] = ef1_pmos
        ef1[R_indices] = ef1_R
        ef1[L_indices] = ef1_L
        ef1[C_indices] = ef1_C


        ## Compute Node Features 1 (input)    --->   [input_num_nodes, hidden_dim]
        nf1_input = batch['input']['dc_voltages'].unsqueeze(-1)               # Shape: [input_num_nodes, 1]
        nf1_input = self.nf1_input(nf1_input)                                 # Shape: [input_num_nodes, hidden_dim]
        ## Compute Node Features 1 (output)   --->   [output_num_nodes, hidden_dim]
        dst_attention_output = torch.matmul(ef1, ef1.T)[dst_node_output_indices]                      # Shape: [num_edges_with_dst_output, total_num_edges]
        dst_attention_output = torch.matmul(dst_attention_output, ef1)                                # Shape: [num_edges_with_dst_output, hidden_dim]
        dst_attention_output = softmax(dst_attention_output, all_dst_nodes[dst_node_output_indices])  # Shape: [num_edges_with_dst_output, hidden_dim]

        nf1_output = scatter_add(dst_attention_output * ef1[dst_node_output_indices].squeeze(-1), index=all_dst_nodes[dst_node_output_indices],
                                 dim=0, dim_size=output_num_nodes)                         # Shape: [output_num_nodes, hidden_dim]

        nf1_src = torch.zeros((total_num_edges, self.hidden_dim), device=device)               # Shape: [total_num_edges, hidden_dim]
        nf1_src[src_node_input_indices] = nf1_input[all_src_nodes[src_node_input_indices]]
        nf1_src[src_node_output_indices] = nf1_output[all_src_nodes[src_node_output_indices]]
        nf1_dst = torch.zeros((total_num_edges, self.hidden_dim), device=device)               # Shape: [total_num_edges, hidden_dim]
        nf1_dst[dst_node_input_indices] = nf1_input[all_dst_nodes[dst_node_input_indices]]
        nf1_dst[dst_node_output_indices] = nf1_output[all_dst_nodes[dst_node_output_indices]]
        
        ## Compute Edge Features 2
        # nmos
        vg_nmos_node_types = all_vg_vb_i_mos[nmos_indices][:, 0].long()                 # Shape: [nmos_num_edges]
        vg_nmos_input_indices = torch.where(vg_nmos_node_types == 0)
        vg_nmos_output_indices = torch.where(vg_nmos_node_types == 1)
        vb_nmos_node_types = all_vg_vb_i_mos[nmos_indices][:, 2].long()                 # Shape: [nmos_num_edges]
        vb_nmos_input_indices = torch.where(vb_nmos_node_types == 0)
        vb_nmos_output_indices = torch.where(vb_nmos_node_types == 1)

        vg_nmos_node_indices = all_vg_vb_i_mos[nmos_indices][:, 1].long()               # Shape: [nmos_num_edges]
        vb_nmos_node_indices = all_vg_vb_i_mos[nmos_indices][:, 3].long()               # Shape: [nmos_num_edges]
        nf1_input_vg_nmos = nf1_input[vg_nmos_node_indices[vg_nmos_input_indices]]
        nf1_input_vb_nmos = nf1_input[vb_nmos_node_indices[vb_nmos_input_indices]]
        nf1_output_vg_nmos = nf1_output[vg_nmos_node_indices[vg_nmos_output_indices]]
        nf1_output_vb_nmos = nf1_output[vb_nmos_node_indices[vb_nmos_output_indices]]

        nf1_vg_nmos = torch.zeros((nf1_input_vg_nmos.size(0) + nf1_output_vg_nmos.size(0), self.hidden_dim), device=device)
        nf1_vg_nmos[vg_nmos_input_indices] = nf1_input_vg_nmos
        nf1_vg_nmos[vg_nmos_output_indices] = nf1_output_vg_nmos
        nf1_vb_nmos = torch.zeros((nf1_input_vb_nmos.size(0) + nf1_output_vb_nmos.size(0), self.hidden_dim), device=device)
        nf1_vb_nmos[vb_nmos_input_indices] = nf1_input_vb_nmos
        nf1_vb_nmos[vb_nmos_output_indices] = nf1_output_vb_nmos

        # pmos
        vg_pmos_node_types = all_vg_vb_i_mos[pmos_indices][:, 0].long()                 # Shape: [pmos_num_edges]
        vg_pmos_input_indices = torch.where(vg_pmos_node_types == 0)
        vg_pmos_output_indices = torch.where(vg_pmos_node_types == 1)
        vb_pmos_node_types = all_vg_vb_i_mos[pmos_indices][:, 2].long()                 # Shape: [pmos_num_edges]
        vb_pmos_input_indices = torch.where(vb_pmos_node_types == 0)
        vb_pmos_output_indices = torch.where(vb_pmos_node_types == 1)

        vg_pmos_node_indices = all_vg_vb_i_mos[pmos_indices][:, 1].long()               # Shape: [pmos_num_edges]
        vb_pmos_node_indices = all_vg_vb_i_mos[pmos_indices][:, 3].long()               # Shape: [pmos_num_edges]
        nf1_input_vg_pmos = nf1_input[vg_pmos_node_indices[vg_pmos_input_indices]]
        nf1_input_vb_pmos = nf1_input[vb_pmos_node_indices[vb_pmos_input_indices]]
        nf1_output_vg_pmos = nf1_output[vg_pmos_node_indices[vg_pmos_output_indices]]
        nf1_output_vb_pmos = nf1_output[vb_pmos_node_indices[vb_pmos_output_indices]]
        
        nf1_vg_pmos = torch.zeros((nf1_input_vg_pmos.size(0) + nf1_output_vg_pmos.size(0), self.hidden_dim), device=device)
        nf1_vg_pmos[vg_pmos_input_indices] = nf1_input_vg_pmos
        nf1_vg_pmos[vg_pmos_output_indices] = nf1_output_vg_pmos
        nf1_vb_pmos = torch.zeros((nf1_input_vb_pmos.size(0) + nf1_output_vb_pmos.size(0), self.hidden_dim), device=device)
        nf1_vb_pmos[vb_pmos_input_indices] = nf1_input_vb_pmos
        nf1_vb_pmos[vb_pmos_output_indices] = nf1_output_vb_pmos

        # non self-loop edge features 2
        ef2_nmos = self.ef2_embed['nmos'](torch.cat([nf1_src[nmos_indices],             # Shape: [nmos_num_edges, hidden_dim]
                                                     nf1_dst[nmos_indices],
                                                     ef1[nmos_indices],
                                                     nf1_vg_nmos,
                                                     nf1_vb_nmos], dim=1))
        ef2_pmos = self.ef2_embed['pmos'](torch.cat([nf1_src[pmos_indices],             # Shape: [pmos_num_edges, hidden_dim]
                                                     nf1_dst[pmos_indices],
                                                     ef1[pmos_indices],
                                                     nf1_vg_pmos,
                                                     nf1_vb_pmos], dim=1))
        ef2_R = self.ef2_embed['R'](torch.cat([nf1_src[R_indices],                      # Shape: [R_num_edges, hidden_dim]
                                               nf1_dst[R_indices],
                                               ef1[R_indices]], dim=1))
        ef2_L = self.ef2_embed['L'](torch.cat([nf1_src[L_indices],                      # Shape: [L_num_edges, hidden_dim]
                                               nf1_dst[L_indices],
                                               ef1[L_indices]], dim=1))
        ef2_C = self.ef2_embed['C'](torch.cat([nf1_src[C_indices],                      # Shape: [C_num_edges, hidden_dim]
                                               nf1_dst[C_indices],
                                               ef1[C_indices]], dim=1))

        ef2 = torch.zeros((all_device_params.size(0), self.hidden_dim), device=all_device_params.device)
                                                      # Shape: [total_num_edges, hidden_dim]
        ef2[nmos_indices] = ef2_nmos
        ef2[pmos_indices] = ef2_pmos
        ef2[R_indices] = ef2_R
        ef2[L_indices] = ef2_L
        ef2[C_indices] = ef2_C
 
        # self-loop edge features 2
        ef2_self = self.ef2_self_embed(nf1_output)    # Shape: [output_num_nodes, hidden_dim]

        ## Compute node features 2
        score_ij = self.score(ef2[dst_node_output_indices]) # Shape: [num_edges_with_dst_output, hidden_dim]
        score_ii = self.score(ef2_self)                     # Shape: [output_num_nodes, hidden_dim]
        scores = torch.cat([score_ij, score_ii], dim=0)     # Shape: [output_num_nodes + num_edges_with_dst_output, hidden_dim]
        output_destination_nodes_indices = torch.cat([all_dst_nodes[dst_node_output_indices],
                                                      torch.arange(output_num_nodes, device=device)], dim=0)
                                                            # Shape: [output_num_nodes + num_edges_with_dst_output]
        output_attn = softmax(scores, output_destination_nodes_indices)    # Shape: [output_num_nodes + num_edges_with_dst_output, hidden_dim]
        nf2_output = scatter_add(output_attn * nf1_output[output_destination_nodes_indices],
                                 index=output_destination_nodes_indices, dim=0,
                                 dim_size=output_num_nodes)                # Shape: [output_num_nodes, hidden_dim]

        # Logging information (optional)
        info = {}

        return nf2_output.squeeze(), info


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))





class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        
        def build_layer(input_dim, output_dim, bias=True):
            layers = []
            layers.append(nn.Linear(input_dim, params['hidden_dim'], bias=True))
            layers.append(nn.GELU())
            for _ in range(params['layer_num'] - 2):
                layers.append(nn.Linear(params['hidden_dim'], params['hidden_dim'], bias=True))
                layers.append(nn.GELU())
            layers.append(nn.Linear(params['hidden_dim'], output_dim, bias=bias))
            layers.append(nn.Sigmoid())
            return nn.Sequential(*layers)

        # Layer
        self.dc_voltage_predict = build_layer(params['hidden_dim'], 1, bias=True)

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["lr"])


    def forward(self, z):
        v = self.dc_voltage_predict(z)
        info = {}
        return v.squeeze(), info


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))




class PretrainModel(nn.Module):
    
        def __init__(self, params):
            super(PretrainModel, self).__init__()
            self.encoder = Encoder(params)
            self.decoder = Decoder(params)
            self.optimizer = torch.optim.Adam(list(self.encoder.parameters())\
                                              + list(self.decoder.parameters()), lr=params["lr"])
    
        def forward(self, batch):
            z, e_info = self.encoder(batch)
            v, d_info = self.decoder(z)
    
            return v, e_info, d_info
    
        def save(self, path):
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.encoder.save(f"{directory}/encoder.pt")
            self.decoder.save(f"{directory}/decoder.pt")
    
        def load(self, path):
            directory = os.path.dirname(path)
            self.encoder.load(f"{directory}/encoder.pt")
            self.decoder.load(f"{directory}/decoder.pt")







import json
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), './dataset/data/torch_datasets'))
sys.path.append(parent_dir)

from dataloader import HeteroGraphDataLoader

if __name__ == "__main__":

    dataset = torch.load('./dataset/data/torch_datasets/dc.pt')
    train_data = dataset['train_data']
    dataloader = HeteroGraphDataLoader(train_data, batch_size=2, shuffle=True)

    for batch in dataloader:
        print()
        print(batch)
        print()

        
        # Load Parameters
        with open(f"./model_params.json", 'r') as f:
            model_params = json.load(f)

        model = PretrainModel(model_params)
        v, e_info, d_info = model(batch)
        # print()
        # print(f"v_shape: {v.shape}")
        # print(f"v: {v}")

        encoder = Encoder(model_params)
        z, info = encoder(batch)

        decoder = Decoder(model_params)
        v, info = decoder(z)        
        
        print()
        print(f"z_shape: {z.shape}")
        print(f"z: {z}")

        print()
        print(f"v_shape: {v.shape}")
        print(f"v: {v}")

        print()
        print(f"dc_voltages_shape: {batch['output']['dc_voltages'].shape}")
        print(f"dc_voltages: {batch['output']['dc_voltages']}")
        print()        

        break

