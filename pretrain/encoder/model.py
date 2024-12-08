import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_add



def send_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = send_to_device(batch[key], device)
    return batch


def build_layer(input_dim, hidden_dim, output_dim, bias=True):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
    layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
    layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class GNNlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias=True):
        super(GNNlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, bias=bias)
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, bias=bias)

    def forward(self, nh, eh, edge_index):
        src_node_i, dst_node_i = edge_index

        n_h, e_h = self.nf_lin(nh), self.ef_lin(eh)
        src_nh, dst_nh = n_h[src_node_i], n_h[dst_node_i]   # (edge_num, hidden_dim)
        msg = src_nh * e_h                                  # (edge_num, hidden_dim)
        attn = torch.sum(msg * dst_nh, dim=-1)              # (edge_num)
        attn = scatter_softmax(attn, dst_node_i, dim=0)     # (edge_num)

        nz = scatter_add(attn.unsqueeze(-1) * src_nh,
                         dst_node_i, dim=0,
                         dim_size=nh.size(0))               # (node_num, hidden_dim)
        n_h = n_h + nz
        e_h = e_h + nz[src_node_i]

        return n_h, e_h




class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.gnn_layer_num = params['gnn_layer_num']

        # Layers
        self.nf_lin_init = build_layer(9, params['hidden_dim'], params['hidden_dim'], bias=True)
        self.ef_lin_init = build_layer(5, params['hidden_dim'], params['hidden_dim'], bias=True)
        self.gnn = GNNlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], bias=True)

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params["lr"])


    def forward(self, batch):
        nf = batch['x']
        edge_i = batch['edge_index']
        ef = batch['edge_attr']

        nh = self.nf_lin_init(nf)      # (node_num, hidden_dim)
        eh = self.ef_lin_init(ef)      # (edge_num, hidden_dim)

        ##### GNN Layers #####
        for _ in range(self.gnn_layer_num):
            nh, eh = self.gnn(nh, eh, edge_i)
        ######################
        gh = scatter_mean(nh, batch['batch'], dim=0,\
                         dim_size=batch['batch'].max().item()+1)

        # Logging information (optional)
        info = {}

        return nh, eh, gh, info


    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))




parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader

if __name__ == "__main__":

    # Load dataset
    dataset = torch.load('../dataset/pretraining_dataset_wo_device_params.pt')
    train_data = dataset['train_data']
    batch_size = 20
    dataloader = GraphDataLoader(train_data, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        
        # Load Parameters
        with open(f"./params.json", 'r') as f:
            model_params = json.load(f)['model']

        encoder = Encoder(model_params)

        nf = batch['x']
        edge_i = batch['edge_index']
        ef = batch['edge_attr']
        batch_index = batch['batch']

        nh, eh, gh, info = encoder(batch)

        print()        
        print(batch)
        print()
        print(nh)
        print()
        print(eh)
        print()
        print(gh)