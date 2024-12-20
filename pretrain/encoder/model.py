import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_mean



def send_to_device(batch, device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
        elif isinstance(batch[key], dict):
            batch[key] = send_to_device(batch[key], device)
    return batch


def build_layer(input_dim, hidden_dim, output_dim, layer_num, bias=True):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
    for _ in range(layer_num-2):
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
    layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
    return nn.Sequential(*layers)


class GNNlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True):
        super(GNNlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, bias=bias)
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, bias=bias)
        self.nf_bn = nn.BatchNorm1d(hidden_dim)
        self.ef_bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, nh, eh, edge_index):
        src_node_i, dst_node_i = edge_index

        n_h, e_h = self.nf_lin(nh), self.ef_lin(eh)
        src_nh, dst_nh = n_h[src_node_i], n_h[dst_node_i]   # (edge_num, hidden_dim)
        msg = src_nh + e_h                                  # (edge_num, hidden_dim)
        attn = torch.sum(msg * dst_nh, dim=-1)              # (edge_num)
        attn = scatter_softmax(attn, dst_node_i, dim=0)     # (edge_num)

        nz = scatter_add(attn.unsqueeze(-1) * src_nh,
                         dst_node_i, dim=0,
                         dim_size=nh.size(0))               # (node_num, hidden_dim)
        n_h = n_h + nz
        e_h = e_h * ( 1 + nz[src_node_i] - nz[dst_node_i] )

        n_h, e_h = self.nf_bn(n_h), self.ef_bn(e_h)

        return n_h, e_h


class GINlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, bias=True):
        super(GINlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, bias=bias)
        self.nf_eps = nn.Parameter(torch.zeros(input_dim))
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, bias=bias)
        self.ef_eps = nn.Parameter(torch.zeros(input_dim))
        self.nf_bn = nn.BatchNorm1d(hidden_dim)
        self.ef_bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, nh, eh, edge_index):
        src_node_i, dst_node_i = edge_index

        src_nh, dst_nh = nh[src_node_i], nh[dst_node_i]     # (edge_num, hidden_dim)
        msg = src_nh + eh                                   # (edge_num, hidden_dim)
        attn = torch.sum(msg * dst_nh, dim=-1)              # (edge_num)
        attn = scatter_softmax(attn, dst_node_i, dim=0)     # (edge_num)

        nz = scatter_add(src_nh * attn.unsqueeze(-1),
                         dst_node_i, dim=0,
                         dim_size=nh.size(0))               # (node_num, hidden_dim)

        n_h = (1+self.nf_eps) * nh + nz
        e_h = (1+self.ef_eps) * eh + nz[src_node_i] - nz[dst_node_i]

        n_h, e_h = self.nf_lin(n_h), self.ef_lin(e_h)
        n_h, e_h = self.nf_bn(n_h), self.ef_bn(e_h)

        return n_h, e_h



class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.gnn_depth = params['gnn_depth']

        # Layers
        self.nf_lin_init = build_layer(9, params['hidden_dim'], params['hidden_dim'],
                                          params['layer_num'], bias=True)
        self.ef_lin_init = build_layer(5, params['hidden_dim'], params['hidden_dim'],
                                          params['layer_num'], bias=True)
        self.nh_batch_norm_1 = nn.BatchNorm1d(params['hidden_dim'])
        self.nh_batch_norm_2 = nn.BatchNorm1d(params['hidden_dim'])
        self.eh_batch_norm_1 = nn.BatchNorm1d(params['hidden_dim'])
        self.eh_batch_norm_2 = nn.BatchNorm1d(params['hidden_dim'])
        self.gh_lin = build_layer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                  params['layer_num'], bias=True)
        self.gh_batch_norm = nn.BatchNorm1d(params['hidden_dim'])
        # self.gnn = GNNlayer(params['hidden_dim'], params['hidden_dim'],
        #                     params['hidden_dim'], params['layer_num'], bias=True)
        self.gnn = GINlayer(params['hidden_dim'], params['hidden_dim'],
                            params['hidden_dim'], params['layer_num'], bias=True)


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

        nh, eh = self.nh_batch_norm_1(nh), self.eh_batch_norm_1(eh)

        ##### GNN Layers #####
        for _ in range(self.gnn_depth):
            nh, eh = self.gnn(nh, eh, edge_i)
        ######################

        nh, eh = self.nh_batch_norm_2(nh), self.eh_batch_norm_2(eh)

        gh = scatter_add(nh, batch['batch'], dim=0,\
                          dim_size=batch['batch'].max().item()+1)
        gh = self.gh_lin(gh)
        gh = self.gh_batch_norm(gh)

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