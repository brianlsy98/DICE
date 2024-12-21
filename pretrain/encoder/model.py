import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_mean

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import build_layer


class GATlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, activation, bias=True):
        super(GATlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias=bias)
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias=bias)
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
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, activation, bias=True):
        super(GINlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias=bias)
        self.nf_eps = nn.Parameter(torch.zeros(input_dim))
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias=bias)
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



class DICE(nn.Module):

    def __init__(self, params):
        super(DICE, self).__init__()
        self.gnn_depth = params['gnn_depth']

        # Linear & BatchNorm Layers
        self.nf_lin_for_n = nn.Sequential(nn.Linear(9, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin_for_n = nn.Sequential(nn.Linear(5, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.nf_lin_for_e = nn.Sequential(nn.Linear(9, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin_for_e = nn.Sequential(nn.Linear(5, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.nf_lin_for_g = nn.Sequential(nn.Linear(9, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin_for_g = nn.Sequential(nn.Linear(5, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.nh_batch_norm_for_n = nn.BatchNorm1d(params['hidden_dim'])
        self.eh_batch_norm_for_n = nn.BatchNorm1d(params['hidden_dim'])
        self.nh_batch_norm_for_e = nn.BatchNorm1d(params['hidden_dim'])
        self.eh_batch_norm_for_e = nn.BatchNorm1d(params['hidden_dim'])
        self.nh_batch_norm_for_g = nn.BatchNorm1d(params['hidden_dim'])
        self.eh_batch_norm_for_g = nn.BatchNorm1d(params['hidden_dim'])
        
        # GNN Layers
        if params['gnn_type'] == 'GIN':
            self.gnn_nf = GINlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                   params['layer_num'], activation='gelu', bias=True)
            self.gnn_ef = GINlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                   params['layer_num'], activation='gelu', bias=True)
            self.gnn_gf = GINlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                   params['layer_num'], activation='gelu', bias=True)
        elif params['gnn_type'] == 'GAT':
            self.gnn_nf = GATlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                   params['layer_num'], activation='gelu', bias=True)
            self.gnn_ef = GATlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                   params['layer_num'], activation='gelu', bias=True)
            self.gnn_gf = GATlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                   params['layer_num'], activation='gelu', bias=True)

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)


    def forward(self, batch):
        nf = batch['x']
        edge_i = batch['edge_index']
        ef = batch['edge_attr']

        ### Linear & BatchNorm Layers ###
        nh_for_n, eh_for_n = self.nf_lin_for_n(nf), self.ef_lin_for_n(ef)
        nh_for_n, eh_for_n = self.nh_batch_norm_for_n(nh_for_n), self.eh_batch_norm_for_n(eh_for_n)
        nh_for_e, eh_for_e = self.nf_lin_for_e(nf), self.ef_lin_for_e(ef)
        nh_for_e, eh_for_e = self.nh_batch_norm_for_e(nh_for_e), self.eh_batch_norm_for_e(eh_for_e)
        nh_for_g, eh_for_g = self.nf_lin_for_g(nf), self.ef_lin_for_g(ef)
        nh_for_g, eh_for_g = self.nh_batch_norm_for_g(nh_for_g), self.eh_batch_norm_for_g(eh_for_g)
        #################################

        ########## GNN Layers ###########
        for _ in range(self.gnn_depth):
            nh_for_n, eh_for_n = self.gnn_nf(nh_for_n, eh_for_n, edge_i)
            nh_for_e, eh_for_e = self.gnn_ef(nh_for_e, eh_for_e, edge_i)
            nh_for_g, eh_for_g = self.gnn_gf(nh_for_g, eh_for_g, edge_i)
        #################################

        gh_for_g = scatter_add(nh_for_g, batch['batch'], dim=0, dim_size=batch['batch'].max().item() + 1)

        return nh_for_n, eh_for_e, gh_for_g


    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))




parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from dataloader import GraphDataLoader

if __name__ == "__main__":

    # Load dataset
    dataset = torch.load('./pretrain/dataset/pretraining_dataset_wo_device_params.pt')
    train_data = dataset['train_data']
    batch_size = 20
    dataloader = GraphDataLoader(train_data, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        
        # Load Parameters
        with open(f"./params.json", 'r') as f:
            model_params = json.load(f)['model']['dice']

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