import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add, scatter_mean

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import build_layer, init_weights



class GATlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, activation, bias):
        super(GATlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias, dropout=0.0)
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias, dropout=0.0)
        
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

        return n_h, e_h



class GINlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, activation, bias):
        super(GINlayer, self).__init__()
        self.nf_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias, dropout=0.0)
        self.nf_eps = nn.Parameter(torch.FloatTensor([0.0]))
        self.ef_lin = build_layer(input_dim, hidden_dim, output_dim, layer_num, activation, bias, dropout=0.0)
        self.ef_eps = nn.Parameter(torch.FloatTensor([0.0]))
        
    def forward(self, nh, eh, edge_index):
        src_node_i, dst_node_i = edge_index

        src_nh, dst_nh = nh[src_node_i], nh[dst_node_i]     # (edge_num, hidden_dim)
        msg = src_nh * eh                                   # (edge_num, hidden_dim)
        
        nz = scatter_add(msg, dst_node_i, dim=0, dim_size=nh.size(0))     # (node_num, hidden_dim)

        n_h = (1+self.nf_eps) * nh + nz
        e_h = (1+self.ef_eps) * eh + nz[src_node_i] - nz[dst_node_i]

        n_h, e_h = self.nf_lin(n_h), self.ef_lin(e_h)

        return n_h, e_h





class DICE(nn.Module):
    def __init__(self, params, gnn_depth=3):
        super(DICE, self).__init__()
        self.gnn_depth = gnn_depth

        # Linear Layers
        self.init_lin = nn.ModuleDict({
            'nf': nn.Sequential(nn.Linear(params['nf_in_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['hidden_dim'])),
            'ef': nn.Sequential(nn.Linear(params['ef_in_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['hidden_dim']))
        })
        self.lin_out = nn.ModuleDict({
            'nf': nn.Sequential(nn.Linear(params['hidden_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['nf_out_dim'])),
            'ef': nn.Sequential(nn.Linear(params['hidden_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['ef_out_dim'])),
            'gf': nn.Sequential(nn.Linear(params['hidden_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['gf_out_dim']))
        })

        # GNN Layers
        if params['gnn_type'] == 'GIN':
            self.gnn = nn.ModuleList([GINlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True) for _ in range(self.gnn_depth)])
        elif params['gnn_type'] == 'GAT':
            self.gnn = nn.ModuleList([GATlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True) for _ in range(self.gnn_depth)])


    def forward(self, batch):
        nf, ef, edge_i = batch['x'], batch['edge_attr'], batch['edge_index']
        
        ####### Init Linear Layers #######
        nh, eh = self.init_lin['nf'](nf), self.init_lin['ef'](ef)
        ##################################

        ########## GNN Layers ############
        for i in range(self.gnn_depth):
            nh, eh = self.gnn[i](nh, eh, edge_i)
        ##################################

        ######## Graph Embeddings ########
        gh = scatter_add(nh, batch['batch'], dim=0, dim_size=batch['batch'].max().item() + 1)\
            + scatter_add(eh, batch['batch'][edge_i[0]], dim=0, dim_size=batch['batch'][edge_i[0]].max().item() + 1)
        gh = self.lin_out['gf'](gh)
        ##################################
        
        ##### Node / Edge Embeddings #####
        nh = self.lin_out['nf'](nh).unsqueeze(1)    # (N, 1, nf_out_dim)
        eh = self.lin_out['ef'](eh).unsqueeze(1)    # (E, 1, ef_out_dim)

        nh_label_one_hot = F.one_hot(batch['node_y'], num_classes=9).float().unsqueeze(-1) # (N, 9, 1)
        eh_label_one_hot = F.one_hot(batch['edge_y'], num_classes=5).float().unsqueeze(-1) # (E, 5, 1)

        nh_3d = nh_label_one_hot * nh   # (N, 9, nf_out_dim)
        eh_3d = eh_label_one_hot * eh   # (E, 5, ef_out_dim)

        nh = nh_3d.view(nh.size(0), -1) # (N, 9*nf_out_dim)
        eh = eh_3d.view(eh.size(0), -1) # (E, 5*ef_out_dim)
        ##################################
        
        return nh, eh, gh


    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))