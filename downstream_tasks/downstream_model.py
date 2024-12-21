import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pretrain/encoder'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import build_layer
from model import DICE, GATlayer, GINlayer
from dataloader import GraphDataLoader



class GE_MLP(nn.Module):
    def __init__(self, params):
        super(GE_MLP, self).__init__()
        
        # Basic Linear Layers & Batch Norms
        self.nf_lin = nn.Sequential(nn.Linear(9, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin = nn.Sequential(nn.Linear(5, params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.gf_lin = nn.Sequential(nn.Linear(params['hidden_dim'], params['hidden_dim']), nn.GELU(), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.nf_batch_norm = nn.BatchNorm1d(params['hidden_dim'])
        self.ef_batch_norm = nn.BatchNorm1d(params['hidden_dim'])
        self.gf_batch_norm = nn.BatchNorm1d(params['hidden_dim'])

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)

    def forward(self, batch):
        # Multiply node features by device_params if they exist
        nf = batch['x'] * batch['device_params'].unsqueeze(-1)
        ef = batch['edge_attr']

        nf, ef = self.nf_lin(nf), self.ef_lin(ef)        
        nf, ef = self.nf_batch_norm(nf), self.ef_batch_norm(ef)

        gf = scatter_add(nf, batch['batch'], dim=0, dim_size=batch['batch'].max().item() + 1)
        gf = self.gf_lin(gf)
        gf = self.gf_batch_norm(gf)

        return nf, ef, gf

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))



class GE_GNN(nn.Module):
    def __init__(self, params, with_device_param=False):
        super(GE_GNN, self).__init__()
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



class DownstreamEncoder(nn.Module):
    def __init__(self, model_type, model_params):
        super(DownstreamEncoder, self).__init__()
        # Always build the GE_MLP
        self.ge_mlp = GE_MLP(model_params['ge_mlp'])
        
        # Depending on model_type, pick the structure encoder
        if model_type == 0:
            # model_type = 0 -> DICE (we want to freeze these params!)
            self.structure_encoder = DICE(model_params['dice'])
            self.structure_encoder.load(
                f"./pretrain/encoder/saved_models"\
                f"/DICE_pretrained_model"\
                f"_{model_params['dice']['gnn_type']}.pt"
            )
            # Freeze DICE parameters so they do NOT update:
            for param in self.structure_encoder.parameters():
                param.requires_grad = False

        elif model_type == 1:
            # No structure encoder
            self.structure_encoder = None
        elif model_type == 2:
            # GE_GNN without device params
            self.structure_encoder = GE_GNN(model_params['dice'], with_device_param=False)
        elif model_type == 3:
            # GE_GNN with device params
            self.structure_encoder = GE_GNN(model_params['dice'], with_device_param=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)


    def forward(self, batch):
        # non-structural encodings
        nf_ge_mlp, ef_ge_mlp, gf_ge_mlp = self.ge_mlp(batch)

        # structural encodings
        if self.structure_encoder is None:
            nf_structure, ef_structure, gf_structure =\
                torch.zeros_like(nf_ge_mlp), torch.zeros_like(ef_ge_mlp), torch.zeros_like(gf_ge_mlp)
        else:
            nf_structure, ef_structure, gf_structure = self.structure_encoder(batch)

        # combine
        nf = torch.cat([nf_ge_mlp, nf_structure], dim=1)
        ef = torch.cat([ef_ge_mlp, ef_structure], dim=1)
        gf = torch.cat([gf_ge_mlp, gf_structure], dim=1)

        return nf, ef, gf

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class DownstreamDecoder(nn.Module):
    def __init__(self, subtask_name, model_params):
        super(DownstreamDecoder, self).__init__()
        self.gf_lin = build_layer(
            input_dim=2*model_params['hidden_dim'],
            hidden_dim=model_params['hidden_dim'],
            output_dim=model_params['hidden_dim'],
            num_layers=model_params['layer_num'],
            activation='gelu',
            bias=True
        )
        self.gf_batch_norm = nn.BatchNorm1d(model_params['hidden_dim'])

        # Output layer depends on the subtask
        if subtask_name == "delay_prediction":
            self.output_layer = build_layer(
                input_dim=model_params['hidden_dim'],
                hidden_dim=model_params['hidden_dim'],
                output_dim=2,
                num_layers=model_params['layer_num'],
                activation='gelu',
                bias=True
            )
        else:
            raise ValueError(f"Unknown subtask: {subtask_name}")

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)

    def forward(self, nf, ef, gf, batch_indices):
        # Aggregate node features to get a graph-level representation
        gh = gf + scatter_add(nf, batch_indices, dim=0, dim_size=batch_indices.max().item() + 1)
        gf = self.gf_lin(gh)
        gf = self.gf_batch_norm(gf)
        output = self.output_layer(gf)
        return output

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class DownstreamModel(nn.Module):
    def __init__(self, subtask_name, model_type, model_params):
        super(DownstreamModel, self).__init__()
        # Encoder & Decoder
        self.encoder = DownstreamEncoder(model_type, model_params['encoder'])
        self.decoder = DownstreamDecoder(subtask_name, model_params['decoder'])

        # Layer Initialization
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        self.apply(init_weights)

        self.optimizer = None  # will be set externally (in train.py)

    def forward(self, batch):
        nf, ef, gf = self.encoder(batch)
        output = self.decoder(nf, ef, gf, batch["batch"])
        return output

    def save_encoder(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.encoder.state_dict(), path)

    def save_decoder(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.decoder.state_dict(), path)

    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))

    def load_decoder(self, path):
        self.decoder.load_state_dict(torch.load(path))

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))