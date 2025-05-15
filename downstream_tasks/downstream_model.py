import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pretrain/encoder'))
sys.path.append(parent_dir)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import build_layer
from model import DICE, GINlayer_w_ef_update
from model import GCNlayer, GraphSAGElayer, GATlayer, GINlayer


############################################################################################################
# ENCODER
############################################################################################################
class Added_GNN(nn.Module):
    def __init__(self, params, gnn_depth=2):
        super(Added_GNN, self).__init__()
        self.gnn_depth = gnn_depth
        self.nf_in_dim = params['nf_in_dim']
        self.ef_in_dim = params['ef_in_dim']

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
        if params['gnn_type'] == 'GCN':
            self.gnn = nn.ModuleList([GCNlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True, params['dropout']) for _ in range(self.gnn_depth)])
        elif params['gnn_type'] == 'GraphSAGE':
            self.gnn = nn.ModuleList([GraphSAGElayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True, params['dropout']) for _ in range(self.gnn_depth)])
        elif params['gnn_type'] == 'GAT':
            self.gnn = nn.ModuleList([GATlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True, params['dropout']) for _ in range(self.gnn_depth)])
        elif params['gnn_type'] == 'GIN':
            self.gnn = nn.ModuleList([GINlayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True, params['dropout']) for _ in range(self.gnn_depth)])
        elif params['gnn_type'] == 'GIN_w_ef_update':
            self.gnn = nn.ModuleList([GINlayer_w_ef_update(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['layer_num'],
                                               params['activation'], True, params['dropout']) for _ in range(self.gnn_depth)])


    def forward(self, nf, ef, batch):
        edge_i = batch['edge_index']
        
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

        nh_label_one_hot = F.one_hot(batch['node_y'], num_classes=self.nf_in_dim).float().unsqueeze(-1) # (N, 9, 1)
        eh_label_one_hot = F.one_hot(batch['edge_y'], num_classes=self.ef_in_dim).float().unsqueeze(-1) # (E, 5, 1)

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



class DownstreamEncoder(nn.Module):
    def __init__(self, args, params):
        super(DownstreamEncoder, self).__init__()
        ##### 1) DICE
        self.dice = DICE(params['dice'], args.dice_depth)
        ##### 2) GNN added parallel to DICE
        self.parallel_gnn = Added_GNN(params['parallel_gnn'], args.p_gnn_depth)
        ##### 3) GNN added series to DICE
        self.series_gnn = Added_GNN(params['series_gnn'], args.s_gnn_depth)
        self.series_gnn.nf_in_dim = params['parallel_gnn']['nf_in_dim']
        self.series_gnn.ef_in_dim = params['parallel_gnn']['ef_in_dim']

    def forward(self, batch):
        ##### 1) DICE
        nf, ef, gf = self.dice(batch)
        ##### 2) GNN added parallel to DICE
        nf_p, ef_p, gf_p = self.parallel_gnn(batch['x'], batch['edge_attr'], batch)
        nf += nf_p; ef += ef_p; gf += gf_p
        ##### 3) GNN added series to DICE
        nf, ef, gf_s = self.series_gnn(nf, ef, batch)
        gf += gf_s
        return nf, ef, gf

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
############################################################################################################



############################################################################################################
# DECODER for Task 1 : Circuit Similarity Prediction
############################################################################################################
import itertools
class CircuitSimilarity_MLP(nn.Module):
    def __init__(self, params):
        super(CircuitSimilarity_MLP, self).__init__()
        self.layer_1 = build_layer(2*params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                    params['layer_num'], params['activation'], False, params['dropout'])
        self.layer_2 = build_layer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'],
                                    params['layer_num'], params['activation'], False, params['dropout'])
        self.classifier = build_layer(params['hidden_dim'], params['hidden_dim'], 3,
                                    params['layer_num'], params['activation'], False, params['dropout'])
        

    # input : gf
    # model input : N x 2 x gf_dim
    # model output : N x 3
    def forward(self, nf, ef, gf, batch):
        target_gf = gf[0].view(1, 1, -1)  # (1, 1, gf_dim)

        # all possible combinations of gf
        permutations = list(itertools.permutations(range(batch['batch'].max().item() + 1), 2))  # (N, 2)
        N = len(permutations)
        indices = torch.tensor(permutations).to(gf.device)

        gf_pairs = gf[indices,:].view(N, 2, -1)  # (N, 2, gf_dim)

        model_inputs = target_gf * gf_pairs      # (N, 2, gf_dim)
        model_inputs = model_inputs.view(N, -1)  # (N, 2*gf_dim)

        out = self.layer_1(model_inputs)         # (N, hidden_dim)
        out = self.layer_2(out)                  # (N, hidden_dim)
        out = self.classifier(out)               # (N, 3)

        return out


class CircuitClassificationModel(nn.Module):
    def __init__(self, params, task_name):
        super(CircuitClassificationModel, self).__init__()
        if task_name == "circuit_similarity_prediction":
            self.out_layer = CircuitSimilarity_MLP(params)
        else: raise ValueError(f"Unknown task: {task_name}")

    def forward(self, nf, ef, gf, batch):
        return self.out_layer(nf, ef, gf, batch)
############################################################################################################



############################################################################################################
# DECODER for Task 2, 3 : Delay-line Delay Prediction, OpAmp Metric Prediction
############################################################################################################
class DeviceParam_MLP(nn.Module):
    def __init__(self, params):
        super(DeviceParam_MLP, self).__init__()
        # Linear Layers
        self.nf_lin = nn.Sequential(nn.Linear(params['nf_in_dim']+9, params['hidden_dim']), nn.BatchNorm1d(params['hidden_dim']),
                                    nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin = nn.Sequential(nn.Linear(params['ef_in_dim'], params['hidden_dim']), nn.BatchNorm1d(params['hidden_dim']),
                                    nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['hidden_dim'], params['hidden_dim']))

    def forward(self, nf, ef, batch):
        # Multiply node features by device_params & concat
        nf = torch.cat([nf, F.one_hot(batch['node_y'], 9)\
                            * batch['device_params'].unsqueeze(-1)], dim=-1)
        nf, ef = self.nf_lin(nf), self.ef_lin(ef)
        
        return nf, ef

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))



class SimResultPredictionModel(nn.Module):
    def __init__(self, model_params, task_name):
        super(SimResultPredictionModel, self).__init__()
        self.init_layer = DeviceParam_MLP(model_params)
        if task_name == "delay_prediction":
            self.output_layer = build_layer(model_params['hidden_dim'], model_params['hidden_dim'], 2,
                                            model_params['layer_num'], model_params['activation'], True, model_params['dropout'])
        elif task_name == "opamp_metric_prediction":
            self.output_layer = build_layer(model_params['hidden_dim'], model_params['hidden_dim'], 5,
                                            model_params['layer_num'], model_params['activation'], True, model_params['dropout'])
        else: raise ValueError(f"Unknown task: {task_name}")

    def forward(self, nf, ef, gf, batch):

        nf, ef = self.init_layer(nf, ef, batch)

        gh = gf
        gh += scatter_add(nf, batch['batch'],
                          dim=0, dim_size=batch['batch'].max().item() + 1)
        gh += scatter_add(ef, batch['batch'][batch['edge_index'][0]],
                          dim=0, dim_size=batch['batch'][batch['edge_index'][0]].max().item() + 1)

        output = self.output_layer(gh)
        
        return output

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
############################################################################################################



############################################################################################################
# Downstream Model
############################################################################################################
class DownstreamModel(nn.Module):
    def __init__(self, args, model_params):
        super(DownstreamModel, self).__init__()
        ##### Encoder : forward should output (nf, ef, gf)
        self.encoder = DownstreamEncoder(args, model_params['encoder'])
        ##### Decoder : forward input should be (nf, ef, gf, batch)
        if args.task_name == "circuit_similarity_prediction":
            self.decoder = CircuitClassificationModel(model_params['decoder'], args.task_name)
        elif args.task_name == "delay_prediction" or args.task_name == "opamp_metric_prediction":
            self.decoder = SimResultPredictionModel(model_params['decoder'], args.task_name)
        else: raise ValueError(f"Unknown subtask: {args.task_name}")


    def forward(self, batch):
        nf, ef, gf = self.encoder(batch)
        output = self.decoder(nf, ef, gf, batch)
        return output

    def load_dice(self, path):
        self.encoder.dice.load(path)
        for param in self.encoder.dice.parameters(): param.requires_grad = False    # freeze DICE

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
############################################################################################################