import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_softmax

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import build_layer

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../pretrain/encoder'))
sys.path.append(parent_dir)

from model import DICE

############################################################################################################
# Baseline Graph Neural Networks
############################################################################################################


########################
######## DICE ##########
########################
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream_tasks'))
sys.path.append(parent_dir)
from downstream_model import DeviceParam_MLP, Added_GNN

class DICE_baselinecompare(nn.Module):
    def __init__(self, params):
        super(DICE_baselinecompare, self).__init__()
        self.dice = DICE(params['dice'], 2)
        self.p_gnn = Added_GNN(params['parallel_gnn'], 0)
        self.s_gnn = Added_GNN(params['series_gnn'], 2)

    def forward(self, batch):
        ##### 1) DICE
        nf, ef, gf = self.dice(batch)
        ##### 2) GNN added parallel to DICE
        nf_p, ef_p, gf_p = self.p_gnn(batch['x'], batch['edge_attr'], batch)
        nf += nf_p; ef += ef_p; gf += gf_p
        ##### 3) GNN added series to DICE
        nf, ef, gf_s = self.s_gnn(nf, ef, batch)
        gf += gf_s
        return nf, ef, gf



########################
###### ParaGraph #######
########################
class ParaGraphGNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bias):
        super(ParaGraphGNNLayer, self).__init__()
        self.nf_lin = nn.Linear(input_dim, hidden_dim, bias=False)
        self.attn = nn.Linear(2*hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.nf_out_lin = nn.Linear(2*hidden_dim, output_dim, bias=bias)
        self.relu = nn.ReLU()
    
    def forward(self, nh, edge_y, edge_index):
        src_node_i, dst_node_i = edge_index

        h = torch.zeros_like(nh)
        for i in range(5):
            edge_indices = torch.where(edge_y == i)[0]
            src_node_i, dst_node_i = edge_index[0][edge_indices], edge_index[1][edge_indices]
            src_nf, dst_nf = self.nf_lin(nh[src_node_i]), self.nf_lin(nh[dst_node_i])
            ef = self.attn(torch.cat([src_nf, dst_nf], dim=-1)).squeeze(-1)
            alpha = scatter_softmax(self.leaky_relu(ef), dst_node_i, dim=0).unsqueeze(-1)

            h += scatter_add(alpha * src_nf, dst_node_i, dim=0, dim_size=nh.size(0))

        nf = self.relu(self.nf_out_lin(torch.concat([nh, h], dim=-1)))

        return nf



class ParaGraph(nn.Module):
    def __init__(self, params):
        super(ParaGraph, self).__init__()
        self.gnn_depth = params['gnn_depth']
        self.hidden_dim = params['hidden_dim']

        self.init_node_lin = nn.Linear(params['nf_in_dim'], self.hidden_dim)

        self.gnn = nn.ModuleList([ParaGraphGNNLayer(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], True)\
                                            for _ in range(self.gnn_depth)])

        self.lin_out = nn.ModuleDict({
            'nf': nn.Sequential(nn.Linear(params['hidden_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['nf_out_dim'])),
            'gf': nn.Sequential(nn.Linear(params['nf_out_dim'], params['hidden_dim']),
                                nn.BatchNorm1d(params['hidden_dim']), nn.GELU(),
                                nn.Linear(params['hidden_dim'], params['gf_out_dim']))
        })


    def forward(self, batch):
        edge_i = batch['edge_index']
        edge_y = batch['edge_y']

        nf = self.init_node_lin(batch['x'])  # (N, nf_in_dim)
        ef = batch['edge_attr']              # (E, ef_in_dim)

        for i in range(self.gnn_depth):
            nf = self.gnn[i](nf, edge_y, edge_i)

        nf = self.lin_out['nf'](nf)          # (N, nf_out_dim)
        gh = scatter_add(nf, batch['batch'], dim=0, dim_size=batch['batch'].max().item() + 1)
        gf = self.lin_out['gf'](gh)

        return nf, ef, gf



########################
####### DeepGen ########
########################
class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GATLayer, self).__init__()
        self.linear = nn.Linear(2*input_dim, output_dim, bias=False)
        self.attn = nn.Linear(2*output_dim, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, nf, edge_index):
        src_node_i, dst_node_i = edge_index[0], edge_index[1]

        src_nf, dst_nf = nf[src_node_i], nf[dst_node_i]
        ef = self.attn(torch.concat([src_nf, dst_nf], dim=-1)).squeeze(-1)
        alpha = scatter_softmax(ef, dst_node_i, dim=0).unsqueeze(-1)

        msg = alpha * src_nf
        agg_nf = scatter_add(msg, dst_node_i, dim=0, dim_size=nf.size(0))

        nh = self.relu(self.linear(torch.concat([nf, agg_nf], dim=-1)))

        return nh


class GNNBackbone(nn.Module):
    def __init__(self, params):
        super(GNNBackbone, self).__init__()
        self.gnn_depth = params['gnn_depth']
        self.hidden_dim = params['hidden_dim']

        self.init_node_lin = nn.Linear(params['nf_in_dim'], self.hidden_dim)
        self.gnn = nn.ModuleList([GATLayer(self.hidden_dim, self.hidden_dim)\
                                            for _ in range(self.gnn_depth)])

    def forward(self, nf, edge_index):
        nf = self.init_node_lin(nf)
        for i in range(self.gnn_depth):
            nf = self.gnn[i](nf, edge_index)

        return nf



class DCvoltagePrediction(nn.Module):
    def __init__(self, args, params):
        super(DCvoltagePrediction, self).__init__()
        self.hidden_dim = params['hidden_dim']

        self.gnn_backbone = GNNBackbone(params)
        self.lin_out = nn.Sequential(
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                            nn.Linear(self.hidden_dim, params['nf_out_dim'])
                        )

    def forward(self, batch):
        edge_i = batch['edge_index']

        nf = self.gnn_backbone(batch['x'], edge_i)
        nf = self.lin_out(nf)

        return nf


    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save_gnn_backbone(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.gnn_backbone.state_dict(), path)




class DeepGen_u(nn.Module):
    def __init__(self, params):
        super(DeepGen_u, self).__init__()
        self.gnn_depth = params['gnn_depth']
        self.hidden_dim = params['hidden_dim']

        self.init_node_lin = nn.Linear(params['nf_in_dim'], self.hidden_dim)

        self.gnn = nn.ModuleList([GATLayer(self.hidden_dim, self.hidden_dim)\
                                            for _ in range(self.gnn_depth)])

        self.lin_out = nn.ModuleDict({
            'nf': nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                                nn.Linear(self.hidden_dim, params['nf_out_dim'])),
            'gf': nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                                nn.Linear(self.hidden_dim, params['gf_out_dim']))
        })

    def forward(self, batch):
        edge_i = batch['edge_index']
        src_node_i, dst_node_i = edge_i[0], edge_i[1]

        nf = self.init_node_lin(batch['x'])
        ef = batch['edge_attr']

        for i in range(self.gnn_depth):
            nf = self.gnn[i](nf, edge_i)

        h = nf
        for i in range(3):
            alpha = scatter_softmax(torch.sum(h*nf, dim=-1), batch['batch'], dim=0).unsqueeze(-1)
            h = alpha * nf
        gf = scatter_add(h, batch['batch'], dim=0, dim_size=batch['batch'].max().item() + 1)

        nf = self.lin_out['nf'](nf)
        gf = self.lin_out['gf'](gf)

        return nf, ef, gf



class DeepGen_p(nn.Module):
    def __init__(self, params):
        super(DeepGen_p, self).__init__()
        self.gnn_depth = params['gnn_depth']
        self.hidden_dim = params['hidden_dim']

        self.gnn_backbone = GNNBackbone(params)
        
        self.lin_out = nn.ModuleDict({
            'nf': nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                                nn.Linear(self.hidden_dim, params['nf_out_dim'])),
            'gf': nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.BatchNorm1d(self.hidden_dim), nn.ReLU(),
                                nn.Linear(self.hidden_dim, params['gf_out_dim']))
        })

    def forward(self, batch):
        edge_i = batch['edge_index']

        nf = batch['x']
        ef = batch['edge_attr']

        nf = self.gnn_backbone(nf, edge_i)
        
        h = nf
        for i in range(3):
            alpha = scatter_softmax(torch.sum(h*nf, dim=-1), batch['batch'], dim=0).unsqueeze(-1)
            h = alpha * nf
        gf = scatter_add(h, batch['batch'], dim=0, dim_size=batch['batch'].max().item() + 1)

        nf = self.lin_out['nf'](nf)
        gf = self.lin_out['gf'](gf)

        return nf, ef, gf
############################################################################################################




############################################################################################################
# ENCODER
############################################################################################################
class BaselineEncoder(nn.Module):
    def __init__(self, args, params):
        super(BaselineEncoder, self).__init__()
        ##### Baseline GNN
        if args.baseline_name == "DICE":
            self.encoder = DICE_baselinecompare(params[args.baseline_name])
        elif args.baseline_name == "ParaGraph":
            self.encoder = ParaGraph(params['ParaGraph'])
        elif args.baseline_name == "DeepGen_u":
            self.encoder = DeepGen_u(params['DeepGen_u'])
        elif args.baseline_name == "DeepGen_p":
            self.encoder = DeepGen_p(params['DeepGen_p'])


    def forward(self, batch):
        nf, ef, gf = self.encoder(batch)
        return nf, ef, gf
############################################################################################################




############################################################################################################
# DECODER
############################################################################################################


###########################
######### Task 1 ##########
###########################
# DECODER for Task 1 : Circuit Similarity Prediction
## Same with this work's model (downstream_model.py)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream_tasks'))
sys.path.append(parent_dir)
from downstream_model import CircuitClassificationModel


###########################
######## Task 2, 3 ########
###########################
# DECODER for Task 2, 3 : Delay Prediction, Opamp Metric Prediction
## Differs slightly depending on the graph structure

########################
class DeviceParam_MLP_ParaGraph(nn.Module):
    def __init__(self, params):
        super(DeviceParam_MLP_ParaGraph, self).__init__()
        # Linear Layers
        self.nf_lin = nn.Sequential(nn.Linear(params['nf_in_dim']+7, params['hidden_dim']), nn.BatchNorm1d(params['hidden_dim']),
                                    nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin = nn.Sequential(nn.Linear(params['ef_in_dim'], params['hidden_dim']), nn.BatchNorm1d(params['hidden_dim']),
                                    nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['hidden_dim'], params['hidden_dim']))

    def forward(self, nf, ef, batch):
        # Multiply edge features by device_params & concat
        nf = torch.cat([nf, F.one_hot(batch['node_y'], 7)\
                            * batch['device_params'].unsqueeze(-1)], dim=-1)
        nf, ef = self.nf_lin(nf), self.ef_lin(ef)
        
        return nf, ef



class DeviceParam_MLP_DeepGen(nn.Module):
    def __init__(self, params):
        super(DeviceParam_MLP_DeepGen, self).__init__()
        # Linear Layers
        self.nf_lin = nn.Sequential(nn.Linear(params['nf_in_dim']+15, params['hidden_dim']), nn.BatchNorm1d(params['hidden_dim']),
                                    nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['hidden_dim'], params['hidden_dim']))
        self.ef_lin = nn.Sequential(nn.Linear(params['ef_in_dim'], params['hidden_dim']), nn.BatchNorm1d(params['hidden_dim']),
                                    nn.GELU(), nn.Dropout(params['dropout']), nn.Linear(params['hidden_dim'], params['hidden_dim']))

    def forward(self, nf, ef, batch):
        # Multiply edge features by device_params & concat
        nf = torch.cat([nf, F.one_hot(batch['node_y'], 15)\
                            * batch['device_params'].unsqueeze(-1)], dim=-1)
        nf, ef = self.nf_lin(nf), self.ef_lin(ef)
        
        return nf, ef
########################



class SimResultPredictionModel(nn.Module):
    def __init__(self, model_params, task_name, baseline_name):
        super(SimResultPredictionModel, self).__init__()
        if baseline_name == "DICE":
            param = model_params['DICE']
            self.init_layer = DeviceParam_MLP(param)
        elif baseline_name == "ParaGraph":
            param = model_params['ParaGraph']
            self.init_layer = DeviceParam_MLP_ParaGraph(param)
        elif baseline_name == "DeepGen_u":
            param = model_params['DeepGen_u']
            self.init_layer = DeviceParam_MLP_DeepGen(param)
        elif baseline_name == "DeepGen_p":
            param = model_params['DeepGen_p']
            self.init_layer = DeviceParam_MLP_DeepGen(param)
        else: raise ValueError(f"Unknown baseline: {baseline_name}")

        if task_name == "delay_prediction":
            self.output_layer = build_layer(param['hidden_dim'], param['hidden_dim'], 2,
                                            param['layer_num'], param['activation'], True, param['dropout'])
        elif task_name == "opamp_metric_prediction":
            self.output_layer = build_layer(param['hidden_dim'], param['hidden_dim'], 5,
                                            param['layer_num'], param['activation'], True, param['dropout'])
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
############################################################################################################



############################################################################################################
# Baseline Model
############################################################################################################
class BaselineModel(nn.Module):
    def __init__(self, args, model_params):
        super(BaselineModel, self).__init__()
        ##### Encoder : forward should output (nf, ef, gf)
        self.encoder = BaselineEncoder(args, model_params['encoder'])
        ##### Decoder : forward input should be (nf, ef, gf, batch)
        if args.task_name == "circuit_similarity_prediction":
            self.decoder = CircuitClassificationModel(model_params['decoder'][args.baseline_name], args.task_name)
        elif args.task_name == "delay_prediction" or args.task_name == "opamp_metric_prediction":
            self.decoder = SimResultPredictionModel(model_params['decoder'], args.task_name, args.baseline_name)
        else: raise ValueError(f"Unknown subtask: {args.task_name}")

    def forward(self, batch):
        nf, ef, gf = self.encoder(batch)
        output = self.decoder(nf, ef, gf, batch)
        
        return output

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
############################################################################################################