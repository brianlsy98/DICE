import os
import sys
import json
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from torch.amp import autocast

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)

from utils import send_to_device, init_weights, set_seed
from dataloader import GraphDataLoader
from model import DICE


def get_init_gf(batch):

    init_batch = send_to_device(batch, 'cuda')
    nf, ef, edge_index = init_batch['x'], init_batch['edge_attr'], init_batch['edge_index']
    nf, ef = F.pad(nf, (0, 5)), F.pad(ef, (0, 9))

    gf = scatter_add(nf, init_batch['batch'], dim=0,
                        dim_size=init_batch['batch'].max().item() + 1) \
        + scatter_add(ef, init_batch['batch'][edge_index[0]], dim=0,
                        dim_size=init_batch['batch'][edge_index[0]].max().item() + 1)

    return gf



def get_cosine_similarities(gf, gf_labels):

    graph_labels = gf_labels

                    # (same label & both pos label)
    positive_mask =   (graph_labels.unsqueeze(1) == graph_labels.unsqueeze(0))\
                    & ((graph_labels > 0).unsqueeze(1) & (graph_labels > 0).unsqueeze(0))

                    # negative data augementation relation
    negative_mask = (graph_labels.unsqueeze(1) == -graph_labels.unsqueeze(0))

    gf_n = F.normalize(gf, dim=-1)
    gf_cosine_similarities = torch.mm(gf_n, gf_n.t())

    return torch.tensor(gf_cosine_similarities[positive_mask].tolist()),\
           torch.tensor(gf_cosine_similarities[~positive_mask & ~negative_mask].tolist()),\
           torch.tensor(gf_cosine_similarities[negative_mask].tolist())



def test_model(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ### Dataset
    test_dataset = torch.load(f'./pretrain/dataset/{args.dataset_name}_test.pt')
    print()
    print("Dataset Loaded")
    print()

    with open('./params.json', 'r') as f:
        params = json.load(f)


    ########################
    # untrained model
    untrained_model = DICE(params['model']['encoder']['dice'], gnn_depth=args.gnn_depth)
    untrained_model.apply(init_weights)
    # trained model with only positive data augmentation
    trained_model_pda = DICE(params['model']['encoder']['dice'], gnn_depth=args.gnn_depth)
    trained_model_pda.load(f"./pretrain/encoder/DICE_pretrained_model_GIN_depth2_taup02tau005taun005.pt")
    # trained model with neg data augmentation
    trained_model = DICE(params['model']['encoder']['dice'], gnn_depth=args.gnn_depth)
    trained_model.load(f"./pretrain/encoder/DICE_pretrained_model_GIN_depth2_taup02tau005taun005.pt")
    ########################
    untrained_model.to('cuda'); trained_model_pda.to('cuda'); trained_model.to('cuda')
    print()
    print("Model Initialized")
    print()


    ### Dataloader
    # test dataloader
    test_data = []
    for circuit_name, pos_neg_test_data in test_dataset.items():
        test_data += pos_neg_test_data['pos']
        test_data += pos_neg_test_data['neg']
    random.shuffle(test_data)

    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=512,
        shuffle=True
    )

    print()
    print("Dataloader Initialized")
    print("test_data:", len(test_data))
    print()


    ### Test
    untrained_model.eval(); trained_model_pda.eval(); trained_model.eval()
    init_cs_values_p, untrained_cs_values_p, trained_w_cs_values_pda_p, trained_w_cs_values_p = [], [], [], []
    init_cs_values_np, untrained_cs_values_np, trained_w_cs_values_pda_np, trained_w_cs_values_np = [], [], [], []
    init_cs_values_n, untrained_cs_values_n, trained_w_cs_values_pda_n, trained_w_cs_values_n = [], [], [], []

    for s in [99, 88, 77, 66, 55]:
        print(f"testing on seed {s}...")
        set_seed(s)
        with torch.no_grad():
            for test_batch in test_dataloader:
                test_batch = send_to_device(test_batch, 'cuda')
                ############################
                with autocast('cuda'):
                    gf_i = get_init_gf(test_batch)
                    _, _, gf_u = untrained_model(test_batch)
                    _, _, gf_pda = trained_model_pda(test_batch)
                    _, _, gf_t = trained_model(test_batch)
                    init_cs_p, init_cs_np, init_cs_n = get_cosine_similarities(gf_i, test_batch['circuit'])
                    untrained_cs_p, untrained_cs_np, untrained_cs_n = get_cosine_similarities(gf_u, test_batch['circuit'])
                    trained_w_cs_pda_p, trained_w_cs_pda_np, trained_w_cs_pda_n = get_cosine_similarities(gf_pda, test_batch['circuit'])
                    trained_w_cs_p, trained_w_cs_np, trained_w_cs_n = get_cosine_similarities(gf_t, test_batch['circuit'])
                    ############################
                init_cs_values_p.append(init_cs_p)
                untrained_cs_values_p.append(untrained_cs_p)
                trained_w_cs_values_pda_p.append(trained_w_cs_pda_p)
                trained_w_cs_values_p.append(trained_w_cs_p)
                init_cs_values_np.append(init_cs_np)
                untrained_cs_values_np.append(untrained_cs_np)
                trained_w_cs_values_pda_np.append(trained_w_cs_pda_np)
                trained_w_cs_values_np.append(trained_w_cs_np)
                init_cs_values_n.append(init_cs_n)
                untrained_cs_values_n.append(untrained_cs_n)
                trained_w_cs_values_pda_n.append(trained_w_cs_pda_n)
                trained_w_cs_values_n.append(trained_w_cs_n)
    
    init_cs_values_p, init_cs_values_np, init_cs_values_n = torch.cat(init_cs_values_p, dim=0), torch.cat(init_cs_values_np, dim=0), torch.cat(init_cs_values_n, dim=0)
    untrained_cs_values_p, untrained_cs_values_np, untrained_cs_values_n = torch.cat(untrained_cs_values_p, dim=0), torch.cat(untrained_cs_values_np, dim=0), torch.cat(untrained_cs_values_n, dim=0)
    trained_w_cs_values_pda_p, trained_w_cs_values_pda_np, trained_w_cs_values_pda_n = torch.cat(trained_w_cs_values_pda_p, dim=0), torch.cat(trained_w_cs_values_pda_np, dim=0), torch.cat(trained_w_cs_values_pda_n, dim=0)
    trained_w_cs_values_p, trained_w_cs_values_np, trained_w_cs_values_n = torch.cat(trained_w_cs_values_p, dim=0), torch.cat(trained_w_cs_values_np, dim=0), torch.cat(trained_w_cs_values_n, dim=0)
    
    print()
    print("Cosine Similarities Calculated")
    print()
    print("Positive Pairs")
    print("Init:", init_cs_values_p.mean().item(), init_cs_values_p.std().item())
    print("U   :", untrained_cs_values_p.mean().item(), untrained_cs_values_p.std().item())
    # print("Tpda:", trained_w_cs_values_pda_p.mean().item(), trained_w_cs_values_pda_p.std().item())
    print("T   :", trained_w_cs_values_p.mean().item(), trained_w_cs_values_p.std().item())
    print()
    print("Non-Positive Pairs")
    print("Init:", init_cs_values_np.mean().item(), init_cs_values_np.std().item())
    print("U   :", untrained_cs_values_np.mean().item(), untrained_cs_values_np.std().item())
    # print("Tpda:", trained_w_cs_values_pda_np.mean().item(), trained_w_cs_values_pda_np.std().item())
    print("T   :", trained_w_cs_values_np.mean().item(), trained_w_cs_values_np.std().item())
    print()
    print("Negative Pairs")
    print("Init:", init_cs_values_n.mean().item(), init_cs_values_n.std().item())
    print("U   :", untrained_cs_values_n.mean().item(), untrained_cs_values_n.std().item())
    # print("Tpda:", trained_w_cs_values_pda_n.mean().item(), trained_w_cs_values_pda_n.std().item())
    print("T   :", trained_w_cs_values_n.mean().item(), trained_w_cs_values_n.std().item())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICE Pre-Training.")
    parser.add_argument(
        "--dataset_name",
        default="pretraining_dataset_wo_device_params",
        type=str,
        help="Name of the dataset directory"
    )
    parser.add_argument("--gnn_depth", default=2, type=int, help="Depth of GNN")
    parser.add_argument("--seed", default=0, type=int, help="Seed for reproducibility")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    args = parser.parse_args()
    test_model(args)
