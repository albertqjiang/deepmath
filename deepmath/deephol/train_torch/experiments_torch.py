"""PyTorch Implementation of Holparam experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import torch
import torch.nn as nn
from datetime import datetime
from torch_geometric.data import DataLoader as GeoLoader
from deepmath.deephol.train_torch import data_torch
from deepmath.deephol.train_torch.model_torch import GNN
from deepmath.deephol.train_torch.utils_torch import accuracy
import tensorflow as tf
tf.enable_eager_execution()
timestamp = str(datetime.now()).split(".")[:-1][0].replace("-", "_").replace(" ", "_").replace(":", "_")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-dd", "--dataset_dir",
                    required=False, default="/u/ajiang/deephol-data/deepmath/deephol/proofs/human/",
                    help="directory where all the proof data live in")
    ap.add_argument("-md", "--model_dir",
                    required=False, default="/u/ajiang/Projects/ineqSolver/Inequality/pt_models/holist/",
                    help="directory where the model should be stored")
    ap.add_argument("-gv", "--goal_vocab",
                    required=False, default="vocab_goal_ls.txt",
                    help="goal vocabulary")
    ap.add_argument("-tv", "--thm_vocab",
                    required=False, default="vocab_thms_ls.txt",
                    help="theorem vocabulary")
    ap.add_argument("-mv", "--missing_vocab",
                    required=False, default="missing_vocab.txt",
                    help="missing vocabulary")
    ap.add_argument("-bs", "--batch_size",
                    required=False, default=32,
                    help="batch size to use")
    ap.add_argument("-sd", "--state_dimension",
                    required=False, default=16,
                    help="state dimension")
    ap.add_argument("-gt", "--gnn_type",
                    required=False, default="GIN",
                    help="graph neural network type to use")
    ap.add_argument("-hl", "--hidden_layers",
                    required=False, default=10,
                    help="graph neural network number of hidden layers")
    ap.add_argument("-cuda", "--cuda",
                    required=False, default=False,
                    help="whether to use cuda")
    params = vars(ap.parse_args())

    batch_size = params["batch_size"]
    dataset_dir = params["dataset_dir"]
    model_dir = params["model_dir"]
    cuda = params["cuda"]
    device = torch.device("cuda") if cuda else torch.device("cpu")
    if not os.path.isdir(model_dir + timestamp):
        os.mkdir(model_dir + timestamp)
    json.dump(params, open(model_dir + timestamp + "/config.json", "w"))

    train_set = data_torch.TacticDataset(
        dataset_dir + "/train",
        "tfexample",
        params
    )
    val_set = data_torch.TacticDataset(
        dataset_dir + "/valid",
        "tfexample",
        params
    )
    val_batch_size = 100
    val_batch_sampler = data_torch.get_directory_batch_sampler(val_set, val_batch_size)

    net = GNN(params)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    losses = {
        "training": {
            "losses": [],
            "acc_top1": [],
            "acc_top5": []
        },
        "validation": {
            "losses": [],
            "acc_top1": [],
            "acc_top5": [],
        }
    }
    ce = nn.CrossEntropyLoss()

    batch_sampler = data_torch.get_directory_batch_sampler(train_set, batch_size)

    from pprint import pprint as pp
    update = 0
    for batch in batch_sampler:
        net.train()
        batch_data = train_set.get_multiple(batch)
        goals = list(map(lambda x: x[0]['goal_ids'], batch_data))
        tactics = list(map(lambda x: x[1]['tac_id'], batch_data))
        geo_loader = GeoLoader(goals, batch_size=batch_size)
        tactic_tensor = torch.cat(tactics, 0)
        # print(tactic_tensor)
        for graphs in geo_loader:
            graphs.to(device)
            tactic_tensor.to(device)
            update += 1
            optimizer.zero_grad()
            out = net(graphs)
            loss = ce(out, tactic_tensor)
            loss.backward()
            optimizer.step()

        if update % 10 == 1:
            print("*"* 100)
            print("Train loss", loss.cpu().item())
            losses["training"]["losses"].append(loss.cpu().item())
            top1 = accuracy(out, tactic_tensor)[0].cpu().item()
            top5 = accuracy(out, tactic_tensor, topk=(5,))[0].cpu().item()
            print("Training Top 1 accuracy:", top1)
            print("Training Top 5 accuracy:", top5)
            losses["training"]["acc_top1"].append(top1)
            losses["training"]["acc_top5"].append(top5)
        if update % 100 == 1:
            # Validation
            net.eval()
            for val_batch in val_batch_sampler:
                val_batch_data = val_set.get_multiple(val_batch)
                val_goals = list(map(lambda x: x[0]['goal_ids'], val_batch_data))
                val_tactics = list(map(lambda x: x[1]['tac_id'], val_batch_data))
                val_geo_loader = GeoLoader(val_goals, batch_size=val_batch_size)
                val_tactic_tensor = torch.cat(val_tactics, 0)
                for val_graphs in val_geo_loader:
                    print("*" * 100)
                    val_graphs.to(device)
                    val_tactic_tensor.to(device)
                    val_out = net(val_graphs)
                    loss = ce(val_out, val_tactic_tensor).cpu().item()
                    print("Validation loss", loss)
                    losses["validation"]["losses"].append(loss)
                    top1 = accuracy(val_out, val_tactic_tensor)[0].cpu().item()
                    top5 = accuracy(val_out, val_tactic_tensor, topk=(5,))[0].cpu().item()
                    print("Validation Top 1 accuracy:", top1)
                    print("Validation Top 5 accuracy:", top5)
                    losses["validation"]["acc_top1"].append(top1)
                    losses["validation"]["acc_top5"].append(top5)
                    break
                break

            # pp(losses)
            json.dump(losses, open(model_dir + timestamp + "/loss.json", "w"))
