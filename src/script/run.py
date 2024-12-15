# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Time    :   2024/4/3 16:33
@Author  :   zhongjian zhang
"""
import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))

import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import coalesce
from torch_geometric.utils import to_dense_adj
from model.GCN import GCN
from model.Trainer import Trainer
from util.load import load_data
from util.utils import setup_seed, get_path_to, csr_matrix_to_edge_index, edge_index_to_csr_matrix, accuracy, \
    sparse_mx_to_torch_sparse_tensor


def get_args():
    parser = argparse.ArgumentParser(description='Large Language Models for Robust Graph Neural Networks')
    # =============================== General Setting =====================================
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'])
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--llm', type=str, default="mistral-7b-merge")

    # =============================== Attack Setting ======================================
    parser.add_argument('--attack', default='meta', type=str, choices=['meta', 'dice'])
    parser.add_argument('--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4])

    # =============================== Edge Predictor Setting ==============================
    parser.add_argument('--sample_candidate_node', default=False, action='store_true')
    parser.add_argument('--candidate_node_num', type=int, default=2000)
    parser.add_argument('--edge_hidden_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--edge_lr', type=float, default=0.001)
    parser.add_argument('--edge_epochs', type=int, default=1000)
    parser.add_argument('--edge_early_stop', type=int, default=100)
    parser.add_argument('--edge_num_layer', type=int, default=4)
    parser.add_argument('--edge_weight_decay', type=float, default=5e-5)
    parser.add_argument('--edge_dropout', type=float, default=0.7)

    parser.add_argument('--positive_threshold', type=int, default=5, choices=[5, 6])
    parser.add_argument('--negative_threshold', type=int, default=4, choices=[4, 1])
    parser.add_argument('--purify_threshold', type=int, default=4, choices=[1, 2, 4])
    parser.add_argument('--confidence', type=float, default=0.91, choices=[0.91, 0.93, 0.95, 0.97, 0.99])
    parser.add_argument('--top_k', type=int, default=3, choices=[1, 3, 5, 7, 9])

    args = parser.parse_args()
    return args


def purify_graph():
    logging.info("=" * 35 + " Purifying Attacked Graph Topology " + "=" * 35)
    args = get_args()
    setup_seed(args.seed)
    logging.info("Arguments: %s", ", ".join([f"{k}={v}" for k, v in args.__dict__.items()]))
    data = load_data(dataset=args.dataset)
    model_path = get_path_to("saved_model")
    response_path = get_path_to("llm_response")
    if args.ptb_rate > 0:
        attack_file = f"{model_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        purify_file = f'{model_path}/purify/global/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.seed}_add_{str(args.top_k)}.pth'
        llm_response_file = f"{response_path}/{args.llm}/global/{args.dataset}_{args.attack}_{args.ptb_rate}/generated_predictions.jsonl"
        ptb_edge_index = torch.load(attack_file)
        logging.info("Load the attack from: %s", attack_file)
    else:
        purify_file = f'{model_path}/purify/clean/{args.dataset}_{args.seed}_add_{str(args.top_k)}.pth'
        llm_response_file = f"{response_path}/{args.llm}/clean/{args.dataset}/generated_predictions.jsonl"
        ptb_edge_index = data.edge_index
    if args.sample_candidate_node:
        candidate_node_file = f"{model_path}/candidate_node/{args.dataset}/{args.attack}_{args.ptb_rate}_{args.candidate_node_num}.json"
        logging.info("Load the candidate important nodes from: %s", candidate_node_file)
    else:
        candidate_node_file = None
    node_emb_file = f'{model_path}/node_emb/sbert/{args.dataset}.pth'
    negative_edge_file = f"{model_path}/negative_edge/{args.dataset}.pth"
    negative_llm_response = f"{response_path}/{args.llm}/negative/{args.dataset}/generated_predictions.jsonl"
    logging.info("Load the LLM prediction from: %s", llm_response_file)
    logging.info("Load the node embedding from: %s", node_emb_file)
    logging.info("Save the purified structure to: %s", purify_file)
    ptb_edge_index = to_dense_adj(ptb_edge_index)[0].triu_()
    ptb_edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(ptb_edge_index))))
    trainer = Trainer(data.text, ptb_edge_index, llm_response_file, node_emb_file, negative_edge_file,
                      negative_llm_response, candidate_node_file, purify_file, args)
    trainer.train()
    trainer.defense(sample_candidate_node=args.sample_candidate_node, candidate_node_num=args.candidate_node_num)


def test(data, args):
    device = args.device
    train_iters = 1000 if args.dataset in ["pubmed", "ogbn_arxiv", "ogbn_arxiv_full", "ogbn_product"] else 200
    n_hid = 256 if args.dataset in ["ogbn_arxiv", "ogbn_arxiv_full", "ogbn_product"] else 16
    gnn = GCN(nfeat=args.gnn_input_dim, nhid=n_hid, nclass=args.gnn_num_classes, device=device).to(device)
    gnn.fit(data.x, data.adj, data.y, data.train_id, data.val_id, verbose=False, train_iters=train_iters)
    output = gnn.output
    acc_test = accuracy(output[data.test_id], data.y[data.test_id])

    return acc_test


def gnn_perf_test():
    logging.info("=" * 35 + " Testing the Performance of GCN " + "=" * 35)
    args = get_args()
    logging.info("Arguments: %s", ", ".join([f"{k}={v}" for k, v in args.__dict__.items()]))
    data = load_data(dataset=args.dataset)
    setup_seed(args.seed)
    model_path = get_path_to("saved_model")
    if args.ptb_rate > 0:
        purify_file = f'{model_path}/purify/global/{args.dataset}_{args.attack}_{args.ptb_rate}_{args.seed}_add_{str(args.top_k)}.pth'
    else:
        purify_file = f'{model_path}/purify/clean/{args.dataset}_{args.seed}_add_{str(args.top_k)}.pth'
    vars(args)['gnn_input_dim'] = data.x.shape[1]
    vars(args)['gnn_num_classes'] = data.num_classes
    edge_index = torch.load(purify_file, map_location='cpu')
    logging.info("Load the purified structure from: %s", purify_file)
    logging.info("The edge index shape: %s", edge_index.shape)
    adj = edge_index_to_csr_matrix(edge_index, data.num_nodes)
    data.adj = sparse_mx_to_torch_sparse_tensor(adj)
    data.edge_index = edge_index
    data = data.to(args.device)
    acc = test(data, args)
    logging.info("Acc(↑): %.4f", acc)


def main():
    purify_graph()
    gnn_perf_test()


if __name__ == '__main__':
    main()
