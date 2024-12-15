# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   create_instruction.py
@Time    :   2024/3/29 14:46
@Author  :   zhongjian zhang
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch.nn.functional import cosine_similarity
from torch_geometric.utils import coalesce, to_dense_adj
from tqdm import trange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))
from util.load import load_data
from util.utils import get_path_to, csr_matrix_to_edge_index, text_to_embedding, save_jsonl

instruction = """In the context of graph neural networks, attackers manipulate models by adding irrelevant edges or removing relevant ones, leading to incorrect predictions. Your role is crucial in defending against such attacks by evaluating the relevance between pairs of nodes, which will help in identifying and removing the irrelevant edges to mitigate the impact of adversarial attacks on graph-based models. Given textual information about two nodes, analyze the relevance of these two nodes. Provide a concise analysis(approximately 100 words) and assign an integer relevance score from 1 to 6, where 1 indicates completely irrelevant and 6 indicates directly relevant. Your response should be formatted in JSON, with two keys: "Analysis" for your written analysis and "Relevance Score" for your numerical evaluation."""


def get_args():
    parser = argparse.ArgumentParser(description='Creating the jsonl file for LLMs inference!')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'])
    parser.add_argument('--attack', default='meta', type=str, choices=['meta', 'dice'])
    parser.add_argument('--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4])
    args = parser.parse_args()
    return args


def get_test_jsons(edge_index, text):
    json_list = []
    for index in trange(len(edge_index[0])):
        node1 = edge_index[0][index]
        node2 = edge_index[1][index]
        node1_text, node2_text = text[node1], text[node2]
        input_data = f"Node1 -> {node1_text}\n\nNode2 -> {node2_text}"
        json_data = {
            "instruction": instruction,
            "input": input_data,
            "output": ""
        }
        json_list.append(json_data)
    return json_list


def crate_test_instruction():
    args = get_args()
    data = load_data(args.dataset)
    text = data.text
    model_path = get_path_to("saved_model")
    if args.ptb_rate > 0:
        attack_file = f"{model_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        jsonl_path = f"../LLaMA-Factory/data/{args.dataset}_{args.attack}_{args.ptb_rate}.jsonl"
        ptb_edge_index = torch.load(attack_file)
    else:
        ptb_edge_index = data.edge_index
        jsonl_path = f"../LLaMA-Factory/data/{args.dataset}.jsonl"
    ptb_edge_index = to_dense_adj(ptb_edge_index)[0].triu_()
    ptb_edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(ptb_edge_index))))
    json_list = get_test_jsons(ptb_edge_index, text)
    save_jsonl(json_list, jsonl_path)


def create_aug_negative_sample():
    args = get_args()
    data = load_data(args.dataset)
    embeddings = text_to_embedding(data.text, device=0)
    n = embeddings.shape[0]
    similarity_matrix = cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)
    eye = torch.eye(n) > 0.5
    similarity_matrix[eye] = 100000
    k = min(4000, n * (n - 1) // 2)
    values, indices = torch.topk(similarity_matrix.view(-1), k, largest=False)
    rows = indices // n
    cols = indices % n
    negative_edge_index = []
    for i in range(k):
        negative_edge_index.append([rows[i].item(), cols[i].item()])
        print(f"Pair {i + 1}: Node {rows[i].item()} and Node {cols[i].item()}, Similarity: {values[i].item()}")
    negative_edge_index = torch.tensor(np.array(negative_edge_index).transpose())
    print(negative_edge_index)
    model_path = get_path_to("saved_model")
    json_list = get_test_jsons(negative_edge_index, data.text)
    jsonl_path = f"../LLaMA-Factory/data/negative_{args.dataset}.jsonl"
    edge_path = f"{model_path}/negative_edge/{args.dataset}.pth"
    torch.save(negative_edge_index, edge_path)
    save_jsonl(json_list, jsonl_path)


if __name__ == '__main__':
    crate_test_instruction()
    create_aug_negative_sample()
