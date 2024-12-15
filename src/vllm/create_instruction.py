# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   create_instruction.py
@Time    :   2024/9/30 14:28
@Author  :   zhongjian zhang
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj, coalesce
from tqdm import trange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))
from util.load import load_data
from util.utils import get_path_to, csr_matrix_to_edge_index, save_jsonl

system_content = """In the context of graph neural networks, attackers manipulate models by adding irrelevant edges or removing relevant ones, leading to incorrect predictions. Your role is crucial in defending against such attacks by evaluating the relevance between pairs of nodes, which will help in identifying and removing the irrelevant edges to mitigate the impact of adversarial attacks on graph-based models. Given textual information about two nodes, analyze the relevance of these two nodes. Provide a concise analysis(approximately 100 words) and assign an integer relevance score from 1 to 6, where 1 indicates completely irrelevant and 6 indicates directly relevant. Your response should be formatted in JSON, with two keys: "Analysis" for your written analysis and "Relevance Score" for your numerical evaluation."""


def get_args():
    parser = argparse.ArgumentParser(description='Creating the jsonl file for LLMs inference!')
    parser.add_argument('--llm', type=str, default="mistral-7b-merge")
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'])
    parser.add_argument('--attack', default='meta', type=str, choices=['meta', 'dice'])
    parser.add_argument('--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4])
    args = parser.parse_args()
    return args


def complete_test_instruction(edge_index, text, llm_path, request_path):
    complete_json_list = []
    complete_edge_index = []
    for index in trange(len(edge_index[0])):
        node1_id = edge_index[0][index].item()
        node2_id = edge_index[1][index].item()
        assert node1_id < node2_id
        file_name = f"{node1_id}_{node2_id}.json"
        file_path = f"{request_path}/{file_name}"
        if not os.path.exists(file_path):
            node1_text, node2_text = text[node1_id], text[node2_id]
            user_content = f"Node1 -> {node1_text}\n\nNode2 -> {node2_text}"
            json_data = {
                "custom_id": f"{node1_id}_{node2_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": llm_path,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    "max_tokens": 1000
                }
            }
            complete_json_list.append(json_data)
            complete_edge_index.append([node1_id, node2_id])
    return np.array(complete_edge_index).T, complete_json_list


def main():
    args = get_args()
    path = get_path_to("saved_model")
    data = load_data(args.dataset)
    text = data.text
    if args.ptb_rate > 0:
        attack_file = f"{path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        edge_index = torch.load(attack_file)
    else:
        edge_index = data.edge_index
    edge_index = to_dense_adj(edge_index)[0].triu_()
    edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(edge_index))))
    print(edge_index.shape)
    request_path = get_path_to("llm_response") / f"{{args.llm}}/all/{args.dataset}"
    llm_path = str(get_path_to("saved_model") / f"llm/{args.llm}")
    complete_edge_index, complete_instruction_list = complete_test_instruction(edge_index, text, llm_path, request_path)
    print(complete_edge_index.shape)
    file_path = f'./instruction/{args.dataset}_{args.attack}_{args.ptb_rate}.jsonl'
    save_jsonl(complete_instruction_list, file_path)


if __name__ == '__main__':
    main()
