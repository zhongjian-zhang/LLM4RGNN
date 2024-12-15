# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   process_output.py
@Time    :   2024/12/9 19:48
@Author  :   zhongjian zhang
"""
import argparse
import json
import os
import sys

import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj, coalesce
from tqdm import tqdm, trange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))
from util.load import load_data
from util.utils import get_path_to, csr_matrix_to_edge_index, save_jsonl


def get_args():
    parser = argparse.ArgumentParser(description='Creating the jsonl file for LLMs inference!')
    parser.add_argument('--llm', type=str, default="mistral-7b-merge")
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'])
    parser.add_argument('--attack', default='meta', type=str, choices=['meta', 'dice'])
    parser.add_argument('--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4])
    args = parser.parse_args()
    return args


def construct_single_infer():
    args = get_args()
    save_path = get_path_to("llm_response") / f"{args.llm}/all/{args.dataset}/"
    os.makedirs(save_path, exist_ok=True)
    inference_result_file = f'./output/{args.dataset}_{args.attack}_{args.ptb_rate}.jsonl'
    with open(inference_result_file, 'r', encoding='utf-8') as f:
        for index, line in tqdm(enumerate(f), desc="Processing inference results"):
            response_content = json.loads(line.strip())
            request_id = response_content['custom_id'].split("_")
            node1_id = request_id[0]
            node2_id = request_id[1]
            infer_content = response_content['response']['body']['choices'][0]['message']['content'].strip()
            file_name = f"{node1_id}_{node2_id}.json"
            file_path = f"{save_path}/{file_name}"
            if not os.path.exists(file_path):
                assert int(node1_id) < int(node2_id)
                json_data = {
                    "label": "",
                    "predict": infer_content
                }
                with open(file_path, 'w', encoding='utf-8') as f2:
                    json.dump(json_data, f2, ensure_ascii=False, indent=4)


def get_test_jsons(edge_index, request_path, text):
    json_list = []
    for index in trange(len(edge_index[0])):
        node1_id = edge_index[0][index].item()
        node2_id = edge_index[1][index].item()
        file_name = f"{node1_id}_{node2_id}.json"
        file_path = f"{request_path}/{file_name}"
        assert node1_id < node2_id
        assert os.path.exists(file_path)
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            json_list.append(json_data)
    return json_list


def construct_attack_infer():
    args = get_args()
    data = load_data(args.dataset)
    attack_path = get_path_to("saved_model")
    response_path = get_path_to("llm_response") / f"{args.llm}/"
    request_path = response_path / f"all/{args.dataset}"
    if args.ptb_rate > 0:
        attack_file = f"{attack_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        ptb_edge_index = torch.load(attack_file)
        save_response_path = f"{response_path}/global/{args.dataset}_{args.attack}_{args.ptb_rate}"
    else:
        ptb_edge_index = data.edge_index
        save_response_path = f"{response_path}/clean/{args.dataset}"

    os.makedirs(save_response_path, exist_ok=True)
    save_response_file = f"{save_response_path}/generated_predictions.jsonl"
    ptb_edge_index = to_dense_adj(ptb_edge_index.long())[0].triu_()
    ptb_edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(ptb_edge_index))))
    json_list = get_test_jsons(ptb_edge_index, request_path, data.text)
    print(ptb_edge_index.shape[1], len(json_list))
    save_jsonl(json_list, save_response_file)
    print(f"Save {save_response_file} successfully!")


if __name__ == '__main__':
    construct_single_infer()
    construct_attack_infer()
