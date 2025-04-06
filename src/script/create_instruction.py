#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   create_instruction.py
@Time    :   2024/3/29 14:46
@Author  :   zhongjian zhang
@Description: Script for creating instruction datasets for LLM inference.
              Generates data for evaluating node relevance in graphs.
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

# Add parent directory to path to correctly import util module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from util.load import load_data
from util.utils import get_path_to, csr_matrix_to_edge_index, text_to_embedding, save_jsonl

# Instruction template for the LLM task
NODE_RELEVANCE_INSTRUCTION = """In the context of graph neural networks, attackers manipulate models by adding irrelevant edges or removing relevant ones, leading to incorrect predictions. Your role is crucial in defending against such attacks by evaluating the relevance between pairs of nodes, which will help in identifying and removing the irrelevant edges to mitigate the impact of adversarial attacks on graph-based models. Given textual information about two nodes, analyze the relevance of these two nodes. Provide a concise analysis(approximately 100 words) and assign an integer relevance score from 1 to 6, where 1 indicates completely irrelevant and 6 indicates directly relevant. Your response should be formatted in JSON, with two keys: "Analysis" for your written analysis and "Relevance Score" for your numerical evaluation."""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create JSONL files for LLM inference on node relevance')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed',
                                 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'],
                        help='Dataset to use')
    parser.add_argument('--attack', default='meta', type=str, choices=['meta', 'dice'],
                        help='Attack method')
    parser.add_argument('--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4],
                        help='Perturbation rate for the attack')
    return parser.parse_args()


def create_node_pair_json(edge_index, text):
    """
    Create JSON objects for each node pair in the edge index.

    Args:
        edge_index: Tensor containing source and target node indices
        text: Text features for each node

    Returns:
        List of JSON objects formatted for LLM inference
    """
    json_list = []
    for index in trange(len(edge_index[0]), desc="Creating JSON samples"):
        node1_idx = edge_index[0][index]
        node2_idx = edge_index[1][index]
        node1_text, node2_text = text[node1_idx], text[node2_idx]

        input_data = f"Node1 -> {node1_text}\n\nNode2 -> {node2_text}"
        json_data = {
            "instruction": NODE_RELEVANCE_INSTRUCTION,
            "input": input_data,
            "output": ""
        }
        json_list.append(json_data)
    return json_list


def ensure_dir_exists(file_path):
    """Create directory if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def create_test_instruction(args):
    """
    Create test instruction dataset from original or perturbed graph.
    Uses existing attack data if perturbation rate > 0.
    """
    data = load_data(args.dataset)
    text = data.text
    model_path = get_path_to("saved_model")

    # Handle perturbed vs. original graph
    if args.ptb_rate > 0:
        attack_file = f"{model_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        jsonl_path = os.path.join(parent_dir, "LLaMA-Factory", "data",
                                  f"{args.dataset}_{args.attack}_{args.ptb_rate}.jsonl")
        ptb_edge_index = torch.load(attack_file)
    else:
        ptb_edge_index = data.edge_index
        jsonl_path = os.path.join(
            parent_dir, "LLaMA-Factory", "data", f"{args.dataset}.jsonl")

    # Ensure output directory exists
    ensure_dir_exists(jsonl_path)

    # Process edge index to ensure upper triangular form and coalesced
    ptb_edge_index = to_dense_adj(ptb_edge_index)[0].triu_()
    ptb_edge_index = coalesce(torch.tensor(
        csr_matrix_to_edge_index(csr_matrix(ptb_edge_index))))

    # Create and save JSON samples
    json_list = create_node_pair_json(ptb_edge_index, text)
    save_jsonl(json_list, jsonl_path)
    print(f"Test instruction dataset saved to {jsonl_path}")


def create_negative_samples(args):
    """
    Create negative samples dataset based on low cosine similarity between node embeddings.
    These represent node pairs that are likely irrelevant to each other.
    """
    data = load_data(args.dataset)

    # Get text embeddings and calculate similarity matrix
    print(f"Generating embeddings for {args.dataset}...")
    embeddings = text_to_embedding(data.text, device=0)
    n = embeddings.shape[0]

    print("Calculating similarity matrix...")
    similarity_matrix = cosine_similarity(
        embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)

    # Set diagonal (self-similarity) to high value to exclude from negative samples
    eye = torch.eye(n) > 0.5
    similarity_matrix[eye] = 100000

    # Find k node pairs with lowest similarity
    k = min(4000, n * (n - 1) // 2)
    print(f"Selecting top {k} dissimilar node pairs...")
    values, indices = torch.topk(similarity_matrix.view(-1), k, largest=False)

    # Convert flat indices to 2D coordinates
    rows = indices // n
    cols = indices % n

    # Create negative edge index
    negative_edge_index = []
    for i in range(k):
        row, col = rows[i].item(), cols[i].item()
        negative_edge_index.append([row, col])
        if i < 10 or i % 1000 == 0:  # Only print a few examples
            print(
                f"Pair {i + 1}: Node {row} and Node {col}, Similarity: {values[i].item():.4f}")

    negative_edge_index = torch.tensor(
        np.array(negative_edge_index).transpose())

    # Save edges and create instruction dataset
    model_path = get_path_to("saved_model")
    edge_path = f"{model_path}/negative_edge/{args.dataset}.pth"
    jsonl_path = os.path.join(
        parent_dir, "LLaMA-Factory", "data", f"negative_{args.dataset}.jsonl")

    # Create directories if they don't exist
    ensure_dir_exists(edge_path)
    ensure_dir_exists(jsonl_path)

    json_list = create_node_pair_json(negative_edge_index, data.text)

    torch.save(negative_edge_index, edge_path)
    save_jsonl(json_list, jsonl_path)
    print(f"Negative samples saved to {edge_path}")
    print(f"Negative samples dataset saved to {jsonl_path}")


if __name__ == '__main__':
    args = parse_arguments()
    create_test_instruction(args)
    create_negative_samples(args)
