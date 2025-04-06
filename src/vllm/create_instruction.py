#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   create_instruction.py
@Time    :   2024/9/30 14:28
@Author  :   zhongjian zhang
@Description: Generate instruction dataset for LLM inference with vLLM, including attacked graph structures
"""
import argparse
import os
import sys
import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj, coalesce
from tqdm import trange

# Add project root directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))
from util.load import load_data
from util.utils import get_path_to, csr_matrix_to_edge_index, save_jsonl

# LLM instruction template as system content
SYSTEM_CONTENT = """In the context of graph neural networks, attackers manipulate models by adding irrelevant edges or removing relevant ones, leading to incorrect predictions. Your role is crucial in defending against such attacks by evaluating the relevance between pairs of nodes, which will help in identifying and removing the irrelevant edges to mitigate the impact of adversarial attacks on graph-based models. Given textual information about two nodes, analyze the relevance of these two nodes. Provide a concise analysis(approximately 100 words) and assign an integer relevance score from 1 to 6, where 1 indicates completely irrelevant and 6 indicates directly relevant. Your response should be formatted in JSON, with two keys: "Analysis" for your written analysis and "Relevance Score" for your numerical evaluation."""


def get_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Object containing all command line arguments
    """
    parser = argparse.ArgumentParser(description='Creating the jsonl file for LLMs inference with vLLM!')
    parser.add_argument('--llm', type=str, default="mistral-7b-merge",
                        help='LLM model name to use for inference')
    parser.add_argument('--dataset', type=str, default='cora',
                        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'],
                        help='Dataset name to process')
    parser.add_argument('--attack', default='meta', type=str, choices=['meta', 'dice'],
                        help='Attack method type')
    parser.add_argument('--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4],
                        help='Perturbation rate (attack strength)')
    args = parser.parse_args()
    return args


def complete_test_instruction(edge_index, text, llm_path, request_path):
    """
    Generate vLLM inference requests for edges that haven't been processed yet
    
    Args:
        edge_index (torch.Tensor): Edge indices, shape [2, num_edges]
        text (list): Node text list
        llm_path (str): Path to the LLM model
        request_path (str): Path to store/check completed requests
        
    Returns:
        tuple: (Remaining edge indices to process, List of JSON request objects)
    """
    complete_json_list = []
    complete_edge_index = []
    
    # Create output directory if it doesn't exist
    os.makedirs(request_path, exist_ok=True)
    
    # Iterate through edges to create instruction data
    for index in trange(len(edge_index[0]), desc="Generating vLLM instruction data"):
        node1_id = edge_index[0][index].item()
        node2_id = edge_index[1][index].item()
        assert node1_id < node2_id  # Ensure node1_id < node2_id (upper triangular form)
        
        # Check if this edge has already been processed
        file_name = f"{node1_id}_{node2_id}.json"
        file_path = f"{request_path}/{file_name}"
        
        if not os.path.exists(file_path):
            # Get text content for both nodes
            node1_text, node2_text = text[node1_id], text[node2_id]
            user_content = f"Node1 -> {node1_text}\n\nNode2 -> {node2_text}"
            
            # Create vLLM API request format
            json_data = {
                "custom_id": f"{node1_id}_{node2_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": llm_path,
                    "messages": [
                        {"role": "system", "content": SYSTEM_CONTENT},
                        {"role": "user", "content": user_content}
                    ],
                    "max_tokens": 1000
                }
            }
            complete_json_list.append(json_data)
            complete_edge_index.append([node1_id, node2_id])
    
    # Return remaining edges to process and their instruction data
    return np.array(complete_edge_index).T if complete_edge_index else np.zeros((2, 0)), complete_json_list


def main():
    """
    Main function to create vLLM instruction data for the specified dataset and attack settings
    
    Process flow:
    1. Load dataset and model paths
    2. Load original or attacked edge indices based on parameters
    3. Process edge indices into the correct format
    4. Generate instruction data for remaining unprocessed edges
    5. Save instruction data to JSONL file for vLLM processing
    """
    args = get_args()
    path = get_path_to("saved_model")
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_data(args.dataset)
    text = data.text
    
    # Get edge index - either attacked or original based on ptb_rate
    if args.ptb_rate > 0:
        attack_file = f"{path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        print(f"Loading attacked graph from: {attack_file}")
        edge_index = torch.load(attack_file)
    else:
        print("Using original clean graph")
        edge_index = data.edge_index
    
    # Process edge index to upper triangular form and remove duplicates
    edge_index = to_dense_adj(edge_index)[0].triu_()  # Convert to dense adjacency matrix and get upper triangular
    edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(edge_index))))  # Convert back to edge index and deduplicate
    print(f"Total edges to process: {edge_index.shape[1]}")
    
    # Set paths for LLM and results
    request_path = os.path.join(get_path_to("llm_response"), f"{args.llm}/all/{args.dataset}")
    llm_path = f"../../saved_model/llm/{args.llm}"
    
    # Generate instruction data for remaining edges
    print(f"Checking for unprocessed edges in: {request_path}")
    complete_edge_index, complete_instruction_list = complete_test_instruction(edge_index, text, llm_path, request_path)
    print(f"Remaining edges to process: {complete_edge_index.shape[1]}")
    
    # Save instruction data if there are any remaining edges
    if complete_instruction_list:
        # Create instruction directory if it doesn't exist
        os.makedirs("./instruction", exist_ok=True)
        
        file_path = f'./instruction/{args.dataset}_{args.attack}_{args.ptb_rate}.jsonl'
        save_jsonl(complete_instruction_list, file_path)
        print(f"Instruction data saved to: {file_path}")
    else:
        print("No new edges to process. All edges have already been evaluated.")


if __name__ == '__main__':
    main()
