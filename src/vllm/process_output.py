#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
LLM Output Processing

This module processes the output of Large Language Models (LLMs) used for 
edge classification in graph neural networks. It converts raw inference results
into structured JSON format and organizes them for further analysis.

Author: Zhongjian Zhang
Date: 2024/12/9
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj, coalesce
from tqdm import tqdm, trange

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))
from util.load import load_data
from util.utils import get_path_to, csr_matrix_to_edge_index, save_jsonl


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Process LLM inference results for graph edge classification')
    
    parser.add_argument(
        '--llm', 
        type=str, 
        default="mistral-7b-merge",
        help='Name of the large language model used for inference'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='cora',
        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'],
        help='Dataset used for the experiment'
    )
    
    parser.add_argument(
        '--attack', 
        default='meta', 
        type=str, 
        choices=['meta', 'dice'],
        help='Type of adversarial attack applied to the graph'
    )
    
    parser.add_argument(
        '--ptb_rate', 
        default=0.1, 
        type=float, 
        choices=[0, 0.05, 0.1, 0.2, 0.4],
        help='Perturbation rate used in the attack (0 for clean graph)'
    )
    
    return parser.parse_args()


def process_raw_inference_results() -> None:
    """
    Process raw LLM inference results and save them as individual JSON files.
    
    This function reads the raw inference output and creates individual JSON files
    for each edge prediction, organized by node IDs.
    """
    args = parse_arguments()
    
    # Create output directory
    save_path = get_path_to("llm_response") / f"{args.llm}/all/{args.dataset}/"
    os.makedirs(save_path, exist_ok=True)
    
    # Input file with raw inference results
    inference_result_file = f'./output/{args.dataset}_{args.attack}_{args.ptb_rate}.jsonl'
    
    # Process each line in the inference results file
    with open(inference_result_file, 'r', encoding='utf-8') as f:
        for index, line in tqdm(enumerate(f), desc="Processing inference results"):
            # Parse the JSON response
            response_content = json.loads(line.strip())
            
            # Extract node IDs from the request ID
            request_id = response_content['custom_id'].split("_")
            node1_id = request_id[0]
            node2_id = request_id[1]
            
            # Extract the LLM's response
            inference_content = response_content['response']['body']['choices'][0]['message']['content'].strip()
            
            # Define output file path
            file_name = f"{node1_id}_{node2_id}.json"
            file_path = f"{save_path}/{file_name}"
            
            # Save the result if it doesn't already exist
            if not os.path.exists(file_path):
                # Ensure node1_id < node2_id for consistency
                assert int(node1_id) < int(node2_id), f"Expected node1_id < node2_id, got {node1_id} and {node2_id}"
                
                # Create JSON data structure
                json_data = {
                    "label": "",  # Empty label field, to be filled later if needed
                    "predict": inference_content
                }
                
                # Write to file
                with open(file_path, 'w', encoding='utf-8') as f2:
                    json.dump(json_data, f2, ensure_ascii=False, indent=4)


def load_edge_predictions(edge_index: torch.Tensor, request_dir: str, node_text: List[str]) -> List[Dict[str, Any]]:
    """
    Load JSON prediction files for the specified edges.
    
    Args:
        edge_index: Tensor of edge indices to process
        request_dir: Directory containing the JSON prediction files
        node_text: List of node text features (for potential future use)
        
    Returns:
        List of JSON objects containing the edge predictions
    """
    json_list = []
    
    # Process each edge
    for index in trange(len(edge_index[0]), desc="Loading edge predictions"):
        # Get node IDs for this edge
        node1_id = edge_index[0][index].item()
        node2_id = edge_index[1][index].item()
        
        # Construct file path
        file_name = f"{node1_id}_{node2_id}.json"
        file_path = os.path.join(request_dir, file_name)
        
        # Verify edge ordering and file existence
        assert node1_id < node2_id, f"Expected node1_id < node2_id, got {node1_id} and {node2_id}"
        assert os.path.exists(file_path), f"Prediction file not found: {file_path}"
        
        # Load and store the JSON data
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            json_list.append(json_data)
            
    return json_list


def construct_attack_predictions() -> None:
    """
    Construct a consolidated JSONL file with edge predictions for attacked graphs.
    
    This function processes individual edge prediction files and combines them
    into a single JSONL file for further analysis.
    """
    args = parse_arguments()
    
    # Load dataset
    data = load_data(args.dataset)
    
    # Set up paths
    model_path = get_path_to("saved_model")
    response_path = get_path_to("llm_response") / f"{args.llm}/"
    request_dir = response_path / f"all/{args.dataset}"
    
    # Set up paths based on whether using clean or attacked graph
    if args.ptb_rate > 0:
        # For attacked graph
        attack_file = f"{model_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        perturbed_edge_index = torch.load(attack_file)
        output_dir = f"{response_path}/global/{args.dataset}_{args.attack}_{args.ptb_rate}"
    else:
        # For clean graph
        perturbed_edge_index = data.edge_index
        output_dir = f"{response_path}/clean/{args.dataset}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/generated_predictions.jsonl"
    
    # Process edge index to upper triangular form and coalesce
    perturbed_edge_index = to_dense_adj(perturbed_edge_index.long())[0].triu_()
    perturbed_edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(perturbed_edge_index))))
    
    # Load JSON data for each edge
    json_list = load_edge_predictions(perturbed_edge_index, request_dir, data.text)
    
    # Verify data consistency
    print(f"Edge count: {perturbed_edge_index.shape[1]}, Prediction count: {len(json_list)}")
    
    # Save consolidated JSONL file
    save_jsonl(json_list, output_file)
    print(f"Successfully saved predictions to: {output_file}")


def main() -> None:
    """
    Main function to execute the LLM output processing pipeline.
    
    This runs both stages of the processing:
    1. Process raw inference results into individual JSON files
    2. Construct consolidated JSONL files for clean or attacked graphs
    """
    # Process raw inference results
    process_raw_inference_results()
    
    # Construct consolidated prediction files
    construct_attack_predictions()


if __name__ == '__main__':
    main()
