#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?

This script implements a framework for improving the robustness of 
Graph Neural Networks (GNNs) via Large Language Models (LLMs). 

Author: Zhongjian Zhang
Date: 2024/4/3
"""
import argparse
import logging
import os
import sys
import warnings

# Configure logging and suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import coalesce, to_dense_adj

# Add parent directory to path to correctly import util module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.GCN import GCN
from model.Trainer import Trainer
from util.load import load_data
from util.utils import (
    setup_seed, get_path_to, csr_matrix_to_edge_index, 
    edge_index_to_csr_matrix, accuracy, sparse_mx_to_torch_sparse_tensor
)


def parse_arguments():
    """
    Parse command line arguments for the framework.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Large Language Models for Robust Graph Neural Networks'
    )
    
    # =============================== General Settings =====================================
    general_group = parser.add_argument_group('General Settings')
    general_group.add_argument(
        '--dataset', type=str, default='cora',
        choices=['cora', 'citeseer', 'pubmed', 'ogbn_arxiv_full', 'ogbn_arxiv', 'ogbn_product'],
        help='Dataset to use for experiments'
    )
    general_group.add_argument(
        '--device', type=int, default=1,
        help='GPU device ID to use'
    )
    general_group.add_argument(
        '--seed', type=int, default=1,
        help='Random seed for reproducibility'
    )
    general_group.add_argument(
        '--llm', type=str, default="mistral-7b-merge",
        help='Large language model to use for edge prediction'
    )

    # =============================== Attack Settings ======================================
    attack_group = parser.add_argument_group('Attack Settings')
    attack_group.add_argument(
        '--attack', default='meta', type=str, choices=['meta', 'dice'],
        help='Type of adversarial attack to apply'
    )
    attack_group.add_argument(
        '--ptb_rate', default=0.1, type=float, choices=[0, 0.05, 0.1, 0.2, 0.4],
        help='Perturbation rate for the attack'
    )

    # =============================== Edge Predictor Settings ==============================
    edge_pred_group = parser.add_argument_group('Edge Predictor Settings')
    edge_pred_group.add_argument(
        '--sample_candidate_node', default=False, action='store_true',
        help='Whether to sample candidate nodes for efficiency'
    )
    edge_pred_group.add_argument(
        '--candidate_node_num', type=int, default=2000,
        help='Number of candidate nodes to consider for each node'
    )
    edge_pred_group.add_argument(
        '--edge_hidden_dim', type=int, default=512,
        help='Hidden dimension size for edge predictor model'
    )
    edge_pred_group.add_argument(
        '--batch_size', type=int, default=1024,
        help='Batch size for training'
    )
    edge_pred_group.add_argument(
        '--edge_lr', type=float, default=0.001,
        help='Learning rate for edge predictor'
    )
    edge_pred_group.add_argument(
        '--edge_epochs', type=int, default=1000,
        help='Maximum number of training epochs'
    )
    edge_pred_group.add_argument(
        '--edge_early_stop', type=int, default=100,
        help='Early stopping patience'
    )
    edge_pred_group.add_argument(
        '--edge_num_layer', type=int, default=4,
        help='Number of layers in edge predictor'
    )
    edge_pred_group.add_argument(
        '--edge_weight_decay', type=float, default=5e-5,
        help='Weight decay for edge predictor'
    )
    edge_pred_group.add_argument(
        '--edge_dropout', type=float, default=0.7,
        help='Dropout rate for edge predictor'
    )

    # =============================== LLM Thresholds ======================================
    threshold_group = parser.add_argument_group('LLM Thresholds')
    threshold_group.add_argument(
        '--positive_threshold', type=int, default=5, choices=[5, 6],
        help='Threshold for positive edge classification'
    )
    threshold_group.add_argument(
        '--negative_threshold', type=int, default=4, choices=[4, 1],
        help='Threshold for negative edge classification'
    )
    threshold_group.add_argument(
        '--purify_threshold', type=int, default=4, choices=[1, 2, 4],
        help='Threshold for edge preservation during purification'
    )
    threshold_group.add_argument(
        '--confidence', type=float, default=0.91, choices=[0.91, 0.93, 0.95, 0.97, 0.99],
        help='Confidence threshold for adding predicted edges'
    )
    threshold_group.add_argument(
        '--top_k', type=int, default=3, choices=[1, 3, 5, 7, 9],
        help='Number of top edges to consider per node'
    )

    args = parser.parse_args()
    return args


def purify_graph():
    """
    Purify the graph topology by using LLM-based edge prediction.
    
    This function:
    1. Loads data and attack information
    2. Initializes the edge predictor model
    3. Trains the model to predict edge existence
    4. Applies the model to purify the graph by adding high-confidence edges
    """
    logging.info("=" * 35 + " Purifying Attacked Graph Topology " + "=" * 35)
    
    # Parse arguments and set up environment
    args = parse_arguments()
    setup_seed(args.seed)
    logging.info("Arguments: %s", ", ".join([f"{k}={v}" for k, v in args.__dict__.items()]))
    
    # Load data and prepare file paths
    data = load_data(dataset=args.dataset)
    model_path = get_path_to("saved_model")
    response_path = get_path_to("llm_response")
    
    # Set up file paths based on whether using clean or attacked graph
    if args.ptb_rate > 0:
        # For attacked graph
        attack_file = f"{model_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
        purify_file = (f'{model_path}/purify/global/{args.dataset}_{args.attack}_'
                       f'{args.ptb_rate}_{args.seed}_add_{str(args.top_k)}.pth')
        llm_response_file = (f"{response_path}/{args.llm}/global/{args.dataset}_"
                            f"{args.attack}_{args.ptb_rate}/generated_predictions.jsonl")
        
        # Load attacked edge index
        ptb_edge_index = torch.load(attack_file)
        logging.info("Load the attack from: %s", attack_file)
    else:
        # For clean graph
        purify_file = f'{model_path}/purify/clean/{args.dataset}_{args.seed}_add_{str(args.top_k)}.pth'
        llm_response_file = f"{response_path}/{args.llm}/clean/{args.dataset}/generated_predictions.jsonl"
        ptb_edge_index = data.edge_index
    
    # Set up candidate node file if sampling is enabled
    if args.sample_candidate_node:
        candidate_node_file = (f"{model_path}/candidate_node/{args.dataset}/"
                              f"{args.attack}_{args.ptb_rate}_{args.candidate_node_num}.json")
        logging.info("Load the candidate important nodes from: %s", candidate_node_file)
    else:
        candidate_node_file = None
    
    # Set up additional file paths
    node_emb_file = f'{model_path}/node_emb/sbert/{args.dataset}.pth'
    negative_edge_file = f"{model_path}/negative_edge/{args.dataset}.pth"
    negative_llm_response = f"{response_path}/{args.llm}/negative/{args.dataset}/generated_predictions.jsonl"
    
    # Log file paths
    logging.info("Load the LLM prediction from: %s", llm_response_file)
    logging.info("Load the node embedding from: %s", node_emb_file)
    logging.info("Save the purified structure to: %s", purify_file)
    
    # Process edge index to upper triangular form and coalesce
    ptb_edge_index = to_dense_adj(ptb_edge_index)[0].triu_()
    ptb_edge_index = coalesce(torch.tensor(csr_matrix_to_edge_index(csr_matrix(ptb_edge_index))))
    
    # Initialize trainer and run purification
    trainer = Trainer(
        text=data.text, 
        ptb_edge_index=ptb_edge_index, 
        llm_response_file=llm_response_file, 
        node_emb_file=node_emb_file, 
        negative_edge_file=negative_edge_file,
        negative_llm_response=negative_llm_response, 
        candidate_node_file=candidate_node_file, 
        purify_file=purify_file, 
        args=args
    )
    
    # Train the edge predictor and apply for defense
    trainer.train()
    trainer.defense(
        sample_candidate_node=args.sample_candidate_node, 
        candidate_node_num=args.candidate_node_num
    )


def evaluate_gnn(data, args):
    """
    Evaluate GNN performance on the purified graph.
    
    Args:
        data: The graph data object containing node features, adjacency matrix, etc.
        args: Command line arguments containing model parameters
    
    Returns:
        float: Test accuracy of the trained GNN
    """
    # Set device and model parameters based on dataset
    device = args.device
    
    # Adjust training iterations based on dataset
    if args.dataset in ["pubmed", "ogbn_arxiv", "ogbn_arxiv_full", "ogbn_product"]:
        train_iters = 1000
        n_hidden = 256
    else:
        train_iters = 200
        n_hidden = 16
    
    # Initialize and train GCN model
    gnn = GCN(
        nfeat=args.gnn_input_dim, 
        nhid=n_hidden, 
        nclass=args.gnn_num_classes, 
        device=device
    ).to(device)
    
    # Fit model to data
    gnn.fit(
        data.x, data.adj, data.y, data.train_id, data.val_id, 
        verbose=False, train_iters=train_iters
    )
    
    # Get model output and calculate test accuracy
    output = gnn.output
    test_accuracy = accuracy(output[data.test_id], data.y[data.test_id])

    return test_accuracy


def evaluate_gnn_performance():
    """
    Test the performance of GCN on the purified graph.
    
    Loads the purified graph, constructs the adjacency matrix, and evaluates GNN performance.
    """
    logging.info("=" * 35 + " Testing the Performance of GCN " + "=" * 35)
    
    # Parse arguments and set up environment
    args = parse_arguments()
    logging.info("Arguments: %s", ", ".join([f"{k}={v}" for k, v in args.__dict__.items()]))
    
    # Load data and set random seed
    data = load_data(dataset=args.dataset)
    setup_seed(args.seed)
    
    # Prepare file paths
    model_path = get_path_to("saved_model")
    
    # Set purify file path based on whether using clean or attacked graph
    if args.ptb_rate > 0:
        purify_file = (f'{model_path}/purify/global/{args.dataset}_{args.attack}_'
                      f'{args.ptb_rate}_{args.seed}_add_{str(args.top_k)}.pth')
    else:
        purify_file = f'{model_path}/purify/clean/{args.dataset}_{args.seed}_add_{str(args.top_k)}.pth'
    
    # Add GNN parameters to args
    vars(args)['gnn_input_dim'] = data.x.shape[1]
    vars(args)['gnn_num_classes'] = data.num_classes
    
    # Load purified edge index
    edge_index = torch.load(purify_file, map_location='cpu')
    logging.info("Load the purified structure from: %s", purify_file)
    logging.info("The edge index shape: %s", edge_index.shape)
    
    # Convert edge index to adjacency matrix
    adjacency = edge_index_to_csr_matrix(edge_index, data.num_nodes)
    data.adj = sparse_mx_to_torch_sparse_tensor(adjacency)
    data.edge_index = edge_index
    
    # Move data to device
    data = data.to(args.device)
    
    # Evaluate GNN performance
    accuracy_score = evaluate_gnn(data, args)
    logging.info("Accuracy: %.4f", accuracy_score)


def main():
    """
    Main function to run the complete workflow:
    1. Purify the graph using LLM-based edge prediction
    2. Evaluate GNN performance on the purified graph
    """
    purify_graph()
    evaluate_gnn_performance()


if __name__ == '__main__':
    main()
