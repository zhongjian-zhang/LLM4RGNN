#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Graph Edge Prediction Trainer

This module implements a trainer for graph edge prediction models.
It handles loading data, training an edge predictor model, and applying the model
for graph purification/defense against adversarial attacks.

Author: Zhongjian Zhang
Date: 2024/4/3
"""
import logging
import os
import re
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ujson as json
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import coalesce, to_undirected
from tqdm import trange

from model.EdgePredictor import EdgePredictor
from util.utils import text_to_embedding


class Trainer:
    """
    Trainer for graph edge prediction models.
    
    This class handles the training and evaluation of edge prediction models,
    as well as the application of these models for graph purification/defense.
    """
    
    def __init__(self, text, ptb_edge_index, llm_response_file, node_emb_file, negative_edge_file,
                 negative_llm_response, candidate_node_file, purify_file, args):
        """
        Initialize the Trainer with the necessary data and configuration.
        
        Args:
            text (list): List of node text features
            ptb_edge_index (torch.Tensor): Perturbed edge index tensor
            llm_response_file (str): Path to file containing LLM responses for edge evaluation
            node_emb_file (str): Path to save/load node embeddings
            negative_edge_file (str): Path to negative edge samples
            negative_llm_response (str): Path to LLM responses for negative edges
            candidate_node_file (str): Path to save/load candidate nodes
            purify_file (str): Path to save the purified edge index
            args: Configuration arguments
        """
        # Store configuration parameters
        self.args = args
        self.top_k = args.top_k
        self.device = args.device
        self.epochs = args.edge_epochs
        self.early_stop = args.edge_early_stop
        self.confidence = args.confidence
        self.purify_threshold = args.purify_threshold
        self.positive_threshold = args.positive_threshold
        self.negative_threshold = args.negative_threshold
        
        # Store file paths
        self.llm_response_file = llm_response_file
        self.node_emb_file = node_emb_file
        self.candidate_node_file = candidate_node_file
        self.purify_file = purify_file
        self.negative_edge_file = negative_edge_file
        self.negative_llm_response = negative_llm_response
        
        # Initialize edge data structures
        self.purify_edge = []
        self.purify_edge_index = None
        self.ptb_edge_index = ptb_edge_index

        # Prepare data for training
        self.node_text_embs = self.get_text_embeddings(text)
        edge_index, edge_label = self.get_edge_and_label()
        train_x, train_y, test_x, test_y = self.preprocess_data(edge_index, edge_label, sample=True)
        
        # Create data loaders
        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        # Initialize model
        input_dim = self.node_text_embs.shape[1] * 2  # Concatenated node pairs
        self.edge_predictor = EdgePredictor(
            embedding_dim=input_dim, 
            hidden_dim=args.edge_hidden_dim, 
            num_classes=1, 
            num_layers=args.edge_num_layer,
            dropout=args.edge_dropout
        ).to(self.device)
        
        # Initialize training components
        self.best_weights = None
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.edge_predictor.parameters(), 
            lr=args.edge_lr,
            weight_decay=args.edge_weight_decay
        )

    def get_edge_and_label(self):
        """
        Extract edge indices and labels from LLM response file.
        
        This method processes the LLM responses to determine which edges to keep,
        and categorizes edges as positive or negative for training.
        
        Returns:
            tuple: (edge_index, edge_label) for training the edge predictor
        """
        edge_index, edge_label = [], []
        score_count = [0, 0, 0, 0, 0, 0, 0]  # Counts for each score (0-6)
        
        with open(self.llm_response_file, 'r', encoding="utf-8") as f:
            for index, line in enumerate(f):
                json_data = json.loads(line)
                # Extract relevance score using regex
                match = re.search(r'Relevance Score: (\d+)', json_data['predict'])
                prediction_score = int(match.group(1)) if match else 1
                score_count[prediction_score] += 2  # Count bidirectional edges
                
                # Get node IDs for the current edge
                node1_id, node2_id = self.ptb_edge_index[:, index]
                
                # Process edges based on scores
                if prediction_score >= self.purify_threshold:  # Edges to preserve
                    self.purify_edge.append([node1_id.item(), node2_id.item()])
                    self.purify_edge.append([node2_id.item(), node1_id.item()])
                    
                if prediction_score >= self.positive_threshold:  # Positive examples
                    edge_label.extend([1, 1])
                    edge_index.append([node1_id.item(), node2_id.item()])
                    edge_index.append([node2_id.item(), node1_id.item()])
                elif prediction_score <= self.negative_threshold:  # Negative examples
                    edge_label.extend([0, 0])
                    edge_index.append([node1_id.item(), node2_id.item()])
                    edge_index.append([node2_id.item(), node1_id.item()])
        
        # Convert to tensors
        edge_label = torch.tensor(edge_label).float()
        edge_index = torch.tensor(edge_index).T
        
        # Process purify edges into a coalesced, undirected edge index
        self.purify_edge_index = coalesce(
            to_undirected(torch.tensor(np.array(self.purify_edge).transpose())).int()
        )
        
        # Log score distribution
        logging.info("Edge score distribution: %s", score_count[1:])
        
        return edge_index, edge_label

    def get_negative_sample(self, sample_num, reverse=False):
        """
        Retrieve negative edge samples from LLM responses.
        
        Args:
            sample_num (int): Number of negative samples to retrieve
            reverse (bool): Whether to include edges in both directions
            
        Returns:
            tuple: (edge_index, edge_label) for negative samples
        """
        edge_index, edge_label = [], []
        negative_edge_num = 0
        
        if os.path.exists(self.negative_llm_response):
            # Load negative edge index
            negative_edge_index = torch.load(self.negative_edge_file)
            
            with open(self.negative_llm_response, 'r', encoding="utf-8") as f:
                for index, line in enumerate(f):
                    json_data = json.loads(line)
                    
                    # Extract relevance score
                    match = re.search(r'Relevance Score: (\d+)', json_data['predict'])
                    prediction_score = int(match.group(1)) if match else 6
                    
                    # Get node IDs for the current edge
                    node1_id, node2_id = negative_edge_index[:, index]
                    
                    if prediction_score <= self.negative_threshold:
                        # Add negative edge
                        negative_edge_num += 1
                        edge_label.append(0)
                        edge_index.append([node1_id.item(), node2_id.item()])
                        
                        if negative_edge_num == sample_num:
                            break
                            
                        # Add reverse edge if specified
                        if reverse:
                            negative_edge_num += 1
                            edge_label.append(0)
                            edge_index.append([node2_id.item(), node1_id.item()])
                            
                            if negative_edge_num == sample_num:
                                break
        else:
            logging.warning("Negative LLM response file does not exist: %s", self.negative_llm_response)
            
        logging.info("Sampling negative edge num: %d", negative_edge_num)
        
        # Convert to tensors
        edge_label = torch.tensor(edge_label).float()
        edge_index = torch.tensor(edge_index).T
        
        return edge_index, edge_label

    def get_text_embeddings(self, texts):
        """
        Get or compute text embeddings for nodes.
        
        Loads embeddings from file if available, otherwise computes and saves them.
        
        Args:
            texts (list): List of node text features
            
        Returns:
            torch.Tensor: Node embeddings tensor
        """
        if os.path.exists(self.node_emb_file):
            # Load pre-computed embeddings
            embeddings = torch.load(self.node_emb_file)
            return embeddings
        else:
            # Compute and save embeddings
            embeddings = text_to_embedding(texts, self.device).cpu()
            torch.save(embeddings, self.node_emb_file)
            return embeddings

    def preprocess_data(self, edge_index, edge_label, sample=True):
        """
        Preprocess edge data for training the edge predictor.
        
        This method handles balancing positive and negative samples and
        creating feature vectors for each edge pair.
        
        Args:
            edge_index (torch.Tensor): Tensor of edge indices
            edge_label (torch.Tensor): Tensor of edge labels (0 or 1)
            sample (bool): Whether to balance the dataset through sampling
            
        Returns:
            tuple: (train_x, train_y, test_x, test_y) for model training
        """
        if sample:
            # Count positive and negative samples
            pos_sample_num = int(edge_label.sum().item())
            neg_sample_num = int(edge_label.shape[0] - pos_sample_num)
            logging.info("Positive sample num: %d, Negative sample num: %d", pos_sample_num, neg_sample_num)
            
            # Calculate imbalance
            sample_diff = pos_sample_num - neg_sample_num
            
            # Handle case with more positive than negative samples
            if sample_diff > 0:
                # Get additional negative samples
                reverse = True if self.args.dataset in ["pubmed", "ogbn_arxiv", "ogbn_product"] else False
                negative_edge_index, negative_edge_label = self.get_negative_sample(sample_diff, reverse=reverse)
                
                # Combine with existing data
                edge_label = torch.cat([edge_label, negative_edge_label], dim=0)
                edge_index = torch.cat([edge_index, negative_edge_index], dim=1).int()
                
                # Recalculate counts
                pos_sample_num = int(edge_label.sum().item())
                neg_sample_num = int(edge_label.shape[0] - pos_sample_num)
                
                # If still imbalanced, subsample positive examples
                if pos_sample_num > neg_sample_num:
                    logging.info("Rebalancing - Positive: %d, Negative: %d", pos_sample_num, neg_sample_num)
                    
                    # Find positive examples and select a random subset
                    indices_label_1 = (edge_label == 1).nonzero(as_tuple=True)[0]
                    selected_indices_label_1 = indices_label_1[
                        torch.randperm(indices_label_1.size(0))[:neg_sample_num]
                    ]
                    sampled_edge_index_label_1 = edge_index[:, selected_indices_label_1]
                    
                    # Get all negative examples
                    indices_label_0 = (edge_label == 0).nonzero(as_tuple=True)[0]
                    sampled_edge_index_label_0 = edge_index[:, indices_label_0]
                    
                    # Create balanced dataset
                    edge_label = torch.tensor(
                        [0] * sampled_edge_index_label_0.shape[1] + 
                        [1] * sampled_edge_index_label_1.shape[1]
                    ).float()
                    edge_index = torch.cat(
                        (sampled_edge_index_label_0, sampled_edge_index_label_1), 
                        dim=1
                    ).int()
                    
            # Handle case with more negative than positive samples
            elif sample_diff < 0:
                # Find negative examples and select a random subset
                indices_label_0 = (edge_label == 0).nonzero(as_tuple=True)[0]
                selected_indices_label_0 = indices_label_0[
                    torch.randperm(indices_label_0.size(0))[:pos_sample_num]
                ]
                sampled_edge_index_label_0 = edge_index[:, selected_indices_label_0]
                
                # Get all positive examples
                indices_label_1 = (edge_label == 1).nonzero(as_tuple=True)[0]
                sampled_edge_index_label_1 = edge_index[:, indices_label_1]
                
                # Create balanced dataset
                edge_label = torch.tensor(
                    [0] * sampled_edge_index_label_0.shape[1] + 
                    [1] * sampled_edge_index_label_1.shape[1]
                ).float()
                edge_index = torch.cat(
                    (sampled_edge_index_label_0, sampled_edge_index_label_1), 
                    dim=1
                ).int()
                
        # Log final sample counts
        pos_sample_num = int(edge_label.sum().item())
        neg_sample_num = int(edge_label.shape[0] - pos_sample_num)
        logging.info("Final - Positive sample num: %d, Negative sample num: %d", pos_sample_num, neg_sample_num)
        
        # Create feature vectors by concatenating node embeddings
        data_features = []
        data_labels = edge_label
        
        for i in range(edge_index.shape[1]):
            node1_id, node2_id = edge_index[0][i], edge_index[1][i]
            data_features.append(
                torch.cat([self.node_text_embs[node1_id], self.node_text_embs[node2_id]])
            )
            
        data_features = torch.stack(data_features)
        
        # Split into train and test sets
        train_x, test_x, train_y, test_y = train_test_split(
            data_features, data_labels, test_size=0.2, random_state=self.args.seed
        )
        
        return train_x, train_y, test_x, test_y

    def train(self):
        """
        Train the edge predictor model.
        
        Implements the training loop with early stopping based on validation performance.
        """
        best_accuracy, best_loss = 0, float('inf')
        epochs_no_improve = 0
        
        # Set model to training mode
        self.edge_predictor.train()
        
        for epoch in range(1, self.epochs + 1):
            # Training phase
            train_loss = 0
            for _, (batch_features, batch_labels) in enumerate(self.train_loader):
                # Move data to device
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.edge_predictor(batch_features).squeeze()
                loss = self.criterion(outputs, batch_labels)
                train_loss += loss.item()
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluation phase
            self.edge_predictor.eval()
            with torch.no_grad():
                # Evaluate on training set
                total, correct = 0, 0
                for batch_features, batch_labels in self.train_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.edge_predictor(batch_features)
                    predicted = (outputs.squeeze() > 0.5).float()
                    
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    
                train_accuracy = 100 * correct / total

                # Evaluate on test set
                test_loss = 0
                total, correct = 0, 0
                for batch_features, batch_labels in self.test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.edge_predictor(batch_features)
                    predicted = (outputs.squeeze() > 0.5).float()
                    
                    loss = self.criterion(outputs.squeeze(), batch_labels)
                    test_loss += loss.item()
                    
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                    
                test_accuracy = 100 * correct / total
            
            # Calculate average loss
            train_loss /= len(self.train_loader)
            test_loss /= len(self.test_loader)
            
            # Check if model improved
            if test_accuracy > best_accuracy and test_loss < best_loss:
                best_loss, best_accuracy = test_loss, test_accuracy
                epochs_no_improve = 0
                
                logging.info(
                    "Epoch %d/%d, Train Loss: %.4f, Train ACC: %.4f || Test Loss: %.4f, Test ACC: %.4f",
                    epoch, self.epochs, train_loss, train_accuracy, test_loss, test_accuracy
                )
                
                # Save best model weights
                self.best_weights = deepcopy(self.edge_predictor.state_dict())
            else:
                epochs_no_improve += 1
                
            # Early stopping check
            if epochs_no_improve >= self.early_stop:
                logging.warning("Early stopping triggered after %d epochs", epoch)
                break
                
            # Reset to training mode for next epoch
            self.edge_predictor.train()

    def predict_batch(self, batch_pairs):
        """
        Predict edge scores for a batch of node pairs.
        
        Args:
            batch_pairs (list): List of node ID pairs to predict
            
        Returns:
            torch.Tensor: Predicted scores for each pair
        """
        # Create feature vectors for each pair
        batch_tensors = [
            torch.cat([self.node_text_embs[i], self.node_text_embs[j]]) 
            for i, j in batch_pairs
        ]
        batch_pairs_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            batch_scores = self.edge_predictor(batch_pairs_tensor).squeeze()
            
        # Handle case of single prediction
        if batch_scores.ndim == 0:
            batch_scores = batch_scores.unsqueeze(0)
            
        return batch_scores

    @staticmethod
    def update_edge_scores(edge_scores, batch_scores, batch_pairs):
        """
        Update the edge score matrix with batch prediction results.
        
        Args:
            edge_scores (torch.Tensor): Matrix to update with scores
            batch_scores (torch.Tensor): Predicted scores
            batch_pairs (list): Node pairs corresponding to the scores
        """
        for idx, score in enumerate(batch_scores):
            i, j = batch_pairs[idx]
            # Update scores in both directions (undirected graph)
            edge_scores[i, j] = score
            edge_scores[j, i] = score

    def get_candidate_nodes_dict(self, top_k=2000, batch_size=200):
        """
        Generate or load a dictionary of candidate nodes for each node.
        
        For each node, finds the top_k most similar nodes based on embedding similarity.
        
        Args:
            top_k (int): Number of similar nodes to find for each node
            batch_size (int): Batch size for processing
            
        Returns:
            dict: Dictionary mapping node IDs to lists of similar node pairs
        """
        # Load from file if exists
        if os.path.exists(self.candidate_node_file):
            with open(self.candidate_node_file, 'r') as f:
                candidate_nodes = json.load(f)
            return candidate_nodes
            
        logging.info("%s does not exist, starting to generate it...", self.candidate_node_file)
        
        n = self.node_text_embs.shape[0]
        ptb_edge_set = set(map(tuple, self.ptb_edge_index.t().tolist()))
        candidate_nodes = {}
        num_total_similar_nodes = 0
        
        # Process nodes in batches
        for start in trange(0, n, batch_size, desc=f"Finding top {top_k} most similar nodes", dynamic_ncols=True):
            end = min(start + batch_size, n)
            batch_embeds = self.node_text_embs[start:end]
            
            # Compute similarity between batch nodes and all nodes
            similarity_matrix = F.cosine_similarity(
                batch_embeds.unsqueeze(1), 
                self.node_text_embs.unsqueeze(0), 
                dim=2
            )
            
            # Exclude existing edges from consideration
            for i in range(start, end):
                for j in range(n):
                    if (i, j) in ptb_edge_set or (j, i) in ptb_edge_set:
                        similarity_matrix[i - start, j] = -1
                        
            # Exclude self-connections
            similarity_matrix[:, start:end].fill_diagonal_(-1)
            
            # Find top_k similar nodes for each node in batch
            for i in range(start, end):
                # Only consider nodes with higher indices to avoid duplicates
                similarities = similarity_matrix[i - start, i + 1:]
                
                # Get top_k most similar nodes
                top_k_values, top_k_indices = torch.topk(
                    similarities, 
                    min(top_k, n - i - 1)
                )
                
                # Store node pairs
                candidate_nodes[str(i)] = [
                    (i, i + 1 + j) for j in top_k_indices.cpu().numpy().tolist()
                ]
                
                num_total_similar_nodes += top_k_values.shape[0]

        # Log results
        logging.info(
            "The top %d most similar nodes for each node under %s with a %.2f perturbation rate: %d.",
            top_k, self.args.attack, self.args.ptb_rate, num_total_similar_nodes
        )
        
        # Save to file
        with open(self.candidate_node_file, 'w') as f:
            json.dump(candidate_nodes, f)
            
        logging.info("Candidate nodes saved as JSON to %s", self.candidate_node_file)
        
        return candidate_nodes

    def defense(self, sample_candidate_node=False, candidate_node_num=2000):
        """
        Apply the trained model to purify the graph by adding high-confidence edges.
        
        Args:
            sample_candidate_node (bool): Whether to use candidate node sampling for efficiency
            candidate_node_num (int): Number of candidate nodes to consider for each node
        """
        # Load best model weights
        self.edge_predictor.load_state_dict(self.best_weights)
        self.edge_predictor.eval()
        
        n = self.node_text_embs.shape[0]
        edge_scores = torch.zeros(n, n)
        batch_size = 10000
        
        # Predict edge scores
        if sample_candidate_node:
            # Use pre-computed candidate nodes for efficiency
            candidate_nodes_dict = self.get_candidate_nodes_dict(
                top_k=candidate_node_num, 
                batch_size=200
            )
            
            # Process each node
            for i in trange(n):
                batch_pairs = candidate_nodes_dict[str(i)]
                if len(batch_pairs) > 0:
                    batch_scores = self.predict_batch(batch_pairs)
                    self.update_edge_scores(edge_scores, batch_scores, batch_pairs)
        else:
            # Evaluate all possible edges
            for i in trange(n):
                batch_pairs = []
                for j in range(i + 1, n):
                    batch_pairs.append((i, j))
                    
                    # Process in batches
                    if len(batch_pairs) == batch_size:
                        batch_scores = self.predict_batch(batch_pairs)
                        self.update_edge_scores(edge_scores, batch_scores, batch_pairs)
                        batch_pairs = []
                        
                # Process remaining pairs
                if len(batch_pairs) > 0:
                    batch_scores = self.predict_batch(batch_pairs)
                    self.update_edge_scores(edge_scores, batch_scores, batch_pairs)

        # Select top-k edges for each node
        top_k_edges = []
        for i in range(n):
            scores = edge_scores[i].clone()
            
            # Exclude self-connections
            scores[i] = -float('inf')
            
            # Exclude existing neighbors
            out_neighbors = self.purify_edge_index[1, self.purify_edge_index[0] == i]
            in_neighbors = self.purify_edge_index[0, self.purify_edge_index[1] == i]
            all_neighbors = torch.cat((out_neighbors, in_neighbors)).unique().long()
            scores[all_neighbors] = -float('inf')
            
            # Find top-k highest scoring potential edges
            top_k_values, top_k_indices = torch.topk(scores, self.top_k)
            
            # Add edges above confidence threshold
            for index, idx in enumerate(top_k_indices):
                if top_k_values[index] > self.confidence:
                    top_k_edges.append([i, idx])
                    top_k_edges.append([idx, i])  # Add in both directions

        # Create final purified edge index
        top_k_edges = torch.tensor(np.array(top_k_edges).transpose())
        ori_edges = torch.tensor(np.array(self.purify_edge).transpose())
        
        # Combine original and new edges
        self.purify_edge_index = coalesce(
            to_undirected(torch.cat([ori_edges, top_k_edges], dim=1))
        ).int()
        
        # Save purified graph
        torch.save(self.purify_edge_index, self.purify_file)
