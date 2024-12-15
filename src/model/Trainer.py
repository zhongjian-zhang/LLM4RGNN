# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Training.py
@Time    :   2024/4/3 14:25
@Author  :   zhongjian zhang
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
    def __init__(self, text, ptb_edge_index, llm_response_file, node_emb_file, negative_edge_file,
                 negative_llm_response, candidate_node_file, purify_file, args):
        self.args = args
        self.top_k = args.top_k
        self.device = args.device
        self.epochs = args.edge_epochs
        self.early_stop = args.edge_early_stop
        self.confidence = args.confidence
        self.purify_threshold = args.purify_threshold
        self.positive_threshold = args.positive_threshold
        self.negative_threshold = args.negative_threshold
        self.llm_response_file = llm_response_file
        self.node_emb_file = node_emb_file
        self.candidate_node_file = candidate_node_file
        self.purify_file = purify_file
        self.negative_edge_file = negative_edge_file
        self.negative_llm_response = negative_llm_response
        self.purify_edge = []
        self.purify_edge_index = None
        self.ptb_edge_index = ptb_edge_index

        self.node_text_embs = self.get_text_emb(text)
        edge_index, edge_label = self.get_edge_and_label()
        train_x, train_y, test_x, test_y = self.preprocess_data(edge_index, edge_label, sample=True)
        train_dataset, test_dataset = TensorDataset(train_x, train_y), TensorDataset(test_x, test_y)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        input_dim = self.node_text_embs.shape[1] * 2
        self.edge_predictor = EdgePredictor(input_dim, args.edge_hidden_dim, 1, args.edge_num_layer,
                                            args.edge_dropout).to(self.device)
        self.best_weights = None
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.edge_predictor.parameters(), lr=args.edge_lr,
                                    weight_decay=args.edge_weight_decay)

    def get_edge_and_label(self):
        edge_index, edge_label = [], []
        score_count = [0, 0, 0, 0, 0, 0, 0]
        with open(self.llm_response_file, 'r', encoding="utf-8") as f:
            for index, line in enumerate(f):
                json_data = json.loads(line)
                match = re.search(r'Relevance Score: (\d+)', json_data['predict'])
                prediction_score = int(match.group(1)) if match else 1
                score_count[prediction_score] += 2
                node1_id, node2_id = self.ptb_edge_index[:, index]
                if prediction_score >= self.purify_threshold:  # preserved edges
                    self.purify_edge.append([node1_id.item(), node2_id.item()])
                    self.purify_edge.append([node2_id.item(), node1_id.item()])
                if prediction_score >= self.positive_threshold:  # positive edges
                    edge_label.extend([1, 1])
                    edge_index.append([node1_id.item(), node2_id.item()])
                    edge_index.append([node2_id.item(), node1_id.item()])
                elif prediction_score <= self.negative_threshold:  # negative edges
                    edge_label.extend([0, 0])
                    edge_index.append([node1_id.item(), node2_id.item()])
                    edge_index.append([node2_id.item(), node1_id.item()])
        edge_label, edge_index = torch.tensor(edge_label).float(), torch.tensor(edge_index).T
        self.purify_edge_index = coalesce(to_undirected(torch.tensor(np.array(self.purify_edge).transpose())).int())
        # Analysis the prediction results of LLMs
        logging.info("Edge score distribution: %s", score_count[1:])
        return edge_index, edge_label

    def get_negative_sample(self, sample_num, reverse=False):
        edge_index, edge_label = [], []
        negative_edge_num = 0
        if os.path.exists(self.negative_llm_response):
            negative_edge_index = torch.load(self.negative_edge_file)
            with open(self.negative_llm_response, 'r', encoding="utf-8") as f:
                for index, line in enumerate(f):
                    json_data = json.loads(line)
                    match = re.search(r'Relevance Score: (\d+)', json_data['predict'])
                    prediction_score = int(match.group(1)) if match else 6
                    node1_id, node2_id = negative_edge_index[:, index]
                    if prediction_score <= self.negative_threshold:
                        negative_edge_num += 1
                        edge_label.extend([0])
                        edge_index.append([node1_id.item(), node2_id.item()])
                        if negative_edge_num == sample_num:
                            break
                        if reverse:
                            negative_edge_num += 1
                            edge_label.extend([0])
                            edge_index.append([node2_id.item(), node1_id.item()])
                            if negative_edge_num == sample_num:
                                break
        else:
            logging.warning("Negative LLM response file does not exist: %s", self.negative_llm_response)
        logging.info("Sampling negative edge num: %d", negative_edge_num)
        edge_label = torch.tensor(edge_label).float()
        edge_index = torch.tensor(edge_index).T
        return edge_index, edge_label

    def get_text_emb(self, texts):
        if os.path.exists(self.node_emb_file):
            embs = torch.load(self.node_emb_file)
            return embs
        else:
            embs = text_to_embedding(texts, self.device).cpu()
            torch.save(embs, self.node_emb_file)
            return embs

    def preprocess_data(self, edge_index, edge_label, sample=True):
        if sample:
            pos_sample_num = int(edge_label.sum().item())
            neg_sample_num = int(edge_label.shape[0] - pos_sample_num)
            logging.info("Positive sample num: %d, Negative sample num: %d", pos_sample_num, neg_sample_num)
            sample_num = pos_sample_num - neg_sample_num
            if sample_num > 0:
                reverse = True if self.args.dataset in ["pubmed", "ogbn_arxiv", "ogbn_product"] else False
                negative_edge_index, negative_edge_label = self.get_negative_sample(sample_num, reverse=reverse)
                edge_label = torch.cat([edge_label, negative_edge_label], dim=0)
                edge_index = torch.cat([edge_index, negative_edge_index], dim=1).int()

                pos_sample_num = int(edge_label.sum().item())
                neg_sample_num = int(edge_label.shape[0] - pos_sample_num)
                if pos_sample_num > neg_sample_num:
                    logging.info("Positive sample num: %d, Negative sample num: %d", pos_sample_num, neg_sample_num)
                    indices_label_1 = (edge_label == 1).nonzero(as_tuple=True)[0]
                    selected_indices_label_1 = indices_label_1[torch.randperm(indices_label_1.size(0))[:neg_sample_num]]
                    sampled_edge_index_label_1 = edge_index[:, selected_indices_label_1]
                    indices_label_0 = (edge_label == 0).nonzero(as_tuple=True)[0]
                    sampled_edge_index_label_0 = edge_index[:, indices_label_0]
                    edge_label = torch.tensor(
                        [0] * sampled_edge_index_label_0.shape[1] + [1] * sampled_edge_index_label_1.shape[1]).float()
                    edge_index = torch.cat((sampled_edge_index_label_0, sampled_edge_index_label_1), dim=1).int()
            elif sample_num < 0:
                indices_label_0 = (edge_label == 0).nonzero(as_tuple=True)[0]
                selected_indices_label_0 = indices_label_0[torch.randperm(indices_label_0.size(0))[:pos_sample_num]]
                sampled_edge_index_label_0 = edge_index[:, selected_indices_label_0]
                indices_label_1 = (edge_label == 1).nonzero(as_tuple=True)[0]
                sampled_edge_index_label_1 = edge_index[:, indices_label_1]
                edge_label = torch.tensor(
                    [0] * sampled_edge_index_label_0.shape[1] + [1] * sampled_edge_index_label_1.shape[1]).float()
                edge_index = torch.cat((sampled_edge_index_label_0, sampled_edge_index_label_1), dim=1).int()
        pos_sample_num = int(edge_label.sum().item())
        neg_sample_num = int(edge_label.shape[0] - pos_sample_num)
        logging.info("Positive sample num: %d, Negative sample num: %d", pos_sample_num, neg_sample_num)
        data_x, data_y = [], edge_label
        for i in range(edge_index.shape[1]):
            node1_id, node2_id = edge_index[0][i], edge_index[1][i]
            data_x.append(torch.cat([self.node_text_embs[node1_id], self.node_text_embs[node2_id]]))
        data_x = torch.stack(data_x)
        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=self.args.seed)
        return train_x, train_y, test_x, test_y

    def train(self):
        best_accuracy, best_loss = 0, 10000
        epochs_no_improve = 0
        self.edge_predictor.train()
        for epoch in range(1, self.epochs + 1):
            train_loss, test_loss = 0, 0
            for i, (batch_embeds, batch_labels) in enumerate(self.train_loader):
                batch_embeds, batch_labels = batch_embeds.to(self.device), batch_labels.to(self.device)
                outputs = self.edge_predictor(batch_embeds)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, batch_labels)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.edge_predictor.eval()
            with torch.no_grad():
                total, correct = 0, 0
                for batch_embeds, batch_labels in self.train_loader:
                    batch_embeds, batch_labels = batch_embeds.to(self.device), batch_labels.to(self.device)
                    outputs = self.edge_predictor(batch_embeds)
                    predicted = (outputs.squeeze() > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                train_accuracy = 100 * correct / total

                total, correct = 0, 0
                for batch_embeds, batch_labels in self.test_loader:
                    batch_embeds, batch_labels = batch_embeds.to(self.device), batch_labels.to(self.device)
                    outputs = self.edge_predictor(batch_embeds)
                    predicted = (outputs.squeeze() > 0.5).float()
                    loss = self.criterion(outputs.squeeze(), batch_labels)
                    test_loss += loss.item()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
                test_accuracy = 100 * correct / total
            train_loss /= len(self.train_loader)
            test_loss /= len(self.test_loader)
            if test_accuracy > best_accuracy and test_loss < best_loss:
                best_loss, best_accuracy = test_loss, test_accuracy
                epochs_no_improve = 0
                logging.info("Epoch %d/%d, Train Loss: %.4f, Train ACC: %.4f || Test Loss: %.4f, Test ACC: %.4f",
                             epoch, self.epochs, train_loss, train_accuracy, test_loss, test_accuracy)
                self.best_weights = deepcopy(self.edge_predictor.state_dict())
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self.early_stop:
                logging.warning("Early stopping triggered after %d epochs", epoch + 1)
                break

    def predict_batch(self, batch_pairs):
        batch_tensors = [torch.cat([self.node_text_embs[i], self.node_text_embs[j]]) for i, j in batch_pairs]
        batch_pairs_tensor = torch.stack(batch_tensors).to(self.device)
        with torch.no_grad():
            batch_scores = self.edge_predictor(batch_pairs_tensor).squeeze()
        if batch_scores.ndim == 0:
            batch_scores = batch_scores.unsqueeze(0)
        return batch_scores

    @staticmethod
    def update_edge_scores(edge_scores, batch_scores, batch_pairs):
        for idx, score in enumerate(batch_scores):
            i, j = batch_pairs[idx]
            edge_scores[i, j] = score
            edge_scores[j, i] = score

    def get_candidate_nodes_dict(self, top_k=2000, batch_size=200):
        if os.path.exists(self.candidate_node_file):
            with open(self.candidate_node_file, 'r') as f:
                candidate_nodes = json.load(f)
            return candidate_nodes
        logging.info("%s does not exist, starting to generate it...", self.candidate_node_file)
        n = self.node_text_embs.shape[0]
        ptb_edge_set = set(map(tuple, self.ptb_edge_index.t().tolist()))
        candidate_nodes = {}
        num_total_similar_nodes = 0
        for start in trange(0, n, batch_size, desc=f"Finding top {top_k} most similar nodes", dynamic_ncols=True):
            end = min(start + batch_size, n)
            batch_embeds = self.node_text_embs[start:end]
            similarity_matrix = F.cosine_similarity(batch_embeds.unsqueeze(1), self.node_text_embs.unsqueeze(0), dim=2)
            for i in range(start, end):
                for j in range(n):
                    if (i, j) in ptb_edge_set or (j, i) in ptb_edge_set:
                        similarity_matrix[i - start, j] = -1
            similarity_matrix[:, start:end].fill_diagonal_(-1)
            for i in range(start, end):
                similarities = similarity_matrix[i - start, i + 1:]
                top_k_values, top_k_indices = torch.topk(similarities, min(top_k, n - i - 1))
                candidate_nodes[str(i)] = [(i, i + 1 + j) for j in top_k_indices.cpu().numpy().tolist()]
                num_total_similar_nodes += top_k_values.shape[0]

        logging.info("The top %d most similar nodes for each node under %s with a %.2f perturbation rate: %d.",
                     top_k, self.args.attack, self.args.ptb_rate, num_total_similar_nodes)
        with open(self.candidate_node_file, 'w') as f:
            json.dump(candidate_nodes, f)
        logging.info("Candidate nodes saved as JSON to %s", self.candidate_node_file)
        return candidate_nodes

    def defense(self, sample_candidate_node=False, candidate_node_num=2000):
        self.edge_predictor.load_state_dict(self.best_weights)
        self.edge_predictor.eval()
        n = self.node_text_embs.shape[0]
        edge_scores = torch.zeros(n, n)
        batch_size = 10000
        if sample_candidate_node:
            candidate_nodes_dict = self.get_candidate_nodes_dict(top_k=candidate_node_num, batch_size=200)
            for i in trange(n):
                batch_pairs = candidate_nodes_dict[str(i)]
                if len(batch_pairs) > 0:
                    batch_scores = self.predict_batch(batch_pairs)
                    self.update_edge_scores(edge_scores, batch_scores, batch_pairs)
        else:
            for i in trange(n):
                batch_pairs = []
                for j in range(i + 1, n):
                    batch_pairs.append((i, j))
                    if len(batch_pairs) == batch_size:
                        batch_scores = self.predict_batch(batch_pairs)
                        self.update_edge_scores(edge_scores, batch_scores, batch_pairs)
                        batch_pairs = []
                if len(batch_pairs) > 0:
                    batch_scores = self.predict_batch(batch_pairs)
                    self.update_edge_scores(edge_scores, batch_scores, batch_pairs)

        top_k_edges = []
        for i in range(n):
            scores = edge_scores[i]
            scores[i] = -float('inf')
            out_neighbors = self.purify_edge_index[1, self.purify_edge_index[0] == i]
            in_neighbors = self.purify_edge_index[0, self.purify_edge_index[1] == i]
            all_neighbors = torch.cat((out_neighbors, in_neighbors)).unique().long()
            scores[all_neighbors] = -float('inf')
            top_k_values, top_k_indices = torch.topk(scores, self.top_k)
            for index, idx in enumerate(top_k_indices):
                if top_k_values[index] > self.confidence:
                    top_k_edges.append([i, idx])
                    top_k_edges.append([idx, i])

        top_k_edges = torch.tensor(np.array(top_k_edges).transpose())
        ori_edges = torch.tensor(np.array(self.purify_edge).transpose())
        self.purify_edge_index = coalesce(to_undirected(torch.cat([ori_edges, top_k_edges], dim=1))).int()
        torch.save(self.purify_edge_index, self.purify_file)
