#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Dataset Loading Utilities

This module provides functions for loading different graph datasets used in the
experiments. It handles loading pre-processed PyTorch dataset files and preparing
them for use with GNN models.

Author: Zhongjian Zhang
"""
import logging
from typing import Union

import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

from util.utils import edge_index_to_csr_matrix, get_path_to


def load_dataset(dataset_name: str, file_path: str) -> Data:
    """
    Load a dataset from a PyTorch file.
    
    Args:
        dataset_name: Name of the dataset for logging purposes
        file_path: Path to the PyTorch dataset file
        
    Returns:
        Data object containing the loaded dataset
    """
    logging.info(f"Loading {dataset_name} dataset from {file_path}")
    dataset = torch.load(file_path, weights_only=False)
    return dataset


def get_cora() -> Data:
    """
    Load the Cora citation network dataset.
    
    The Cora dataset consists of machine learning papers, with citation links
    between them. Each paper is classified into one of seven classes.
    
    Returns:
        Data object containing the Cora dataset
    """
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/cora/cora.pt"
    return load_dataset("Cora", file_path)


def get_citeseer() -> Data:
    """
    Load the CiteSeer citation network dataset.
    
    The CiteSeer dataset consists of scientific publications, with citation links
    between them. Each publication is classified into one of six classes.
    
    Returns:
        Data object containing the CiteSeer dataset
    """
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/citeseer/citeseer.pt"
    return load_dataset("CiteSeer", file_path)


def get_pubmed() -> Data:
    """
    Load the PubMed citation network dataset.
    
    The PubMed dataset consists of medical publications related to diabetes,
    with citation links between them. Each publication is classified into
    one of three classes.
    
    Returns:
        Data object containing the PubMed dataset
    """
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/pubmed/pubmed.pt"
    return load_dataset("PubMed", file_path)


def get_ogbn_arxiv_subset() -> Data:
    """
    Load a subset of the ogbn-arxiv citation network dataset.
    
    This is a subset of the full ogbn-arxiv dataset, containing arXiv papers
    and their citation relationships.
    
    Returns:
        Data object containing the ogbn-arxiv subset
    """
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/arxiv/arxiv_subset.pt"
    return load_dataset("ogbn-arxiv subset", file_path)


def get_ogbn_arxiv_full() -> Data:
    """
    Load the full ogbn-arxiv citation network dataset.
    
    The ogbn-arxiv dataset contains arXiv papers from the cs category and
    their citation relationships.
    
    Returns:
        Data object containing the full ogbn-arxiv dataset
    """
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/arxiv/arxiv_full.pt"
    return load_dataset("full ogbn-arxiv", file_path)


def get_ogbn_product_subset() -> Data:
    """
    Load a subset of the ogbn-products co-purchasing network dataset.
    
    This is a subset of the full ogbn-products dataset, containing Amazon products
    and their co-purchasing relationships.
    
    Returns:
        Data object containing the ogbn-products subset
    """
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/product/product_subset.pt"
    return load_dataset("ogbn-products subset", file_path)


def load_data(dataset: str) -> Data:
    """
    Load a dataset by name and prepare it for GNN training.
    
    This function loads the specified dataset, verifies that the graph is undirected,
    and prepares the adjacency matrix in CSR format.
    
    Args:
        dataset: Name of the dataset to load (one of: "cora", "citeseer", "pubmed",
                "ogbn_arxiv", "ogbn_arxiv_full", "ogbn_product")
                
    Returns:
        Data object containing the prepared dataset
        
    Raises:
        AssertionError: If the loaded graph is not undirected
    """
    # Dictionary mapping dataset names to loading functions
    dataset_loaders = {
        "cora": get_cora,
        "citeseer": get_citeseer,
        "pubmed": get_pubmed,
        "ogbn_arxiv": get_ogbn_arxiv_subset,
        "ogbn_arxiv_full": get_ogbn_arxiv_full,
        "ogbn_product": get_ogbn_product_subset
    }
    
    # Check if dataset name is valid
    if dataset not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Available datasets: {list(dataset_loaders.keys())}")
    
    # Load the dataset
    data = dataset_loaders[dataset]()
    
    # Verify the graph is undirected
    assert is_undirected(data.edge_index), f"The {dataset} graph must be undirected"
    
    # Log dataset information
    logging.info(f"Loaded {dataset} dataset: {data}")
    logging.info(f"Number of nodes: {data.num_nodes}, Number of edges: {data.edge_index.size(1)//2}")
    
    # Prepare adjacency matrix in CSR format
    data.adj = edge_index_to_csr_matrix(data.edge_index, data.num_nodes)
    
    return data
