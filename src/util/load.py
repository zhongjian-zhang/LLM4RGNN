import logging

import torch
from torch_geometric.utils import is_undirected

from util.utils import edge_index_to_csr_matrix, get_path_to


def get_cora():
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/cora/cora.pt"
    dataset = torch.load(file_path, weights_only=False)
    return dataset


def get_citeseer():
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/citeseer/citeseer.pt"
    dataset = torch.load(file_path, weights_only=False)
    return dataset


def get_pubmed():
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/pubmed/pubmed.pt"
    dataset = torch.load(file_path, weights_only=False)
    return dataset


def get_ogbn_arxiv_subset():
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/arxiv/arxiv_subset.pt"
    dataset = torch.load(file_path, weights_only=False)
    return dataset


def get_ogbn_arxiv_full():
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/arxiv/arxiv_full.pt"
    dataset = torch.load(file_path, weights_only=False)
    return dataset


def get_ogbn_product_subset():
    data_path = get_path_to("dataset")
    file_path = f"{data_path}/product/product_subset.pt"
    dataset = torch.load(file_path)
    return dataset


def load_data(dataset):
    data = None
    if dataset == "cora":
        data = get_cora()
    elif dataset == "citeseer":
        data = get_citeseer()
    elif dataset == "pubmed":
        data = get_pubmed()
    elif dataset == "ogbn_arxiv":
        data = get_ogbn_arxiv_subset()
    elif dataset == "ogbn_arxiv_full":
        data = get_ogbn_arxiv_full()
    elif dataset == "ogbn_product":
        data = get_ogbn_product_subset()
    assert is_undirected(data.edge_index)
    logging.info(data)
    data.adj = edge_index_to_csr_matrix(data.edge_index, data.num_nodes)
    return data
