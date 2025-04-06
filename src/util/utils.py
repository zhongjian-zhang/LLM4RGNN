import datetime
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pytz
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer


def text_to_embedding(texts, device):
    model = SentenceTransformer(
        model_name_or_path='all-MiniLM-L6-v2', device=device)
    embs = torch.from_numpy(model.encode(texts))
    return embs


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_path_to(dir_name: str) -> Path:
    """Returns the path to fair-vision/data/."""
    assert dir_name in ['dataset', 'saved_model', 'llm_response']
    return get_or_create_path(get_root_path() / dir_name)


def get_root_path() -> Path:
    """Returns the path to fair-vision/."""
    return Path(__file__).resolve().parent.parent.parent


def get_or_create_path(p: Path, empty_dir: bool = False) -> Path:
    """Create a folder if it does not already exist."""
    if not p.is_dir():
        p.mkdir(parents=True)
    else:
        if empty_dir:
            empty_folder(p)
    return p


def empty_folder(p: Path):
    """Delete the contents of a folder."""
    for child_path in p.iterdir():
        if child_path.is_file():
            child_path.unlink()
        elif child_path.is_dir():
            shutil.rmtree(child_path)


def is_undirected(edge_index):
    """
    Determine whether the graph structure is undirected
    :param edge_index: the graph structure
    :return: if the graph structure is undirected, return True, otherwise False
    """
    edge_index = edge_index.cpu().tolist()
    edge_set = set(zip(edge_index[0], edge_index[1]))
    for start, end in edge_set:
        if (end, start) not in edge_set:
            return False
    return True


def edge_index_to_adj_matrix(edge_index, num_nodes):
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    start_nodes, end_nodes = edge_index
    for start, end in zip(start_nodes, end_nodes):
        adj_matrix[start, end] = 1

    return adj_matrix


def edge_index_to_csr_matrix(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    rows, cols = edge_index
    data = np.ones(len(rows))
    return csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow = torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol = torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat = torch.cat((sparserow, sparsecol), 1)
    sparsedata = torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(), sparsedata, torch.Size(sparse_mx.shape))


def csr_matrix_to_edge_index(csr_mat):
    row_indices, col_indices = csr_mat.nonzero()
    return [list(row_indices), list(col_indices)]


def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum().item()
    return correct / len(labels) * 100
