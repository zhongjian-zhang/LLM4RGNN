import argparse
import os
import sys

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack.dice import DICE
from deeprobust.graph.utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), '../')))
from util.utils import csr_matrix_to_edge_index, get_path_to
from util.load import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--attack', default='dice', type=str)
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'ogbn_arxiv', 'citeseer', 'ogbn_product', 'pubmed', 'ogbn_arxiv_full'])
parser.add_argument('--ptb_rate', type=float, default=0.1, help='perturbation rate')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = load_data(dataset=args.dataset)
adj, features, labels = data.adj, data.x.numpy(), data.y.numpy()
idx_train, idx_val, idx_test = data.train_id, data.val_id, data.test_id
idx_unlabeled = np.union1d(idx_val, idx_test)

# Setup Attack Model
model = DICE()

n_perturbations = int(args.ptb_rate * (adj.sum() // 2))

model.attack(adj, labels, n_perturbations)
modified_adj = model.modified_adj
# save attack
perturb_edge_index = torch.tensor(csr_matrix_to_edge_index(model.modified_adj)).long()
saved_path = get_path_to("saved_model")
file = f"{saved_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
# torch.save(perturb_edge_index, file)

# adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, sparse=True, device=device)
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, device=device)

modified_adj = normalize_adj(modified_adj)
modified_adj = sparse_mx_to_torch_sparse_tensor(modified_adj)
modified_adj = modified_adj.to(device)


def test(adj):
    ''' test on GCN '''
    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=256,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, verbose=True, train_iters=1000)  # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    print('=== testing GCN on perturbed graph ===')
    test(modified_adj)
    # save attack
    perturb_edge_index = torch.tensor(csr_matrix_to_edge_index(model.modified_adj)).long()
    saved_path = get_path_to("saved_model")
    file = f"{saved_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
    torch.save(perturb_edge_index, file)


if __name__ == '__main__':
    main()
