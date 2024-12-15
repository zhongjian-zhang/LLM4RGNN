import argparse
import os
import sys

from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack.mettack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from scipy.sparse import csr_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from util.load import load_data
from util.utils import csr_matrix_to_edge_index, get_path_to

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--attack', default='meta', type=str,
                    choices=['random', 'meta', 'nettack', 'sga', 'minmax', 'pgd', 'dice'])
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'ogbn_arxiv', 'ogbn_product', 'citeseer'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.2, help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self',
                    choices=['A-Meta-Self', 'Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

args = parser.parse_args()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
data = load_data(dataset=args.dataset)
adj, features, labels = data.adj, data.x.numpy(), data.y.numpy()
idx_train, idx_val, idx_test = data.train_id, data.val_id, data.test_id
idx_unlabeled = np.union1d(idx_val, idx_test)

perturbations = int(args.ptb_rate * (adj.sum() // 2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Setup Surrogate Model
if args.dataset != "ogbn_arxiv" and args.dataset != "ogbn_product":
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
else:
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=256,
                    dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                       attack_features=False, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=True,
                      attack_features=False, device=device, lambda_=lambda_)

model = model.to(device)


def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)  # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    model.attack(features, adj.numpy(), labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    modified_adj = model.modified_adj
    # modified_features = model.modified_features
    test(modified_adj)
    perturb_edge_index = torch.tensor(csr_matrix_to_edge_index(csr_matrix(model.modified_adj.cpu().numpy()))).long()
    saved_path = get_path_to("saved_model")
    file = f"{saved_path}/attack/global/{args.dataset}_{args.attack}_{args.ptb_rate}.pth"
    torch.save(perturb_edge_index, file)


if __name__ == '__main__':
    main()
