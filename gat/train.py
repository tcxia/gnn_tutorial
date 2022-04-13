import argparse
import random
import numpy as np

import torch


from utils import load_data
from model import GAT

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_ture", default=False, help="Disables CUDA training")
parser.add_argument("--fastmode", action="store_ture", default=False, help="Validate during training pass")
parser.add_argument("--sparse", action="store_ture", default=False, help="GAT with sparse version or not")
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


adj, featrues, labels, idx_train, idx_val, idx_test = load_data()

