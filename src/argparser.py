import argparse
import yaml

parser = argparse.ArgumentParser(description='Generic runner for Composite models')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--checkpoint_dir', default="./checkpoints/cifar10/", type=str)
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--fs', default=4, type=int, help='feature scale')
parser.add_argument('--n_gpu', default=1, type=int)
parser.add_argument('--steps', default=36, type=int)
parser.add_argument('--resume', default=False, action='store_true' )
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--data_path', default='../data', type=str)
parser.add_argument('--current_dir', default='./', type=str)
parser.add_argument('--model', default='Buddhi6', type=str)

args = parser.parse_args()
