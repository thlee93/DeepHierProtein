import os
import argparse
import glob
from shutil import rmtree

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_len', type=int, default=1000)
    parser.add_argument('--charset_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--data_dir', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/data/taeheon/1809_Hierarchical')
    parser.add_argument('--num_cuda', type=int, default=0)

    FLAGS, _ = parser.parse_known_args()
    return FLAGS


def mkdir(path):
    os.makedirs(path, exist_ok=True)

def rmdir(path):
    rmtree(path, ignore_errors=True)
