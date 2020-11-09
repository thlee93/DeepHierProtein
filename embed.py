import os
import time
import argparse
from collections import defaultdict

import torch
import numpy as np
import pandas as pd

from model import HierMetric
from dataset import DataSet
from utils import *


def argparse_embed():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_len', type=int, default=1000)
    parser.add_argument('--charset_size', type=int, default=20)
    parser.add_argument('--data_dir', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/data/taeheon/1809_Hierarchical')
    parser.add_argument('--num_cuda', type=int, default=0)
    parser.add_argument('--num_seq', type=int, default=100)
    parser.add_argument('--cv_num', type=int, default=0)
    parser.add_argument('--num_repeat', type=int, default=100)
    parser.add_argument('--size_data', type=int, default=None)

    flags, _ = parser.parse_known_args()
    return flags


def embed_database(model, dataset, device, embed_csv):
    model.eval()

    label_cls = []
    label_fam = []
    label_sub = []

    pred_cls = []
    pred_fam = []
    pred_sub = []

    embed = []

    vec2seq = dict()

    with torch.no_grad():
        for data, cls, fam, sub in dataset.iter_once( 100 ):
            data, cls, fam, sub = torch.tensor(data, device=device), \
                                  torch.tensor(cls, device=device, dtype=torch.long).max(dim=1)[1], \
                                  torch.tensor(fam, device=device, dtype=torch.long).max(dim=1)[1], \
                                  torch.tensor(sub, device=device, dtype=torch.long).max(dim=1)[1]
            data = data.permute(0, 2, 1)
            pred1, pred2, pred3, hidden = model(data)
            pred1 = pred1.argmax(dim=1, keepdim=True)
            pred2 = pred2.argmax(dim=1, keepdim=True)
            pred3 = pred3.argmax(dim=1, keepdim=True)

            pred_cls = pred_cls + list(pred1.flatten().cpu().detach().numpy())
            pred_fam = pred_fam + list(pred2.flatten().cpu().detach().numpy())
            pred_sub = pred_sub + list(pred3.flatten().cpu().detach().numpy())

            label_cls = label_cls + list(cls.flatten().cpu().detach().numpy())
            label_fam = label_fam + list(fam.flatten().cpu().detach().numpy())
            label_sub = label_sub + list(sub.flatten().cpu().detach().numpy())

            hidden = hidden.cpu().detach().numpy()
            embed.append(hidden)
        
        embed = np.concatenate(embed, axis=0)
        df = pd.DataFrame(embed)
        df['label_cls'] = label_cls
        df['label_fam'] = label_fam
        df['label_sub'] = label_sub
        df['pred_cls'] = pred_cls
        df['pred_fam'] = pred_fam
        df['pred_sub'] = pred_sub

        df.to_csv(embed_csv)

    return df


def find_nearest(model, test_dataset, num_seq, device, output_json):



def main():
    flags = argparse_embed()
    output_root = flags.output_dir
    data_num = flags.data_dir
    num_cuda = flags.num_cuda
    char_size = flags.charset_size
    seqlen = flags.seq_len
    num_seq = flags.num_seq
    num_repeat = flags.num_repeat
    size_data = flags.size_data
    cv_num = flags.cv_num

    os.environ['CUDA_VISIBLE_DEVICES'] = str(num_cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available else 'cpu')

    if cv_num :
        cv_start, cv_end = cv_num, cv_num+1
    else :
        cv_start, cv_end = 0, 10

    for data_num in range(data_num, data_num+1):
        data_dir = os.path.join('data', 'data{}'.format(data_num))

        for cv in range(cv_start, cv_end):
            record = defaultdict(list)

            for idx in range(num_repeat):
                start_time = time.time()

                data_root = os.path.join(data_dir, 'total', 'cv_{}'.format(cv))
                train_data = os.path.join(data_root, 'train.txt')
                val_data = os.path.join(data_root, 'val.txt')
                test_data = os.path.join(data_root, 'test.txt')

                result_dir = os.path.join(output_root, 'data{}_cv{}'.format(data_num, cv))
                result_csv = os.path.join(result_dir, 'embed.csv')
                output_json = os.path.join(output_root, 'query_search.json')
                time_record = os.path.join(output_root, 'time_record.csv')
                mkdir(result_dir)

                model = HierMetric(char_size)
                model = model.to(device)

                train_dataset = DataSet(train_data, seqlen, size_data=size_data)
                val_dataset = DataSet(val_data, seqlen)
                test_dataset = DataSet(test_data, seqlen)

                database = embed_database(model, train_dataset, device, result_csv)
                embed_time = time.time()

                find_nearest(model, test_dataset, num_seq, device, output_json)
                end_time = time.time()

                embed_database(model, test_dataset, device, os.path.join(result_dir, 'test_embed.csv'))

                record['start'].append(start_time)
                record['embed'].append(embed_time)
                record['end'].append(end_time)
                record['datasize'].append(size_data)


if __name__ == '__main__':
    main()


