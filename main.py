import os
import argparse
from collections import defaultdict
import glob

import torch
import torch.optim as optim
import pandas as pd

from model import HierMetric
from dataset import DataSet
from utils import *
from train_func import *


def main():
    # hyper-parameter setting and device configuration
    FLAGS = argparser()
    output_root = FLAGS.output_dir
    data_num = FLAGS.data_dir
    num_cuda = FLAGS.num_cuda
    char_size = FLAGS.charset_size
    seqlen = FLAGS.seq_len

    os.environ['CUDA_VISIBLE_DEVICES'] = str(num_cuda) 
    device = torch.device("cuda:0" if torch.cuda.is_available else 'cpu')

    for data_num in range(data_num, data_num+1):
        cls_dir = os.path.join(output_root, 'cls')
        fam_dir = os.path.join(output_root, 'fam')
        sub_dir = os.path.join(output_root, 'subfam')

        data_dir = os.path.join('data', 'data{}'.format(data_num))

        for cv in range(10):
            result_cls = os.path.join(cls_dir, 'data{}_cv{}'.format(data_num, cv))
            result_fam = os.path.join(fam_dir, 'data{}_cv{}'.format(data_num, cv))
            result_sub = os.path.join(sub_dir, 'data{}_cv{}'.format(data_num, cv))
            mkdir(result_cls)
            mkdir(result_fam)
            mkdir(result_sub)

            embed_csv = os.path.join(result_sub, 'embed.csv')

            # generate model and define optimizer and loss
            model = HierMetric(char_size)
            model = model.to(device)

            # initialize objects for each trial
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            centers = None

            # setting path for train and test data
            data_root = os.path.join(data_dir, 'total', 'cv_{}'.format(cv))
            train_data = os.path.join(data_root, 'train.txt')
            val_data = os.path.join(data_root, 'val.txt')
            test_data = os.path.join(data_root, 'test.txt')

            train_dataset = DataSet(train_data, seqlen)
            val_dataset = DataSet(val_data, seqlen)
            test_dataset = DataSet(test_data, seqlen)

            cls_dict = defaultdict(list)
            fam_dict = defaultdict(list)
            sub_dict = defaultdict(list)

            for epoch in range(1, FLAGS.max_epoch + 1):
                print("Epoch : {} started".format(epoch))
                train_acc, train_loss, embed_df = train(model, train_dataset, device, optimizer, epoch, centers)
                val_acc, val_loss = test(model, val_dataset, device, epoch)
                test_acc, test_loss = test(model, test_dataset, device, epoch, embed_csv)

                update_loss(cls_dict, train_loss[0], train_acc[0], test_loss[0], test_acc[0],
                            val_loss[0], val_acc[0])
                update_loss(fam_dict, train_loss[1], train_acc[1], test_loss[1], test_acc[1],
                            val_loss[1], val_acc[1])
                update_loss(sub_dict, train_loss[2], train_acc[2], test_loss[2], test_acc[2],
                            val_loss[2], val_acc[2])
                
                if epoch % 5 == 0 or epoch == 2:
                    centers = get_centers(embed_df)

                if epoch % 20 == 0:
                    df = pd.DataFrame(cls_dict)
                    df.to_csv(os.path.join(result_cls, 'loss.csv'))

                    df = pd.DataFrame(fam_dict)
                    df.to_csv(os.path.join(result_fam, 'loss.csv'))

                    df = pd.DataFrame(sub_dict)
                    df.to_csv(os.path.join(result_sub, 'loss.csv'))

            df = pd.DataFrame(cls_dict)
            df.to_csv(os.path.join(result_cls, 'loss.csv'))

            df = pd.DataFrame(fam_dict)
            df.to_csv(os.path.join(result_fam, 'loss.csv'))

            df = pd.DataFrame(sub_dict)
            df.to_csv(os.path.join(result_sub, 'loss.csv'))

            torch.save(model.state_dict(), os.path.join(result_sub, 'model.ckpt'))


def update_loss(loss_dict, train_loss, train_acc, test_loss, test_acc, val_loss, val_acc):
    loss_dict['train_loss'].append(train_loss)
    loss_dict['train_acc'].append(train_acc)
    loss_dict['test_loss'].append(test_loss)
    loss_dict['test_acc'].append(test_acc)
    loss_dict['val_loss'].append(val_loss)
    loss_dict['val_acc'].append(val_acc)


if __name__ == "__main__":
    main()
