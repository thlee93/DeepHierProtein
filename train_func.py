import os
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd


def train(model, dataset, device, optimizer, epoch, centers, charse_size=20, seq_len=1000):
    model.train()
    step = 0
    total_loss = 0

    correct_cls = 0
    correct_fam = 0
    correct_sub = 0
    total_len = 0

    m_loss_cls = 0
    m_loss_fam = 0
    m_loss_sub = 0

    if epoch < 14:
        weight_s1 = 0.8
        weight_s2 = 0.15
        weight_s3 = 0.05
        weight_c = 0.01
    elif epoch < 30:
        weight_s1 = 0.2
        weight_s2 = 0.7
        weight_s3 = 0.1
        weight_c = 0.1
    else :
        weight_s1 = 0.1
        weight_s2 = 0.2
        weight_s3 = 0.7
        weight_c = 0.1
    
    if epoch < 5:
        weight_c = 0.0

    df = pd.DataFrame()

    for idx, (data, cls, fam, sub) in enumerate(dataset.iter_once( 100 )):
        data, cls, fam, sub = torch.tensor(data, device=device), \
            	              torch.tensor(cls, device=device, dtype=torch.long).max(dim=1)[1], \
            	              torch.tensor(fam, device=device, dtype=torch.long).max(dim=1)[1], \
            	              torch.tensor(sub, device=device, dtype=torch.long).max(dim=1)[1]

        data = data.permute(0, 2, 1)
        pred1, pred2, pred3, hidden = model(data)

        cls_centers = []
        fam_centers = []
        sub_centers = []

        if epoch > 2:
            for i in range(len(data)):
                cls_centers.append(centers['cls'][int(cls[i])])
                fam_centers.append(centers['fam'][int(fam[i])])
                sub_centers.append(centers['sub'][int(sub[i])])

            cls_centers = torch.from_numpy(np.array(cls_centers)).float().to(device)
            fam_centers = torch.from_numpy(np.array(fam_centers)).float().to(device)
            sub_centers = torch.from_numpy(np.array(sub_centers)).float().to(device)

            subtract_cls = torch.sqrt(torch.sum((hidden - cls_centers) * (hidden - cls_centers), dim=1))
            subtract_fam = torch.sqrt(torch.sum((hidden - fam_centers) * (hidden - fam_centers), dim=1))
            subtract_sub = torch.sqrt(torch.sum((hidden - sub_centers) * (hidden - sub_centers), dim=1))

            min_cls = torch.min(subtract_cls, torch.ones_like(subtract_cls)*1.8)
            min_fam = torch.min(subtract_fam, torch.ones_like(subtract_fam)*0.6)
            min_sub = torch.min(subtract_sub, torch.ones_like(subtract_sub)*0.1)

            m_loss_cls = torch.sum(min_cls)
            m_loss_fam = torch.sum(min_fam)
            m_loss_sub = torch.sum(min_sub)

        temp_df = pd.DataFrame(hidden.cpu().detach().numpy())
        temp_df['cls'] = cls.cpu().detach().numpy()
        temp_df['fam'] = fam.cpu().detach().numpy()
        temp_df['sub'] = sub.cpu().detach().numpy()
        df = df.append(temp_df)

        softmax_cls = F.log_softmax(pred1, dim=1)
        loss_cls = F.nll_loss(softmax_cls, cls)
        pred_cls = pred1.argmax(dim=1, keepdim=True)
        correct_cls += pred_cls.eq(cls.view_as(pred_cls)).sum().item()

        softmax_fam = F.log_softmax(pred2, dim=1)
        loss_fam = F.nll_loss(softmax_fam, fam)
        pred_fam = pred2.argmax(dim=1, keepdim=True)
        correct_fam += pred_fam.eq(fam.view_as(pred_fam)).sum().item()

        softmax_sub = F.log_softmax(pred3, dim=1)
        loss_sub = F.nll_loss(softmax_sub, sub)
        pred_sub = pred3.argmax(dim=1, keepdim=True)
        correct_sub += pred_sub.eq(sub.view_as(pred_sub)).sum().item()

        #loss_val = loss(output, labels)
        class_loss = weight_s1*loss_cls + weight_s2*loss_fam + weight_s3*loss_sub
        metric_loss = weight_s1*m_loss_cls + weight_s2*m_loss_fam + weight_s3*m_loss_sub 
        overall_loss = class_loss + metric_loss*weight_c

        optimizer.zero_grad()
        overall_loss.backward()
        optimizer.step()

        total_len += len(data)
        step += 1

    acc_cls = float(correct_cls)/total_len
    acc_fam = float(correct_fam)/total_len
    acc_sub = float(correct_sub)/total_len

    print("Epoch : %d started \t total number : %d" %(epoch, total_len)) 
    print("Class classification loss=%.2f, accuracy=%.4f" %(loss_cls, acc_cls))
    print("Family classification loss=%.2f, accuracy=%.4f" %(loss_fam, acc_fam))
    print("Subfamily classification loss=%.2f, accuracy=%.4f" %(loss_sub, acc_sub))

    return [acc_cls, acc_fam, acc_sub], [loss_cls, loss_fam, loss_sub], df


def test(model, dataset, device, epoch, embed_csv=None):
    model.eval()
    step = 0
    total_loss = 0

    correct_cls = 0
    correct_fam = 0
    correct_sub = 0
    total_len = 0

    label_cls = []
    label_fam = []
    label_sub = []

    pred_cls_list = []
    pred_fam_list = []
    pred_sub_list = []

    embed = []

    with torch.no_grad():
        for data, cls, fam, sub in dataset.iter_once( 100 ):
            data, cls, fam, sub = torch.tensor(data, device=device), \
                                  torch.tensor(cls, device=device, dtype=torch.long).max(dim=1)[1], \
                                  torch.tensor(fam, device=device, dtype=torch.long).max(dim=1)[1], \
                                  torch.tensor(sub, device=device, dtype=torch.long).max(dim=1)[1]
            data = data.permute(0, 2, 1)
            pred1, pred2, pred3, hidden = model(data)

            softmax_cls = F.log_softmax(pred1, dim=1)
            loss_cls = F.nll_loss(softmax_cls, cls)
            pred_cls = pred1.argmax(dim=1, keepdim=True)
            correct_cls += pred_cls.eq(cls.view_as(pred_cls)).sum().item()

            softmax_fam = F.log_softmax(pred2, dim=1)
            loss_fam = F.nll_loss(softmax_fam, fam)
            pred_fam = pred2.argmax(dim=1, keepdim=True)
            correct_fam += pred_fam.eq(fam.view_as(pred_fam)).sum().item()

            softmax_sub = F.log_softmax(pred3, dim=1)
            loss_sub = F.nll_loss(softmax_sub, sub)
            pred_sub = pred3.argmax(dim=1, keepdim=True)
            correct_sub += pred_sub.eq(sub.view_as(pred_sub)).sum().item()

            #loss_val = loss(output, labels)
            total_loss += (loss_cls.item() + loss_fam.item() + loss_sub.item())

            pred_cls_list = pred_cls_list + list(pred_cls.flatten().cpu().detach().numpy())
            pred_fam_list = pred_fam_list + list(pred_fam.flatten().cpu().detach().numpy())
            pred_sub_list = pred_sub_list + list(pred_sub.flatten().cpu().detach().numpy())

            label_cls = label_cls + list(cls.flatten().cpu().detach().numpy())
            label_fam = label_fam + list(fam.flatten().cpu().detach().numpy())
            label_sub = label_sub + list(sub.flatten().cpu().detach().numpy())

            hidden = hidden.cpu().detach().numpy()
            embed.append(hidden)

            total_len += len(data)
            step += 1

    acc_cls = float(correct_cls)/total_len
    acc_fam = float(correct_fam)/total_len
    acc_sub = float(correct_sub)/total_len

    print("Epoch : %d started \t total number : %d" %(epoch, total_len)) 
    print("Class classification loss=%.2f, accuracy=%.4f" %(loss_cls, acc_cls))
    print("Family classification loss=%.2f, accuracy=%.4f" %(loss_fam, acc_fam))
    print("Subfamily classification loss=%.2f, accuracy=%.4f" %(loss_sub, acc_sub))

    if not embed_csv:
        embed = np.concatenate(embed, axis=0)
        df = pd.DataFrame(embed)
        df['label_cls'] = label_cls
        df['label_fam'] = label_fam
        df['label_sub'] = label_sub
        df['pred_cls'] = pred_cls_list
        df['pred_fam'] = pred_fam_list
        df['pred_sub'] = pred_sub_list
        df.to_csv(embed_csv)

    return [acc_cls, acc_fam, acc_sub], [loss_cls, loss_fam, loss_sub]
        
    
def get_centers(df):
    center_dict = defaultdict(dict)
    center_dict['cls'] = defaultdict(dict)
    center_dict['fam'] = defaultdict(dict)
    center_dict['sub'] = defaultdict(dict)

    for i in range(5):
        center_dict['cls'][i] = np.array(df[df['cls']==i].mean())[:15]
    for i in range(40):
        center_dict['fam'][i] = np.array(df[df['fam']==i].mean())[:15]
    for i in range(86):
        center_dict['sub'][i] = np.array(df[df['sub']==i].mean())[:15]

    return center_dict

    


