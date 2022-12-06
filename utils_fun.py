# -*- coding: utf-8 -*-


import os
import numpy as np
# from torch.utils.data import DataLoader
# import torchvision.datasets as dataset
# import torch
from scipy.optimize import linear_sum_assignment
import tensorflow as tf

import sys
sys.path.insert(0,'/multi-stage-malaria-graph-ml')
import config


#eliminate all the torch in this code, replace it with sth else
def get_features(test_data, net,target_labels, mode=None, source_centers=None):
  features_list = []
  res_features_list = []
  labels_list = []

  for i,t_data in enumerate(test_data):
    t_inputs, t_labels = t_data
    t_labels=np.where(t_labels==1)[1]
    # inputs = inputs.cuda()
    #net.eval()
    ttest_features, _, ttest_res_features = net(t_inputs, source_centers=source_centers, training=False)
    features_list.append(ttest_features)
    res_features_list.append(ttest_res_features)
    for label in t_labels:
      labels_list.append(label)
    print("extract the {} feature".format(i * config.target_batch_size))
    if(i==59):
      break
  test_features = np.concatenate(features_list, axis=0)
  test_res_features = np.concatenate(res_features_list, axis=0)
  labels = np.array(labels_list)
          
  return test_features, labels, test_res_features



#this function doesn't use torch, which is great
def cluster_acc(y_true, y_pred, class_number):  
    cnt_mtx = np.zeros([class_number, class_number])

    # fill in matrix
    for i in range(len(y_true)):
        cnt_mtx[int(y_pred[i]), int(y_true[i])] += 1

    # find optimal permutation
    row_ind, col_ind = linear_sum_assignment(-cnt_mtx)

    # compute error
    acc = cnt_mtx[row_ind, col_ind].sum() / cnt_mtx.sum()

    labels_pred = []
    for index, label in enumerate(y_pred):
        target_label = col_ind[label]
        # print('label', label)
        # print('target', target_label)
        # print('true', y_true[index])
        labels_pred.append(target_label)
    # print(labels_pred[:10])
    # print(list(y_true)[:10])

    return acc, list(y_true), labels_pred