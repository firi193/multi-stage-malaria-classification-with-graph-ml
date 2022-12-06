# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

#import neural_structured_learning as nsl

#from neural_structured_learning.keras.layers import pairwise_distance as pairwise_distance_lib

#regularizer = pairwise_distance_lib.PairwiseDistance()

from tensorflow.keras.losses import Loss

# import the necessary packages
import tensorflow.keras.backend as K
import tensorflow as tf
def contrastive_loss(y, preds, margin=1):
	# explicitly cast the true class label data type to the predicted
	# class label data type (otherwise we run the risk of having two
	# separate data types, causing TensorFlow to error out)
	y = tf.cast(y, preds.dtype)
	# calculate the contrastive loss between the true labels and
	# the predicted labels
	squaredPreds = K.square(preds)
	squaredMargin = K.square(K.maximum(margin - preds, 0))
	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
	# return the computed contrastive loss to the calling function
	return loss

class ContrastiveLoss(Loss):
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def call(self, Y_true, out_1, out_2):
        # torch.FloatTensor(out_1)
        euclidean_distance = euclidean_distances(out_1, out_2)
        Y_true = tf.cast(Y_true, euclidean_distances.dtype)
        #regularizer = pairwise_distance_lib.PairwiseDistance()
        margin_square=K.square(K.maximum(self.margin - euclidean_distance), 0)
        loss_contrastive = K.mean(Y_true * K.square(euclidean_distance) + (1 - Y_true)*margin_square)
        return loss_contrastive

loss=contrastive_loss(np.array([[1,2,3],[2,3,5]]), np.array([[1,2,4],[1,2,4]]))

loss(np.array([[1,2,3],[2,3,5]]), np.array([[1,2,4],[1,2,4]]), np.array([[2,2,3],[3,5,4]]))

