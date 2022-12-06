# -*- coding: utf-8 -*-

import sys
import numpy as np
import os
sys.path.insert(0,'/multi-stage-malaria-graph-ml')
import distance_est
import graph_fun
import config
import custom_data_generator
import loss_funs
import utils_fun

import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans
from sklearn import preprocessing
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator



import spektral

from tensorflow.keras.optimizers import Adam

from sklearn.neighbors import kneighbors_graph
from graph_fun import process_graph
import numpy as np
import config
import graph_fun
import distance_est

from tensorflow import keras
from spektral.layers import GCNConv
#from spektral.models.gcn import GCN
from spektral.utils import gcn_filter, add_self_loops
from spektral.transforms import GCNFilter

from tensorflow.keras import activations

#a class that models the whole model, the cnn and cnn features, where the graph building code is called(for constructed a graph), and the gcn layers are formed
#
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
class Net(keras.Model):
    def __init__(self, num_classes=config.class_num):
        super(Net, self).__init__()
        self.base_cnn_model = keras.applications.vgg19.VGG19(weights='imagenet',input_shape=(224, 224, 3),include_top=False)
        self.base_cnn_model.trainable = False
        self.flat = keras.layers.Flatten()
        self.fc = keras.layers.Dense(512)
        self.GraphConv1=GCNConv(config.features_dim)
        self.act1=keras.layers.Activation(activations.relu)
        self.GraphConv2=GCNConv(config.GCN_hidderlayer_dim_num)
        self.act2=keras.layers.Activation(activations.relu)
    def call(self, x, source_length=0, source_labels=None, source_centers=None):
        x= self.base_cnn_model(x, training=False)
        x = self.flat(x)
        x = self.fc(x)

        features = np.array(x)
        if source_centers is None:
            source_centers = get_centers(features[:source_length], source_labels)
        target_labels = []
        for feature in features[source_length:]:
            dis_list = []
            for center in source_centers:
                dis_list.append(distance_est.euclidean_distance(feature, center))
            target_labels.append(np.argmin(dis_list))
        if source_labels is not None:
          #this returns the adjacency matrix
            adj = graph_fun.domain_cluster_graph(source_labels, target_labels, init_graph=None)
        else:
            adj = graph_fun.supervised_graph(target_labels)

        if source_length > 0:
            adj[:source_length, :source_length] = np.eye(source_length)
        #this function below returns the adjacency and degree matrix

        # A, D = graph_fun.process_graph(adj)
        #change them into tensors
        # A, D = torch.tensor(A, dtype=torch.float32, requires_grad=True).cuda(), torch.tensor(D, dtype=torch.float32).cuda()
        #feed the graph convolution with these matrices, and the node features
        # x = self.gc1(x, A, D)
        # x = self.gc2(x, A, D)
        # return x, A, features
        A=add_self_loops(adj, value=1)
        normalized_A=gcn_filter(A, symmetric=True)
       
        x=self.GraphConv1([x, normalized_A])
        x=self.act1(x)
        x=self.GraphConv2([x,normalized_A])
        x=self.act2(x)
        return x, normalized_A ,features



#to get the class centers for the source domain
def get_centers(features, labels):
    #labels=np.where(labels==1)[1]
    # enc = OneHotEncoder(handle_unknown='ignore')
    # enc.inverse_transform(labels)
    centers = np.zeros([max(labels) + 1, features.shape[1]])
    for i in range(len(labels)):
        centers[labels[i]] += features[i]
    return centers

model=Net()

# model.compile(optimizer = Adam(learning_rate=0.0001, decay = 0.001 / 64))

"""training **bit**"""

source_augmentor = ImageDataGenerator(rotation_range=15,
                            shear_range = 0.1,
                            zoom_range = 0.05,
                            rescale=1.0/255.0,
                            horizontal_flip=True,
                            featurewise_center=True, 
                            featurewise_std_normalization=True,
                        
                            width_shift_range=0.05,
                            height_shift_range=0.05)
t_gen=ImageDataGenerator(rescale=1.0/255.0)

def generate_batch(source_data, target_data):
  #this iterates on the batches of the source_data and target_data
    source_iter = iter(source_data)
    target_iter = iter(target_data)
    #the source images to be put in an array?
    source_x1 = []
    #source labels
    source_labels1 = []
    #why a second one?
    source_x2 = []
    source_labels2 = []
    #target training images
    target_x = []
    target_labels = []
    #the input images and labels in the next batch?
    inputs, labels = source_iter.__next__()
    #append the inputs in this array
    source_x1.append(inputs)
    #print(str(np.shape(labels)))
    for label in labels:
        #print(label)
        source_labels1.append(label)

    inputs, labels = source_iter.__next__()
    #then for the next batch, append in the second source and label list
    source_x2.append(inputs)
    for label in labels:
        source_labels2.append(label)
    #do the same for the target batch
    inputs, labels = target_iter.next()
    target_x.append(inputs)
    for label in labels:
        target_labels.append(label)
    #concatenate along axis-0 and reassign
    source_x1 = np.concatenate(source_x1, 0)
    source_x2 = np.concatenate(source_x2, 0)
    target_x = np.concatenate(target_x, 0)
    #change them into tensors??
    source_labels1 = np.array(source_labels1)
    # source_labels1 = np.where(source_labels1==1)[1]
    source_labels2 = np.array(source_labels2)
    # source_labels2 = np.where(source_labels2==1)[1]
    target_labels = np.array(target_labels)
    target_labels= np.where(target_labels==1)[1]
    #what's the purpose of the code below??
    #commenting the below code under the assumption that it doesn't matter
    #source_labels = (source_labels1.numpy() == source_labels2.numpy()).astype('float32')
    return source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels

X=np.load('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/input_data.npy')
Y=np.load('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/input_label.npy')

# source_data_loader = datagen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/train', 
#                                        target_size=(224, 224), 
#                                        color_mode='rgb', 
#                                        class_mode='categorical', 
#                                        batch_size=config.source_batch_size)
# load and iterate validation dataset
source_balancer = custom_data_generator.BalancedDataGenerator(X, Y, source_augmentor, batch_size=30)
target_data_loader = t_gen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=config.target_batch_size)

test_data_loader = t_gen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test', 
                                      target_size=(224, 224), 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=config.target_batch_size)

class_labels=test_data_loader.class_indices
class_labels

type(test_data_loader)

len(target_data_loader)

#should be done in a loop
source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels = generate_batch(source_balancer, target_data_loader)



labels = np.concatenate((source_labels1, source_labels2))
source_length=len(source_x1) + len(source_x2)
outputs, A, res_features = model(np.concatenate((source_x1, source_x2, target_x), axis=0), source_length=source_length, source_labels=labels, training=True )

import time
import os
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

#con_loss = loss_funs.ContrastiveLoss()

#ContrastiveLoss(np.array(source_outputs1), np.array(source_outputs2), labels)

#type(np.array(source_outputs1))

#type(source_outputs2)

#type(labels)

# Commented out IPython magic to ensure Python compatibility.

#the constrastive(for discriminative features to be learnt from the source data) and mean squared error loss(for what though)
#con_loss = loss_funs.ContrastiveLoss()
domain_loss = tf.keras.losses.MeanSquaredError()

#defining the optimizer, might not need net.parameters()? but we will see
optimizer = keras.optimizers.Adam(learning_rate=1e-5, decay=5e-4)

#defining the scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

best_acc = 0
path='/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/Weights/msmodel.ckpt'
 
# save
# model.save_weights(path)

#for each of the epoches
for epoch in range(100):
    since = time.time()
    sum_loss1 = 0.
    sum_loss2 = 0.
    sum_loss3 = 0.
    #model.train()#train the model
    length = config.source_batch_size + config.target_batch_size
    dis_list = []
    source_centers_sum = np.zeros([config.class_num, 1024])#could 1024 be the max length to hold the whole class center feature
    source_res_centers_sum = np.zeros([config.class_num, 512])
    
    for i in range(227):
      #to generate the batch arrays 
       #should be done in a loop
        if i==111:
          break

        source_x1, source_x2, target_x, source_labels1, source_labels2, target_labels = generate_batch(source_balancer, target_data_loader)
        labels = np.concatenate((source_labels1, source_labels2))
        source_length=len(source_x1) + len(source_x2)

            #then wede device store yidiereg(temporary or in storage??)
        # source_x1, source_x2, target_x, source_labels = \
        #     source_x1.to(device), source_x2.to(device), target_x.to(device), source_labels.to(device)
            #empty the gradients for the next iteration
        with tf.GradientTape() as tape:
        #why concatenate
        #labels = np.concatenate((source_labels1.numpy(), source_labels2.numpy()))
        #forward the image arrays, the length, and the labels of the sources into the net
          
          outputs, A, res_features = model(np.concatenate((source_x1, source_x2, target_x), axis=0), source_length=source_length, source_labels=labels, training=True)
          #the above code returns the graph representation features, the normalized adjacency matrix, and the cnn features of both the source and the target respectively
          #this code below separates the graph representation features? into source outputs and a target output
          source_outputs1, source_outputs2, target_outputs = outputs[:len(source_labels1)], outputs[len(source_labels1):len(source_labels1) + len(source_labels2)], outputs[len(source_labels1) + len(source_labels2):]
          #the source resnet features
          source_res_features = res_features[:len(source_labels1) + len(source_labels2)]
          #the source graph representations into numpy, concatenate
          source_features = np.concatenate([source_outputs1, source_outputs2], axis=0)
          #and the labels associated with them, concatenate that too
          source_labels_numpy = np.concatenate([source_labels1, source_labels2], axis=0)
          #and the target features learn by the graph convolutions, change into numpy too
          target_features = target_outputs
          #and then the target labels(which weren't used for training)
          target_labels_numpy = target_labels
          #create a k means model with 6 clusters, for the source graph features(nodes)
          k_means_source = KMeans(config.class_num)
          #then fit it with the source features
          k_means_source.fit(source_features)
          #this calculates the accuracy score of the k-means model classification(into the 6 sub-classes)
          acc_source, _, _ = utils_fun.cluster_acc(source_labels_numpy, k_means_source.labels_, config.class_num)
          #do the same with the kmeans for the target
          k_means_target = KMeans(config.class_num)
          k_means_target.fit(target_features)
          #find the cluster centers after fitting it with the target_features data, i don't understand why it requires gradient classification though
          #target_label_centers = torch.tensor(k_means_target.cluster_centers_, requires_grad=True).to(device)
          #calculate the accuracy for the target data
          acc_target, target_labels_true, target_labels_pred = utils_fun.cluster_acc(target_labels_numpy, k_means_target.labels_, config.class_num)
          one_hot = preprocessing.OneHotEncoder(sparse=False, categories='auto')
          target_one_hot = one_hot.fit_transform(np.array(target_labels_pred).reshape(-1, 1))
          #what to do with the code below??
          target_labels_pred_torch = target_one_hot
          #target_soft_labels = F.softmax(torch.exp(torch.sqrt(torch.sum(torch.pow((target_outputs.unsqueeze(1) - target_label_centers), 2), 2))*-1), 1)

         
          loss1 = loss_funs.ContrastiveLoss(labels,np.array(source_outputs1), np.array(source_outputs2))
          # loss2 = mmd_loss(source_outputs1[:len(target_outputs)], target_outputs)
          source_outputs = outputs[:len(source_labels1) + len(source_labels2)]
          source_centers = np.mean(source_outputs, axis=0)
          source_res_centers = np.mean(source_res_features, axis=0)
          source_res_centers_sum += source_res_centers
          target_centers = np.mean(target_outputs, axis=0)
          loss2 = domain_loss(source_centers, target_centers)
         # loss3 = lmmd(source_outputs[:len(target_outputs)], target_outputs, source_labels1[:len(target_labels)].long(), target_soft_labels)
          lambda1 = 2/(1 + math.exp(-10/config.epoches)) - 1
          # loss4 = half_feature_matching_loss(A, outputs)
          loss = loss1 + loss2

          sum_loss1 += loss1
          sum_loss2 += loss2
          #sum_loss3 += loss3

        grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        

        iter_num = i + 1 + epoch * length
        print('[epoch:%d, iter:%d] Loss: %f | Acc_source: %f | Acc_target: %f | Loss_con: %f | Loss_domain: %f | time: %f')
#               % (epoch + 1, iter_num, loss, acc_source, acc_target, sum_loss1/iter_num, sum_loss2/iter_num, time.time() -  since))

    source_res_centers = source_res_centers_sum / 300
    test_features, _, test_res_features = utils_fun.get_features(test_data_loader,model, source_centers=source_res_centers)
    k_means_test = KMeans(config.class_num)
    k_means_test.fit(test_features)
    acc_test, labels_true, labels_pred = utils_fun.cluster_acc(target_labels, k_means_test.labels_, config.class_num)
    print('Test_acc:', acc_test, 'Time:', time.time() - since)
    # f = open('test_acc1.txt', 'a')
    # f.write(str(acc_test) + '\n')
    # f.close()
    if acc_test > best_acc:
        best_acc = acc_test
        np.save('features.npy', test_features)
        np.save('labels_true.npy', labels_true)
        np.save('labels_pred.npy', labels_pred)
        model.save_weights(path)
   # scheduler.step(epoch)

test_features_file=np.savetxt('G_feature_file.txt', np.array(test_features))
test_res_features_file=np.savetxt('test_res_feature_file.txt', np.array(source_res_centers))
from sklearn.metrics import classification_report
classification_report(labels_true, labels_pred, target_names=list(test_data_loader.class_indices.keys()))

model.load_weights('/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/Weights/msmodel.ckpt')

# test_data=iter(test_data_loader)
# input, DF=test_data.next()

# counter=0

# test_features, _, test_res_features = model(input, source_centers=source_res_centers, training=False)

# test_features, _, test_res_features = utils_fun.get_features(test_data_loader, source_centers=source_res_centers, training=False)

# test_features

# test_features, _, test_res_features = model(target_x, source_centers=source_res_centers, training=False)

# test_features

# features_list = []
# res_features_list = []
# labels_list = []

# for i,t_data in enumerate(test_data_loader):
#   t_inputs, t_labels = t_data
#   t_labels=np.where(t_labels==1)[1]
#   # inputs = inputs.cuda()
#   #net.eval()
#   ttest_features, _, ttest_res_features = model(t_inputs, source_centers=source_res_centers, training=False)
#   features_list.append(ttest_features)
#   res_features_list.append(ttest_res_features)
#   for label in t_labels:
#     labels_list.append(label)
#   print("extract the {} feature".format(i * config.target_batch_size))
#   if(i==59):
#     break
# test_features = np.concatenate(features_list, axis=0)
# test_res_features = np.concatenate(res_features_list, axis=0)
# labels = np.array(labels_list)
# if mode is None:
#     pass
# elif mode == 'train':
#     np.save('train_features.npy', features, allow_pickle=True)
#     np.save('train_labels.npy', labels, allow_pickle=True)
# elif mode == 'test':
#     np.save('test_features.npy', features, allow_pickle=True)
#     np.save('test_labels.npy', labels, allow_pickle=True)
# return features, labels, res_features

# test_res_features.shape

# test_features.shape

# len(features_list)

# len(labels_list)

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

import cv2

from PIL import Image

def preprocess_image(image):
  
  image = cv2.imread(image)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = Image.fromarray(image)
  image= config.test_transform(image)
  return image

image_path='/content/drive/MyDrive/Colab Notebooks/malaria-MS-classification/DTGCN_DATA/1_Multistage_Malaria_Parasite_Recognition/test/red_blood_cell/a1016_52.tif'

test_features_m=np.loadtxt('G_feature_file.txt')



def preprocess_img(image_path):
    # show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    # preds = model.predict(image)
    return image

res_centers=np.savetxt('res_centers.txt',source_res_centers)

source_res_centers_loaded=np.loadtxt('res_centers.txt')

def predict(model,image_path,test_features):
  image=preprocess_img(image_path)
  image_features,_,_=model(image, source_centers=source_res_centers_loaded,training=False)
  k_means_test = KMeans(6)
  k_means_test.fit(test_features)
  prediction=k_means_test.predict(image_features)
  for value in list(class_labels.values()):
    if prediction==value:
      prediction=list(class_labels.keys())[value]
    
  if (prediction==1 or prediction==2):
    status='uninfected'
  else:
    status='infected'
    
  return status,prediction

predict(model,image_path,test_features_m)

image=preprocess_img(image_path)
predict(model,image_path,)