from logging import raiseExceptions

from torch._C import dtype
import tensorflow_datasets
import torch
import csv
import pickle
import numpy as np

import tensorflow.compat.v1 as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
# import sys
# import os

# import unicodedata
# import string
# import glob

data_path = "./datasets/"

class RNNDataset():
    def __init__(self,device,batch_size,eval_batch_size,dataset_name,train_data_path,*args, **kwargs):

        self.batch_size = batch_size 
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.dataset_name = dataset_name
        # self.train_batch_buffer = None
        # self.test_batch_buffer = None
        # self.train_enum = None
        # self.test_enum = None

        def create_tf_dataset(train_data,test_data,batch_size,eval_batch_size):
        #convert list to tensorflow Dataset
            self.train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
            self.test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

            self.train_dataset = self.train_dataset.shuffle(
                buffer_size=20000, reshuffle_each_iteration=True).batch(batch_size)
            self.test_dataset = self.test_dataset.batch(eval_batch_size)

            # self.train_dataset = self.train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
            # self.test_dataset = self.test_dataset.prefetch(tf.data.experimental.AUTOTUNE)


        # TODO: load dataset from file
        if dataset_name == 'test_binary_seq':
            create_tf_dataset(train_data=np.genfromtxt(data_path + dataset_name + '/train.csv', delimiter=',').tolist(),test_data=np.genfromtxt(data_path + dataset_name + '/test.csv', delimiter=',').tolist(),batch_size=batch_size,eval_batch_size=eval_batch_size)

        elif dataset_name == 'names':
            with open(data_path + dataset_name + '/parmas.json', 'rb') as fp:
                self.params = pickle.load(fp)
            # with open(data_path + dataset_name+'/train.csv', newline='') as f:
            print("training dataset path",data_path + dataset_name+ train_data_path)
            with open(data_path + dataset_name+ train_data_path, newline='') as f:
                reader = csv.reader(f)
                train_idx = list(reader)
            train_idx = [[int(float(ele)) for ele in lis] for lis in train_idx]
            with open(data_path + dataset_name+'/test.csv', newline='') as f:
                reader = csv.reader(f)
                test_idx = list(reader)
            test_idx =  [[int(float(ele)) for ele in lis] for lis in test_idx]

            # dataset shape(num_samples,n_letters,max_len)
            train_names_vectors = []
            for i in range(len(train_idx)):
                arr_temp = np.zeros((self.params['n_letters'],self.params['max_len']))
                arr_temp[0,0] = train_idx[i][0]
                for j in range(1,len(train_idx[i])):
                    arr_temp[int(train_idx[i][j]),j] = 1
                train_names_vectors.append(arr_temp.tolist())

            test_names_vectors = []
            for i in range(len(test_idx)):
                arr_temp = np.zeros((self.params['n_letters'],self.params['max_len']))
                arr_temp[0,0] = test_idx[i][0]
                for j in range(1,len(test_idx[i])):
                    arr_temp[int(test_idx[i][j]),j] = 1
                test_names_vectors.append(arr_temp.tolist())
            create_tf_dataset(train_data=train_names_vectors,test_data=test_names_vectors,batch_size=batch_size,eval_batch_size=eval_batch_size)
        else:
            raise NotImplementedError

        print("Finished initializing dataset" + self.dataset_name)

    def start_train_epoch(self):
        self.train_epoch_ends = False
        self.train_enum = iter(tensorflow_datasets.as_numpy(self.train_dataset))
        self.train_batch_buffer = next(self.train_enum, None)

    def start_test_epoch(self):
        self.test_epoch_ends = False
        self.test_enum = iter(tensorflow_datasets.as_numpy(self.test_dataset))
        self.test_batch_buffer = next(self.test_enum, None)
        
    def get_batch(self,train=True):
        if train:
            batch = self.train_batch_buffer
            self.train_batch_buffer = next(self.train_enum, None)
            if self.train_batch_buffer is None:
                self.train_epoch_ends = True
        else:
            batch = self.test_batch_buffer
            self.test_batch_buffer = next(self.test_enum, None)
            if self.test_batch_buffer is None:
                self.test_epoch_ends = True
                

        if self.dataset_name == 'test_binary_seq':
            x = torch.from_numpy(np.expand_dims(batch[:,1:].T,-1)) # torch.Size([10, 2, 1])
            x = x.to(device=self.device)
            y = torch.from_numpy(batch[:,-1:].T) # torch.Size([1, 2])
            y = y.to(device=self.device)
            return x,y
        elif self.dataset_name == 'names':
            # batch.shape = (batch_size,n_letters,max_len)
            # required input dim in RNN = (sequecne len,batch_size,input size) = (max_len,batch_size,n_letters)
            batch = np.transpose(batch,(2,0,1)) ## batch.shape = (max_len,batch_size,n_letters)
            x = torch.from_numpy(batch[1:,:,:]) 
            x = x.to(device=self.device)

            # output shape = (max_len,batch_size,hidden_size), last output = (1,batch_size,hidden_size)
            y = torch.from_numpy(batch[0,:,0]).long()
            # y = np.zeros((1,batch.shape[1],len(self.params['all_categories'])))
            # for i in range(y_idx.shape[0]):
            #     for j in range(y_idx.shape[1]):
            #         y[i,j,int(y_idx[i,j,0])] = 1
            # y = torch.from_numpy(y)        
            y = y.to(device=self.device)
            return x,y


        # return x,y
        return None

    

    
"""
for testing purpose
"""
# if __name__ == "__main__":
#     dataset = RNNDataset(device='cpu',batch_size=2,eval_batch_size=2,dataset_name='names')
#     x,y = dataset.get_batch(True)
#     print("type of x:",type(x))
#     print(x.size())
#     print("type of y:",type(y))
#     print(y.dtype)
