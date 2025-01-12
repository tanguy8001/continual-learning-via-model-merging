import pdb

from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
from rnn_utils import *
import codecs


class ShakeSpeareWordDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.dataset = torch.from_numpy(np.loadtxt(data_path,delimiter=','))

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        sentence_id_idx = self.dataset[idx][1:] # size: (1,100)
        id_idx = self.dataset[idx][0]
        return sentence_id_idx
    

class ShakeSpeareCharDataset(Dataset):
    def __init__(self, x_path,y_path):
        super().__init__()
        f_x = open(x_path, "r")
        self.dataset_x_raw = [lines[:-1] for lines in f_x] # get rid of the /n at the end of each line

        f_y = open(y_path, "r")
        self.dataset_y_raw = [lines[:-1] for lines in f_y]

    def __len__(self):
        return len(self.dataset_x_raw)

    def __getitem__(self, idx):
        
        x_idx = word_to_index(self.dataset_x_raw[idx])
        y_idx = letter_to_index(self.dataset_y_raw[idx])
        # y_vec = letter_to_vec(self.dataset_y_raw[idx])
        
        return x_idx,y_idx
    

class HomogeneousMNIST(Dataset):
    def __init__(self, data_path,transform=None):
        data_file = np.genfromtxt(data_path,skip_header=0,dtype=float,delimiter=',')
        self.dataset = torch.from_numpy(data_file)#.double()
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x_idx = self.dataset[idx,1:].float().reshape(28,28) # convert the size of input to (28,28)
        if not self.transform is None:
            x_idx = self.transform(x_idx.unsqueeze(0)).squeeze()
        label_idx = self.dataset[idx,0].long()

        return x_idx,label_idx


def load_glove_embeddings(glove_path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    print('loading word embeddings from %s' % glove_path)
    weight_vectors = []
    word_idx = {}
    with codecs.open(glove_path, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            word_idx[word] = len(weight_vectors)
            weight_vectors.append(np.array(vec.split(), dtype=np.float32))
    word_idx['<PAD>'] = len(weight_vectors)
    weight_vectors.append(np.zeros(weight_vectors[0].shape).astype(np.float32))
    word_idx['<UNK>'] = len(weight_vectors)
    weight_vectors.append(np.random.uniform(
      -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
    # padding 
    return np.stack(weight_vectors), word_idx


class SSTDataset(Dataset):
    def __init__(self, data_path,glove_path,max_seq_len=56):
        self.dataset = []
        with open(data_path,'r') as data_file:
            for line in data_file:
                item = line.split('|')
                phrase = item[-1].split(" ")
                label = int(item[0])
                self.dataset.append([label,phrase]) # list of lists of splitted words
        self.max_seq_len = max_seq_len
        self.weight_matrix, self.word_idx = load_glove_embeddings(glove_path) # (20727, 300)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        word_tokens_idx = self.dataset[idx][1] # a list of splitted words     
        label_idx = torch.tensor(self.dataset[idx][0]).long()   
        # add padding    
        while len(word_tokens_idx) < self.max_seq_len:
            word_tokens_idx = ['<PAD>'] + word_tokens_idx
        seq_voacb_idx = np.zeros((self.max_seq_len,self.weight_matrix.shape[0])) # shape: (sequence length, vocab size)
        for i in range(self.max_seq_len):
            try:
                word_id= self.word_idx[word_tokens_idx[i]]
            except KeyError:
                word_id = self.word_idx['<UNK>']
            seq_voacb_idx[i,word_id] = 1
        seq_input_idx = seq_voacb_idx @ self.weight_matrix # shape: (sequence length, input size)
        seq_input_idx = torch.from_numpy(seq_input_idx).float()
        return seq_input_idx,label_idx
    

"""
Testing
"""
# if __name__ == "__main__":
#     dataset = ShakeSpeareDataset("shakespeare/test.csv")
#     loader = DataLoader(dataset,batch_size=10)
#     for i_batch, samples_batched in enumerate(iter(loader)):
#         if i_batch <3:
#             print(samples_batched.size())
#             print(samples_batched.dtype)

if __name__ == "__main__":
    dataset = HomogeneousMNIST("./data/homogeneous_MNIST/mnist_train_0.csv")
    loader = DataLoader(dataset, batch_size=10)
    pdb.set_trace()
    # for i_batch, samples_batched in enumerate(iter(loader)):
    #     if i_batch <3:
    #         x_batched,y_batched = samples_batched
    #         print(x_batched.size())
    #         print(x_batched.dtype)
    #         print(y_batched.size())
    #         print(y_batched.dtype)
    
    
    # dataset = SSTDataset(data_path='./SST/train_phrase_label_small_2.txt',glove_path='./SST/filtered_glove.txt')
    # print(len(dataset))
    # loader = DataLoader(dataset,batch_size=2)

    # for i_batch, samples_batched in enumerate(iter(loader)):
    #     if i_batch <3:
    #         x_batched,y_batched = samples_batched
    #         print(x_batched.size())
    #         print(x_batched.dtype)
    #         print(y_batched.size())
    #         print(y_batched.dtype)
    #     else:
    #         break

