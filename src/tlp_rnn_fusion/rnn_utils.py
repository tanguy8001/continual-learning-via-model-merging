import re
import numpy as np
import torch

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec

def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)

def word_to_index(raw_x):
    """
        Input: one sample of x - The lenght of each x sample is 80 (no further data processing is needed)
        return: a tensor of size(80)
    """
    indices = []
    for c in raw_x:
        indices.append(ALL_LETTERS.find(c))
    return torch.tensor(indices)

def letter_to_index(letter):
    return torch.tensor(ALL_LETTERS.find(letter))

def letter_to_vec(raw_y):
    """
        Input: one sample of y
        return: a tensor of size(len(ALL_LETTERS),1)
    """
    index = ALL_LETTERS.find(raw_y)
    return torch.tensor(_one_hot(index, NUM_LETTERS)).unsqueeze(-1)
