import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
from PhoNode import Tree_,PhoNode
import preprocess
from treeLSTM import BinaryTreeLSTMCell
from encoder import Tree2SeqEncoder
from decoder import Decoder,Attn,NewDecoder
from train import trainEpoch,trainIters,train
import gensim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normal_train(enc,dec,args):
    save_path = 'models/checkpoint'
    trainEpoch(enc,dec,args,0,0,save_path)
def train_from_checkpoint(enc,dec,args):
    path = 'models/checkpoint/checkpoint.pt'
    checkpoint = torch.load(path)
    enc.load_state_dict(checkpoint['enc_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])
    last_epoch = checkpoint['epoch']
    last_iter = checkpoint['iter']
    if iter == 0:
        last_epoch = last_epoch + 1
    save_path = 'models/checkpoint'
    trainEpoch(enc,dec,args,last_epoch,last_iter,save_path)
if __name__=='__main__':
    input_size = 100
    hidden_size = 100
    p_dropout = 0.01
    max_length = 870
    path_to_file_vi = 'models/language_models/vi_model.bin'
    path_to_file_en = 'models/language_models/en_model.bin'
    input_data_path = 'data/train.vi'
    target_data_path = 'data/train.en'
    input_forest_path = 'data/tree_train.txt'
    input_valid_path = 'data/valid.vi'
    target_data_path =  'data/valid.en'
    valid_forest_path = 'data/valid_tree.txt'
    epoch = 15
    en_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_en,binary=True)
    vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)
    enc = Tree2SeqEncoder(input_size,hidden_size,max_length,p_dropout,path_to_file_vi).to(device)
    dec = NewDecoder(input_size,hidden_size,max_length,path_to_file_en,hidden_size,len(en_model.vocab)).to(device)
    args = {"input_data_path":input_data_path,
            "target_data_path":target_data_path,
            "input_forest_path":input_forest_path,
            "epoch":epoch,
            "input_valid_path":input_valid_path,
            "target_valid_path":target_valid_path,
            "valid_forest_path":valid_forest_path
            }
    train_from_checkpoint(enc,dec,args)