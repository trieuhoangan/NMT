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
from decoder import Decoder,Attn
from train import trainEpoch,trainIters,train
import gensim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normal_train():
    input_size = 100
    hidden_size = 100
    p_dropout = 0.01
    max_length = 870
    path_to_file_vi = '../vi_model.bin'
    path_to_file_en = '../en_model.bin'
    en_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_en,binary=True)
    vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)

    enc = Tree2SeqEncoder(input_size,hidden_size,max_length,p_dropout,path_to_file_vi).to(device)
    dec = Decoder(input_size,hidden_size,max_length,path_to_file_en,hidden_size,len(en_model.vocab)).to(device)
    
    input_data_path = '../train.vi'
    target_data_path = '../train.en'
    input_forest_path = '../train_phonlp_token_list.txt'
    epoch = 15
    save_path = 'training'
    trainEpoch(enc,dec,input_data_path,target_data_path,input_forest_path,epoch,0,0,save_path)
def train_from_checkpoint():
    path = 'training/checkpoint.pt'
    checkpoint = torch.load(path)
    input_size = 100
    hidden_size = 100
    p_dropout = 0.01
    max_length = 870
    enc = Tree2SeqEncoder(input_size,hidden_size,max_length,p_dropout,path_to_file_vi).to(device)
    dec = Decoder(input_size,hidden_size,max_length,path_to_file_en,hidden_size,len(en_model.vocab)).to(device)
    enc.load_state_dict(checkpoint['enc_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])
    last_epoch = checkpoint['epoch']
    last_iter = checkpoint['iter']
    if iter == 0:
    last_epoch = last_epoch + 1
    input_data_path = '../DataForNMT/2013/train-en-vi/train.vi'
    target_data_path = '../DataForNMT/2013/train-en-vi/train.en'
    input_forest_path = '../DataForNMT/2013/train_phonlp_token_list.txt'
    epoch = 17

    save_path = '../models/NMTmodels/training'
    trainEpoch(enc,dec,input_data_path,target_data_path,input_forest_path,epoch,last_epoch,last_iter,save_path)
if __name__=='__main__':
    