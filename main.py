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

if __name__=='__main__':
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