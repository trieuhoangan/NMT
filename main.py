import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
import PhoNode
import preprocess
from encoder import Tree2SeqEncoder
from decoder import Decoder,Attn
import gensim

if __name__=='__main__':
    input_size = 100
    hidden_size = 100
    p_dropout = 0.01
    max_length = 870
    path_to_file_vi = '../models/language_models/vi_model.bin'
    path_to_file_en = '../models/language_models/en_model.bin'
    en_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_en,binary=True)
    vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)

    enc = Tree2SeqEncoder(input_size,hidden_size,max_length,p_dropout,path_to_file_vi).to(device)
    dec = Decoder(input_size,hidden_size,max_length,path_to_file_en,hidden_size,len(en_model.vocab)).to(device)
    # target_data_path = '../DataForNMT/2013/dev-2012-en-vi/tst2012.en'
    # input_data_path = '../DataForNMT/2013/dev-2012-en-vi/tst2012.vi'
    # input_forest_path = '../DataForNMT/2013/dev_phonlp_token_list.txt'
    input_data_path = '../DataForNMT/2013/train-en-vi/train.vi'
    target_data_path = '../DataForNMT/2013/train-en-vi/train.en'
    input_forest_path = '../DataForNMT/2013/train_phonlp_token_list.txt'
    epoch = 15
    save_path = '../models/NMTmodels/training'
    trainEpoch(enc,dec,input_data_path,target_data_path,input_forest_path,epoch,0,0,save_path)