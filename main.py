import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math

from encoder import Encoder
from decoder import Decoder,Attn
from train import trainEpoch,trainIters,train
from evaluate import Evaluator
import gensim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(input_size = 100,
                hidden_size = 100,
                max_length = 870,
                p_dropout = 0.01,
                path_to_file_vi = 'models/vi_model.bin',
                path_to_file_en = 'models/en_model.bin',
                load_from_pretrained=True,path = 'trained/checkpoint.pt'):
    
    en_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_en,binary=True)
    vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)
    enc = Encoder(input_size,hidden_size,max_length,p_dropout,path_to_file_vi).to(device)
    dec = Decoder(input_size,hidden_size,max_length,path_to_file_en,hidden_size,len(en_model.vocab)).to(device)
    if load_from_pretrained == True:
        checkpoint = torch.load(path)
        enc.load_state_dict(checkpoint['enc_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
    return enc,dec
def normal_train():
    enc,dec = load_model(load_from_pretrained=False)
    input_data_path = 'data/train.vi'
    target_data_path = 'data/train.en'
    epoch = 15
    save_path = 'trained'
    trainEpoch(enc,dec,input_data_path,target_data_path,epoch,0,0,save_path)
def train_from_checkpoint():
    path = 'trained/checkpoint.pt'
    checkpoint = torch.load(path)
    enc,dec = load_model()
    last_epoch = checkpoint['epoch']
    last_iter = checkpoint['iter']
    if iter == 0:
        last_epoch = last_epoch + 1
    input_data_path = 'data/train.vi'
    target_data_path = 'data/train.en'
    epoch = 15
    save_path = 'models/trained'
    trainEpoch(enc,dec,input_data_path,target_data_path,epoch,last_epoch,last_iter,save_path)
if __name__=='__main__':
    enc,dec = load_model()
    path_to_file_vi = 'models/vi_model.bin',
    vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)
    test_data_en_path = 'data/test-en-vi/tst2013.en'
    test_data_vi_path = 'data/test-en-vi/tst2013.vi'
    test_text_en = open(test_data_en_path, 'rb').read().decode(encoding='utf-8')
    test_text_vi = open(test_data_vi_path, 'rb').read().decode(encoding='utf-8')
    test_vi_sentences = []
    test_en_sentences = []
    test_vi_sentences = test_vi_sentences.extend(test_text_vi.split('\n'))
    test_en_sentences = test_en_sentences.extend(test_text_en.split('\n'))
    test_vi = preprocess_batch(test_vi_sentences)
    test_en = preprocess_batch(test_en_sentences)
    test_tensor,lengs = tensorFromSentence(vi_model,test_vi,870)
    bleu = Evaluator.evaluate(enc,dec,test_tensor,test_en)
    print(bleu)