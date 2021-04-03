import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
from preprocess import preprocessing,tensorFromSentence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchtext.data.metrics import bleu_score
'''
  sentence is a single Vietnamese sentence
'''
class Evaluator:
  def __init__(self):
    return None
  def get_words_from_index(self,listword,model):
    listvocab = list(model.vocab)
    result = []
    # print(len(listword))
    for word in listword:
      result.append(listvocab[word])
    # print(len(result))
    return result
  def pre_translate(self,input_tensor,encoder,decoder):
    encoder_seq_output,encoder_seq_hc,encoder_tree_hc = encoder(input_tensor.to(device))
    word_input = [en_model.vocab['<start>'].index]
    decoder_input = torch.Tensor(word_input).to(torch.int64).to(device)
    # print('first_input ',decoder_input)
    decoder_hidden = decoder.get_first_hidden(encoder_seq_hc[0],encoder_seq_hc[1])
    output = []
    while decoder_input.item()!=en_model.vocab['<end>'].index:
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_seq_output.transpose(0,1))
        '''
          decoder is in shape (B,H)
          mean first word of each sentence.
        '''
        # print('a')
        topv, topi = decoder_output.topk(1)
        # print(decoder_output.topk(2))
        decoder_input = topi.squeeze(0).detach()  # detach from history as input
        # print('\n last input ',decoder_input)
        output.append(decoder_input[0].item())
    # print(len(output))
    output = get_words_from_index(output,en_model)
    return output
  def translate(self,sentence,encoder,decoder):
    sentence = preprocessing(sentence)
    input_tensor,leng = tensorFromSentence(vi_model,[sentence],870)
    output = pre_translate(input_tensor,input_forest,encoder,decoder)
    return output
  '''
    input_test is list of sentence in real sentence
    input_forest is list of parsed tree 
  '''
  def evaluate(self,enc,dec,input_test,target_test):
    limit = len(input_test)
    results = []
    for i in range(limit):
      result = pre_translate(input_test[i].unsqueeze(0),enc,dec)
      results.append(result)
    bleuscore = bleu_score(results,target_test)
    return bleuscore
