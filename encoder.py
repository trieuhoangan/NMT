import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
from treeLSTM import BinaryTreeLSTMCell
from PhoNode import Tree_,PhoNode
from node import Node,Tree
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Tree2SeqEncoder(nn.Module):
  def __init__(self,input_size,hidden_size,max_length,p_dropout,path_to_embedding):
    super(Tree2SeqEncoder,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.p_dropout = p_dropout
    self.max_length = max_length
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_embedding,binary=True)
    weights = torch.FloatTensor(model.vectors).to(device)
    self.embedding = nn.Embedding.from_pretrained(weights)
    self.TreeLSTMcell = BinaryTreeLSTMCell(input_size,hidden_size,max_length,self.embedding,p_dropout)
    self.LSTM = nn.LSTM(input_size, hidden_size,batch_first =True)
    
  
  '''
    input of sequence is shape of (B,T)
    input of Tree is shape (B)
    return : 
      output of tree in shape (B,T,H) and h,c in shape (1,B,H)
      output of sequence in shape (B,T,H) and h,c in shape (1,B,H)
  '''
  def forward(self,input_sequences, input_forest):
    input_sequences = self.embedding(input_sequences)
    output, hidden_of_sequence = self.LSTM(input_sequences)
    output_of_sequence = output
    c_of_sequence = hidden_of_sequence
    # print("finished compute sequence hidden")
    numNode,output_of_tree,c_of_tree = self.TreeLSTMcell(input_forest)
 
    return numNode,output_of_tree, output_of_sequence, c_of_tree,c_of_sequence