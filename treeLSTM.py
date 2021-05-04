import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
from PhoNode import Tree_,PhoNode
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryTreeLSTMCell(nn.Module):
  '''
    TreeLSTMCell is defined based on the format of LSTM function of torch.nn
    TreeLSTMCell receive input as a batch of tree and calculate the state of each node in the tree
  '''
  def __init__(self,input_size, hidden_size,MAX_LENGTH,embedding,p_dropout):
    super(BinaryTreeLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = nn.Dropout(p=p_dropout)
    self.embedding = embedding
    self.max_length = MAX_LENGTH
    # self.U_i_l = nn.Parameter(
    #         torch.Tensor(hidden_size, hidden_size),
    #         requires_grad=True)
    self.U_i_l = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.U_i_r = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.b_i = nn.Parameter(
            torch.ones(hidden_size, 1),
            requires_grad=True).to(device)
    self.U_fl_l = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.U_fl_r = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.b_fl = nn.Parameter(
            torch.ones(hidden_size, 1),
            requires_grad=True).to(device)
    self.U_fr_l = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.U_fr_r = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.b_fr = nn.Parameter(
            torch.ones(hidden_size, 1),
            requires_grad=True).to(device)
    self.U_o_l = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.U_o_r = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.b_o = nn.Parameter(
            torch.ones(hidden_size, 1),
            requires_grad=True).to(device)
    self.U_c_l = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.U_c_r = nn.Parameter(
            torch.ones(hidden_size, hidden_size),
            requires_grad=True).to(device)
    self.b_c = nn.Parameter(
            torch.ones(hidden_size, 1),
            requires_grad=True).to(device)
  '''
    input: root node of the tree
      return:
        output at shape (T,B,H*num_direction) ( in this case num_direction is 1 so shape is T*B*H)
        Tuble (h,c) of shape (num_layer*numdirection,B,H) in this case (1,B,H)
  '''
  def forward(self,input_forest):
    forest_output = None
    forest_h = None
    forest_c = None
    for tree in input_forest:
      tree_output,(tree_h,tree_c) = self.tree_traversal(tree)
      if tree_output is None:
        tree_output = torch.zeros(self.max_length,self.hidden_size)
        tree_h = torch.zeros(1,self.hidden_size)
        tree_c = torch.zeros(1,self.hidden_size)
      else:
        tree_output = self.widen_output(tree_output)
      if forest_output is None:
        forest_output = tree_output.unsqueeze(0)
        forest_h = tree_h.unsqueeze(0)
        forest_c = tree_c.unsqueeze(0)
      else:
        forest_output = torch.cat((forest_output,tree_output.unsqueeze(0)),dim=0)
        forest_c = torch.cat((forest_c,tree_c.unsqueeze(0)),dim=0)
        forest_h = torch.cat((forest_h,tree_h.unsqueeze(0)),dim=0)
        
    return forest_output,(forest_h.transpose(0,1),forest_c.transpose(0,1))
  '''
    param input_left and input_right: 
       left input and right input state of the decoder, in shape (H,1)
    param c_k_left and c_k_right:
        memory cell of left node and right node in shape (H,1)
    return:
        h and c of current node in shape (H,1)
    the current state and memory are calculated as
              ik  = σ (U(i)l*h_k_left + U(i)r*h_k_right + b(i)
              flk = σ (U(fl)l*h_k_left + U(fl)r*h_k_right + b(fl))
              frk = σ (U(fr)l*h_k_left + U(fr)r*h_k_right + b(fr))
              ok  = σ (U(o)l*h_k_left + U(o)r*h_k_right + b(o))
              ck = tanh(U(c)l*h_k_left + U(c)r*h_k_right + b(c))
              c(phr)k = ik

  '''
  def calculate(self,input_left,input_right,c_k_left, c_k_right):
    i = torch.sigmoid((torch.matmul(self.U_i_l,input_left) + torch.matmul(self.U_i_r,input_right) + self.b_i))
    f_k_left = torch.sigmoid((torch.matmul(self.U_fl_l,input_left) + torch.matmul(self.U_fl_r,input_right) + self.b_fl))
    f_k_right = torch.sigmoid((torch.matmul(self.U_fr_l,input_left) + torch.matmul(self.U_fr_r,input_right) + self.b_fr))
    o_k = torch.sigmoid((torch.matmul(self.U_o_l,input_left) + torch.matmul(self.U_o_r,input_right) + self.b_o))
    c_k = torch.tanh((torch.matmul(self.U_c_l,input_left) + torch.matmul(self.U_c_r,input_right) + self.b_c))
    c_k_phr = i*c_k + f_k_left*c_k_left + f_k_right*c_k_right
    h_k_phr = o_k*torch.tanh(c_k_phr)
    return h_k_phr,c_k_phr
  
  def tree_traversal(self,node):
    '''
      input : a single root node of a tree
          tree_traversal function handle 1 tree each time it is called
      return: output is at shape(T,1,H*num_directions ) in this case num_directions  = 1 <=> (T,1,H) => (T,H) to ease
      tuple(h,c) is at shape (num_layers * num_directions,1,H) <=> (1,H)
    '''
    if node is None:
      return torch.zeros(self.max_length,self.hidden_size).to(device), (torch.zeros(1,self.hidden_size).to(device), torch.zeros(1,self.hidden_size).to(device))
    tmp = copy.copy(node)
    output = None
    hs = []
    while tmp.h is None:
      if tmp.left is None and tmp.right is None:
        if len(tmp.part) > 0:
          if tmp.part[0].word_index !=-1:
            h = self.embedding(torch.Tensor([tmp.part[0].word_index]).to(torch.int64).to(device))
            h = torch.transpose(h, 0, 1)
          else:
            h = torch.ones(self.hidden_size,1).to(device)- 0.5
        else:
          h = torch.ones(self.hidden_size,1).to(device) -0.5
        c_phr = torch.ones(self.hidden_size, 1).to(device) - 0.5
        tmp.h = h
        tmp.c = c_phr
        tmp = tmp.father
        continue
      if tmp.left is not None:
        if tmp.left.h is None:
          tmp = tmp.left
          continue
      if tmp.right is not None:
        if tmp.right.h is None:
          tmp = tmp.right
          continue
      h,c = self.calculate(tmp.left.h,tmp.right.h,tmp.left.c,tmp.right.c) # Hx1
      # print(h.shape)
      tmp.h = h
      tmp.c = c
      hs.append(h)
      if tmp.father is not None:
        tmp = tmp.father
    if len(hs) == 0:
      output = None
      print(None)
    else:
      output = hs[0].transpose(0,1)
      for i in range(len(hs)):
        output = torch.cat((output,hs[i].transpose(0,1)),dim=0)
      # print(output.shape)
    return output, (tmp.h.transpose(0,1).to(device), tmp.c.transpose(0,1).to(device))
  def widen_output(self,output):
    while output.shape[0] < self.max_length:
      output = torch.cat((output,torch.zeros(1,self.hidden_size).to(device)),dim=0)
    return output
  def get_inithidden(self):
    return torch.ones((self.hidden_size,1)).to(device)