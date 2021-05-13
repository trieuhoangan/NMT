import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
from PhoNode import Tree_, PhoNode
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gensim

class BinaryTreeLSTMCell(nn.Module):
    '''
      TreeLSTMCell is defined based on the format of LSTM function of torch.nn
      TreeLSTMCell receive input as a batch of tree and calculate the state of each node in the tree
    '''

    def __init__(self, input_size, hidden_size, MAX_LENGTH, embedding, p_dropout):
        super(BinaryTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=p_dropout)
        self.embedding = embedding
        self.max_length = MAX_LENGTH
        self.W_iock = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size).to(device)
        self.U_iock = torch.nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=False).to(device)
        self.W_f_l = torch.nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.U_f_l = torch.nn.Linear(
            self.hidden_size, self.hidden_size, bias=False).to(device)
        self.W_f_r = torch.nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.U_f_r = torch.nn.Linear(
            self.hidden_size, self.hidden_size, bias=False).to(device)
    '''
    input: root node of the tree
      return:
        output at shape (T,B,H*num_direction) ( in this case num_direction is 1 so shape is T*B*H)
        Tuble (h,c) of shape (num_layer*numdirection,B,H) in this case (1,B,H)
  '''
    def forward(self, input_forest):
        forest_output = None
        forest_h = None
        forest_c = None
        for tree in input_forest:
            tree_output, (tree_h, tree_c) = self.tree_traversal(tree)
            if tree_output is None:
                tree_output = torch.zeros(self.max_length, self.hidden_size)
                tree_h = torch.zeros(1, self.hidden_size)
                tree_c = torch.zeros(1, self.hidden_size)
            else:
                tree_output = self.widen_output(tree_output)
            if forest_output is None:
                forest_output = tree_output.unsqueeze(0)
                forest_h = tree_h.unsqueeze(0)
                forest_c = tree_c.unsqueeze(0)
            else:
                forest_output = torch.cat(
                    (forest_output, tree_output.unsqueeze(0)), dim=0)
                forest_c = torch.cat((forest_c, tree_c.unsqueeze(0)), dim=0)
                forest_h = torch.cat((forest_h, tree_h.unsqueeze(0)), dim=0)

        return forest_output, (forest_h.transpose(0, 1), forest_c.transpose(0, 1))
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

    def calculate(self, input_left, input_right, c_k_left, c_k_right):
        iock = self.W_iock(input_left) + self.U_iock(input_right)
        fl = self.W_f_l(input_left) + self.U_f_l(input_right)
        fr = self.W_f_r(input_left) + self.U_f_r(input_right)
        # print(iock.shape)
        i, o, ck = torch.split(iock, iock.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        ck = torch.tanh(ck)
        fl = torch.sigmoid(fl)
        fr = torch.sigmoid(fr)
        c_k_phr = i*ck + fl * c_k_left + fr * c_k_right
        h_k_phr = o*torch.tanh(c_k_phr)
        return h_k_phr, c_k_phr

    def widen_output(self, output):
        while output.shape[0] < self.max_length:
            output = torch.cat((output, torch.zeros(
                1, self.hidden_size).to(device)), dim=0)
        return output

    def get_inithidden(self):
        return torch.ones((self.hidden_size, 1)).to(device)

    def tree_traversal(self,adj_list):
        '''
			input is a list as : [(text,[id_left,id_right])] 
		'''
        
        if len(adj_list) == 0:
            return torch.zeros(self.hidden_size,1),(torch.zeros(self.hidden_size,1),torch.zeros(self.hidden_size,1))
       
        numNode = len(adj_list)
        for i in range(numNode-1, 0, -1):
            if adj_list[i][0] != "":
                
                try:
                    adj_list[i].append(self.embedding(torch.Tensor([adj_list[i][0]]).to(torch.int64).to(device)))
                    adj_list[i].append(torch.zeros(self.hidden_size,1).to(device))
                except:
                    print("err at ",adj_list[i][0])
            else:
                left_id = adj_list[i][1][0]
                right_id = adj_list[i][1][1]
                h_left = adj_list[left_id][2]
                c_left = adj_list[left_id][3]
                h_right = adj_list[right_id][2]
                c_right = adj_list[right_id][3]
                try:
                    h,c = self.calculate(h_left,h_right,c_left,c_right)
                except:
                    print(i)
                adj_list[i].append(h)
                adj_list[i].append(c)
        outputs = None
        for i in range(numNode-1, 0, -1):
            if adj_list[i][0] == "":
                if outputs is None:
                    outputs = adj_list[i][2]
                else:
                    outputs = torch.cat((outputs,adj_list[i][2].transpose(0,1)),dim=0)
        print(adj_list[0])
        h = adj_list[0][2]
        c = adj_list[0][3]
        return outputs,(h,c)