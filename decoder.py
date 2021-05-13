import torch
import torch.nn as nn
import torch.nn.functional as F
from PhoNode import Tree_,PhoNode
import numpy as np
from torch import optim
import math
import PhoNode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from treeLSTM import BinaryTreeLSTMCell
class NewAttn(nn.Module):
  def __init__(self,method,hidden_size):
    self.method = method
    self.hidden_size = hidden_size

  def forward(self,tree_output, seq_ouputs, cur_state):
    '''
      tree_outputs has size: (B,N-1,H)
      seq_outputs has size: (B,N,H)
      cur_state has size: (B,H)
      output has size: (B,H)
    '''
    return None
  def calculate_attn_weigh(self):
    '''
      output is a float number
    '''
    return None

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.ones(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

import gensim
class Decoder(nn.Module):
  def __init__(self,input_size,hidden_size,max_length,embedding_path,embed_size,output_size,n_layers=1,dropout_p=0.1):
    super(Decoder,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=True)
    weights = torch.FloatTensor(model.vectors).to(device)
    self.embedding = nn.Embedding.from_pretrained(weights).to(device)
    # Define parameters
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    # Define layers
    self.dropout = nn.Dropout(dropout_p)
    self.attn = Attn('concat', hidden_size)
    gru_input_size = hidden_size + embed_size
    self.gru = nn.GRU(gru_input_size, hidden_size, dropout=dropout_p)
    #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)
    self.treeLSTM = BinaryTreeLSTMCell(input_size,hidden_size,max_length,self.embedding,dropout_p)
  def forward(self, word_input, last_hidden, encoder_outputs, tree_encoder_outputs):
    '''
    :param word_input:
        word input for current time step, in shape (B)
    :param last_hidden:
        last hidden stat of the decoder, in shape (layers*direction*B*H)
    :param encoder_outputs:
        encoder outputs in shape (T*B*H)
    :return
        decoder output
    Note: we run this one step at a time i.e. you should use a outer loop 
        to process the whole sequence
    Tip(update):
    EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
    different from that of DecoderRNN
    You may have to manually guarantee that they have the same dimension outside this function,
    e.g, select the encoder hidden state of the foward/backward pass.
    '''
    # Get the embedding of the current input word (last output word)
    
    # word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
    try:
      word_embedded = self.embedding(word_input)
    except:
      catch_error(word_input)
    word_embedded = word_embedded.unsqueeze(0)
    word_embedded = self.dropout(word_embedded)
    # Calculate attention weights and apply to encoder outputs
    lhidden = last_hidden[-1]
    # print(lhidden.shape)
    # if index == 0:
      # lhidden,lc = self.treeLSTM()
    attn_weights = self.attn(lhidden, encoder_outputs)
    tree_attn_weights = self.attn(lhidden, tree_encoder_outputs)

    # the attention is not very correct, need to concat hidden state of tree and sequence before these. 
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
    tree_context = tree_attn_weights.bmm(encoder_outputs.transpose(0, 1))
    context = context.transpose(0, 1)  # (1,B,V)
    tree_context = tree_context.transpose(0,1)
    context = tree_context + context


    # Combine embedded input word and attended context, run through RNN
    rnn_input = torch.cat((word_embedded, context), 2)
    # print('rrn input',rnn_input.shape)
    #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
    output, hidden = self.gru(rnn_input, lhidden)
    output = output.squeeze(0)  # (1,B,V)->(B,V)
    # context = context.squeeze(0)
    # update: "context" input before final layer can be problematic.
    # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
    output = F.log_softmax(self.out(output))
    # Return final output, hidden state
    return output, hidden.unsqueeze(0),attn_weights,tree_attn_weights
  '''
    tree_ouput has shape (1,B,H)
    seq_output has shape (1,B,H)
    c_tree has shape (num)layer*direction,B,H)
    c_seq has shape (num)layer*direction,B,H)
    return first_hidden at shape (layers,direction,B,H)
  '''
  def get_first_hidden(self,tree_last_hidden,seq_last_hidden,c_tree,c_seq):
    first_hiddens = []
    print(tree_last_hidden[0][i].unsqueeze(1).shape)
    print(seq_last_hidden[0][i].unsqueeze(1).shape)
    print(c_tree[0][i].unsqueeze(1).shape)
    print(c_seq[0][i].unsqueeze(1).shape)
    for i in range(int(tree_last_hidden.shape[1])):
      hidden_one,c_one = self.treeLSTM.calculate(tree_last_hidden[0][i].unsqueeze(1),
                                           seq_last_hidden[0][i].unsqueeze(1),
                                           c_tree[0][i].unsqueeze(1),
                                           c_seq[0][i].unsqueeze(1)
                                           )
      if i ==0:
        first_hiddens = hidden_one
      else:
        first_hiddens = torch.cat((first_hiddens,hidden_one),dim=1)
    first_hiddens = first_hiddens.transpose(0,1)
    first_hiddens = first_hiddens.unsqueeze(0)
    first_hiddens = first_hiddens.unsqueeze(0)
    return first_hiddens
