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
    # self.attn = Attn('concat', hidden_size)
    self.attn = NewAttn(hidden_size)
    gru_input_size = hidden_size + embed_size
    # self.gru = nn.GRU(gru_input_size, hidden_size, dropout=dropout_p)
    self.LSTM = nn.LSTMCell(gru_input_size, hidden_size, dropout=dropout_p,batch_first=True)
    self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
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
    # attn_weights = self.attn(lhidden, encoder_outputs)
    context = self.attn(tree_encoder_outputs,encoder_outputs,lhidden)
    # the attention is not very correct, need to concat hidden state of tree and sequence before these. 
    # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
    # tree_context = tree_attn_weights.bmm(encoder_outputs.transpose(0, 1))
    # context = context.transpose(0, 1)  # (1,B,V)
    # tree_context = tree_context.transpose(0,1)
    # context = tree_context + context


    # Combine embedded input word and attended context, run through RNN
    rnn_input = torch.cat((word_embedded, context), 2)
    # print('rrn input',rnn_input.shape)
    #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
    # output, hidden = self.LSTM(rnn_input, lhidden)
    hidden = self.attn_combine(rnn_input)
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
    for i in range(int(tree_last_hidden.shape[1])):
      hidden_one,c_one = self.treeLSTM.calculate(tree_last_hidden[0][i].unsqueeze(0),
                                           seq_last_hidden[0][i].unsqueeze(0),
                                           c_tree[0][i].unsqueeze(0),
                                           c_seq[0][i].unsqueeze(0)
                                           )
      if i ==0:
        first_hiddens = hidden_one
      else:
        first_hiddens = torch.cat((first_hiddens,hidden_one),dim=1)
    first_hiddens = first_hiddens.transpose(0,1)
    first_hiddens = first_hiddens.unsqueeze(0)
    first_hiddens = first_hiddens.unsqueeze(0)
    return first_hiddens
class NewAttn(nn.Module):
  def __init__(self,hidden_size):
    super(NewAttn,self).__init__()
    self.hidden_size = hidden_size

  def forward(self,tree_output, seq_ouput, cur_state,numNode):
    '''
      tree_outputs has size: (B,N-1,H)
      seq_outputs has size: (B,MAX_LENGTH,H)
      cur_state has size: (B,H)
      output has size: (B,H)
      numNode have shape (B)
    '''
    ds = None
    batch_size = tree_output.shape[0]
    for i in range(batch_size):
      numLeaf = numNode[i] +1
      '''
        at this time, tree_i has size (N-1,H)
        seq_i has size (N,H)
        state_i has size (H)
      '''
      tree_i = tree_output[i][:numNode[i]]
      seq_i = seq_ouput[i][:numLeaf]
      state_i = cur_state[i]
      enc_out = torch.cat((seq_i,tree_i),dim=0)
      # print("attn enc_out",enc_out.shape)
      # print("attn state_i",state_i.shape)
      attn_weights = self.calculate_attn_weigh(state_i,enc_out)
      '''
        at this time attn_weights has size (2N-1)
        size of enc_out (2N-1,H)
      '''
      d= torch.zeros(self.hidden_size).to(device)
      for j in range(numLeaf+numNode[i]):
        d = d + attn_weights[j]*enc_out[j]
      d = d.unsqueeze(0)
      if ds is None:
        ds = d
      else:
        ds = torch.cat((ds,d),dim=0)
    return ds  
  def calculate_attn_weigh(self,cur_state,enc_output ):
    '''
    size of enc_output: (2N-1,H)
    size of cur_state: (1,H)
      output is a float number
    '''
    weight_matrix = []
    for i in range(enc_output.shape[0]):
      i_enc_state = enc_output[i]
      # print("calculating attn weigh, cur_state ",cur_state.shape)
      # print("calculating attn weigh, i_enc_state ",i_enc_state.shape)
      attn_weight = torch.dot(cur_state,i_enc_state)
      weight_matrix.append(attn_weight)
    weight_matrix = torch.Tensor(weight_matrix).to(torch.float32).to(device)
    weight_matrix = torch.softmax(weight_matrix,dim=0)
    return weight_matrix

class NewDecoder(nn.Module):
  def __init__(self,input_size,hidden_size,max_length,embedding_path,embed_size,output_size,n_layers=1,dropout_p=0.1):
    super(NewDecoder,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=True)
    weights = torch.FloatTensor(model.vectors).to(device)
    self.embedding = nn.Embedding.from_pretrained(weights).to(device)
    self.attn = NewAttn(hidden_size)
    self.treeLSTM = BinaryTreeLSTMCell(input_size,hidden_size,max_length,self.embedding,dropout_p)
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout_p = dropout_p
    self.out = nn.Linear(hidden_size, output_size)
    self.LSTM = nn.LSTMCell(hidden_size,hidden_size)
    
    self.combine_context = nn.Linear(hidden_size*2,hidden_size,bias=True)
  def forward(self, word_indices,last_hidden,tanh_hidden,tree_output, seq_output,numNode):
    '''
    :param word_input:
        word input for current time step, in shape (B)
    :param last_hidden:
        last hidden stat of the decoder, in shape (layers*direction*B*H) -> (1,B,H)
        tanh_hidden has shape (1,B,H)
    :param encoder_outputs:
        encoder outputs in shape (T,B,H)
        tree encoder in shape ()
    :return
        decoder output has the shape (B*O) ( O : vocab size )
    Note: we run this one step at a time i.e. you should use a outer loop 
        to process the whole sequence
    Tip(update):
    EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
    different from that of DecoderRNN
    You may have to manually guarantee that they have the same dimension outside this function,
    e.g, select the encoder hidden state of the foward/backward pass.
    '''
    tree_output = tree_output.transpose(0,1)
    seq_output = seq_output.transpose(0,1)
    print("tree ",tree_output.shape)
    print("seq ",seq_output.shape)
    print("word ",word_indices.shape)
    print("hidden ",last_hidden.shape)
    lhidden = last_hidden[0]
    batch = word_indices.shape[0]
    current_ht = torch.zeros(batch,self.hidden_size)
    if self.is_begin_token(word_indices):
      current_ht = lhidden
    else:
      # try:
      word_embedded = self.embedding(word_indices)
    # except:
      #   catch_error(word_input)
      
      current_ht = lhidden + tanh_hidden
      print("cur ht",current_ht.shape)
      print("word_embedded",word_embedded.shape)
      c = torch.zeros(batch,self.hidden_size).to(device)
      current_ht = self.LSTM(word_embedded,(current_ht,c))
    print("cur ht",current_ht.shape)
    context = self.attn(tree_output,seq_output,current_ht,numNode)
    context_vector = torch.cat((current_ht,context),dim=1)
    current_tanh_hidden = torch.tanh(self.combine_context(context_vector))
    out_vec = self.out(current_tanh_hidden)
    prob = F.softmax(out_vec,dim=0)
    return prob,current_ht,current_tanh_hidden,context

  def is_begin_token(self,word_indices):
    sum = 0
    for index in word_indices:
      sum+= index
    if sum == 0:
      return True
    else:
      return False


  def get_first_hidden(self,tree_last_hidden,seq_last_hidden,c_tree,c_seq):
    '''
      all of input have size(1,H) -> (B,H)
      output have size (B,H)
    '''
    first_hiddens = []
    for i in range(int(tree_last_hidden.shape[1])):
      hidden_one,c_one = self.treeLSTM.calculate(tree_last_hidden[0][i].unsqueeze(0),
                                           seq_last_hidden[0][i].unsqueeze(0),
                                           c_tree[0][i].unsqueeze(0),
                                           c_seq[0][i].unsqueeze(0)
                                           )
      if i ==0:
        first_hiddens = hidden_one
      else:
        first_hiddens = torch.cat((first_hiddens,hidden_one),dim=0)
    first_hiddens = first_hiddens.unsqueeze(0)
    return first_hiddens
  def init_new_hidden(self):
    return torch.zeros((1,self.hidden_size)).to(device)

class customLSTM(nn.Module):
  def __init__(self,input_size: int, hidden_size: int, bias: bool,num_chunks: int):
    super(customLSTM,self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias
    self.weight_inp = Parameter(torch.Tensor(num_chunks*input_size,4*hidden_size))
    self.weight_hid = Parameter(torch.Tensor(num_chunks*2*hidden_size,4*hidden_size))
    self.bias_inp = Parameter(torch.Tensor(num_chunks,hidden_size))
    self.bias_hid = Parameter(torch.Tensor(num_chunks,hidden_size))
  def forward(self, input, hx):
    if hx is None:
      zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
      hx = (zeros, zeros)
    ifgo = F.linear(input, self.weight_inp, self.bias) + F.linear(hx[0],self.weight_hid,self.bias_hid)
    i,f,g,o = torch.split(ifgo, ifgo.size(1) // 4, dim=1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    g = torch.tanh(g)
    o = torch.sigmoid(o)
    c = f*hx[1] + i*g
    h = o*torch.tanh(c)
    return h,c