import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import torch.cuda as tc
import numpy as np
from torch import optim
import math
import PhoNode
from preprocess import preprocess_batch,preprocessing,preprocessing_without_start,load_token_list_from_file,get_k_elements,create_forest,indexesFromSentence,tensorFromSentence
from node import get_indices_list,load_simple_token_list_from_file,create_node_list,make_forest_from_token_list
import encoder
import decoder
import time
import gensim
import random
path_to_file_vi = 'models/language_models/vi_model.bin'
path_to_file_en = 'models/language_models/en_model.bin'
en_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_en,binary=True)
vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)
MAX_LENGTH = 870
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


teacher_forcing_ratio = 0.5
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
def check_end(lst,batch):
  sum = 0 
  for num in lst:
    sum += int(num)
  if sum == batch:
    return True
  else:
    return False
'''
  the input tensor and target tensor should be a batch of sentences
    shape of target and input are (B,T)

'''
def train(input_tensor, target_tensor, input_forest ,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,batch_size,isTrain):
    # encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = 700
    loss = 0
    
    numNode,encoder_tree_output,encoder_seq_output,encoder_tree_hc,encoder_seq_hc = encoder(input_tensor,input_forest)
    maxNode = 0
    for num in numNode:
      if num > maxNode:
        maxNode = num
    maxNode += 1
    tanh_hidden = decoder.init_new_hidden()
    word_input = []
    for i in range(batch_size):
      word_input.append(en_model.vocab['<start>'].index)
    decoder_input = torch.tensor(word_input, device=device)
    
    last_seq_hidden = encoder_seq_output[:,maxNode].unsqueeze(0)
    decoder_hidden = decoder.get_first_hidden(encoder_tree_hc[0],last_seq_hidden,encoder_tree_hc[1],encoder_seq_hc[1])
    use_teacher_forcing = True
    if isTrain == False:
      use_teacher_forcing = False
    else:
      # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
      use_teacher_forcing = True
    c = torch.zeros(batch_size,decoder.hidden_size).to(device)

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # for bi in range(batch_size)
        for di in range(target_length):
            decoder_output, decoder_hidden,decoder_tanh_hidden, decoder_attention,c = decoder(
                decoder_input, decoder_hidden,tanh_hidden ,encoder_seq_output.transpose(0,1),encoder_tree_output.transpose(0,1),numNode,c)
          
            loss += criterion(decoder_output, target_tensor[:,di])
            tanh_hidden = decoder_tanh_hidden
            if check_end(decoder_input,batch_size):
              target_length = di
              break
            decoder_input = target_tensor[:,di]  # Teacher forcing
            # print("decoder_input ",decoder_input)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden,decoder_tanh_hidden, decoder_attention,c = decoder(
                decoder_input, decoder_hidden,tanh_hidden ,encoder_seq_output.transpose(0,1),encoder_tree_output.transpose(0,1),numNode,c)
            '''
              decoder is in shape (B,H)
              mean first word of each sentence.
            '''
            tanh_hidden = decoder_tanh_hidden
            
            loss += criterion(decoder_output, target_tensor[:,di])
            # print("no force teaching used ",target_tensor[:,di])
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # print("decoder_input ",decoder_input)
            if check_end(decoder_input,batch_size):
              target_length = di
              break
            # if decoder_input.item() == EOS_token:
            #     break
    if isTrain:
      loss.backward()
      encoder_optimizer.step()
      decoder_optimizer.step()
    if torch.isnan(loss):
      print("input of nan ",input_tensor)
      print("target of nan ",target_tensor)
      print("tree of nan ",input_forest)
    return loss.item()


'''
  split dataset into batch for training
  input_sentence is all sentence in the dataset
  input_forest is all trees that is parsed from sentences inside of the dataset
  target_sentence is all sentence that is translated from the other side
'''
def trainIters(encoder, decoder, input_sentence,input_tokenlist,target_sentence,batch_size,input_model,target_model, MAX_LENGTH,save_path,epoch,last_iter,encoder_optimizer,decoder_optimizer,print_every=400, plot_every=400,isTrain = True):
    print_every = (print_every * 16) / batch_size
    plot_every = (plot_every * 16) / batch_size
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()

    total_exp = len(input_sentence)
    n_iters = int(total_exp/batch_size)
    checkpoint = last_iter * batch_size
    totalLoss = 0
    for iter in range(last_iter+1, n_iters + 1):
        # print(iter)
        
        input_batch = get_k_elements(source_list=input_sentence,batch_size=batch_size,start_point=checkpoint)
        forest_batch = get_k_elements(source_list=input_tokenlist,batch_size=batch_size,start_point=checkpoint)
        target_batch = get_k_elements(source_list=target_sentence,batch_size=batch_size,start_point=checkpoint)
        checkpoint += batch_size
        input_tensor,in_lengths = tensorFromSentence(model=input_model,sentences=input_batch,MAX_SEQUENCE_LENGTH=MAX_LENGTH)
        target_tensor,tar_lengths = tensorFromSentence(model=target_model,sentences=target_batch,MAX_SEQUENCE_LENGTH=MAX_LENGTH)
        
        input_forest = make_forest_from_token_list(forest_batch,input_model)
        loss = train(input_tensor, target_tensor,input_forest ,encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,MAX_LENGTH,batch_size,isTrain)
        print_loss_total += loss
        plot_loss_total += loss
        totalLoss += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            # print("encoder",encoder.parameters())
            # print("decoder",decoder.parameters())

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / (plot_every*batch_size)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        if iter>0 and iter % 800 == 0:
          enc_path = '{}/checkpoint.pt'.format(save_path)
          torch.save({
            'epoch':epoch,
            'iter': iter,
            'enc_state_dict': encoder.state_dict(),
            'dec_state_dict': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'loss': loss,
            }, enc_path)
    # showPlot(plot_losses)

    return totalLoss/n_iters


def trainEpoch(encoder,decoder,args,last_epoch,last_iter,save_path,learning_rate=0.015):
  import os
  encoder_optimizer = optim.Adamax(encoder.parameters(), lr=learning_rate)
  decoder_optimizer = optim.Adamax(decoder.parameters(), lr=learning_rate)
  input_data_path = args['input_data_path']
  target_data_path = args['target_data_path']
  input_forest_path = args['input_forest_path']
  num_epoch = args['epoch']
  for epoch in range(last_epoch,num_epoch):
    '''
      load data from file each time begin a new epoch
    '''
    encoder.train()
    decoder.train()
    epoch_path = '{}/checkpoint.pt'.format(save_path)
    # if os.path.exists(epoch_dir)== False:
    #   os.mkdir(epoch_dir)
    
    raw_input_text = open(input_data_path, 'rb').read().decode(encoding='utf-8')
    raw_target_text = open(target_data_path, 'rb').read().decode(encoding='utf-8')

    input_sentences = []
    target_sentences = []

    input_sentences.extend(raw_input_text.split('\n'))
    target_sentences.extend(raw_target_text.split('\n'))

    input_sent = preprocess_batch(input_sentences[:130000])
    target_sent = preprocessing_without_start(target_sentences[:130000])
    lst = load_simple_token_list_from_file(input_forest_path)
    
    batch_size = 8
    max_length = 870
    loss = trainIters(encoder, decoder,input_sent,lst[:130000],target_sent, batch_size,vi_model,en_model,max_length,save_path,epoch,last_iter,encoder_optimizer,decoder_optimizer)
    eval_input,eval_target,eval_lst = get_eval_data(args)
    encoder.eval()
    decoder.eval()
    eval_loss = trainIters(encoder,decoder,eval_input,eval_lst,eval_target,batch_size,vi_model,en_model,max_length,save_path,0,0,encoder_optimizer,decoder_optimizer,isTrain=False)
    print('finish epoch {} - loss {}'.format(epoch+1,eval_loss))
    # spath = '{}/epoch_{}.pt'.format(epoch_dir,epoch)
    torch.save({
            'epoch': epoch,
            'iter':0,
            'enc_state_dict': encoder.state_dict(),
            'dec_state_dict': decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'loss': loss,
            }, epoch_path)
    last_iter = 0
    # evaluate(encoder,decoder,args,vi_model,en_model)
def get_eval_data(args):
  input_valid_path = args['input_valid_path']
  target_valid_path = args['target_valid_path']
  valid_forest_path = args['valid_forest_path']
  raw_input_text = open(input_valid_path, 'rb').read().decode(encoding='utf-8')
  raw_target_text = open(target_valid_path, 'rb').read().decode(encoding='utf-8')
  input_sentences = []
  target_sentences = []
  input_sentences.extend(raw_input_text.split('\n'))
  target_sentences.extend(raw_target_text.split('\n'))
  input_sent = preprocess_batch(input_sentences)
  target_sent = preprocessing_without_start(target_sentences)
  lst = load_simple_token_list_from_file(valid_forest_path)
  return input_sent,target_sent,lst

def evaluate(encoder,decoder,args,input_model,target_model):
  
  numExample = len(lst)
  totalLoss = 0
  criterion = nn.NLLLoss()
  
  for i in range(0,limit):
    input_batch = get_k_elements(input_sent,batch_size,i*batch_size)
    target_batch = get_k_elements(target_sent,batch_size,i*batch_size)
    forest_batch = get_k_elements(lst,batch_size,i*batch_size)
    
    input_tensor,in_lengths = tensorFromSentence(model=input_model,sentences=input_batch,MAX_SEQUENCE_LENGTH=MAX_LENGTH)
    target_tensor,tar_lengths = tensorFromSentence(model=target_model,sentences=target_batch,MAX_SEQUENCE_LENGTH=MAX_LENGTH)
    input_forest = make_forest_from_token_list(forest_batch,input_model)
    
    target_length = 700
    loss = 0
    numNode,encoder_tree_output,encoder_seq_output,encoder_tree_hc,encoder_seq_hc = encoder(input_tensor,input_forest)
    
    maxNode = 0
    for num in numNode:
      if num > maxNode:
        maxNode = num
    maxNode += 1
    tanh_hidden = decoder.init_new_hidden()
    word_input = []
    for j in range(batch_size):
      word_input.append(target_model.vocab['<start>'].index)
    decoder_input = torch.tensor(word_input, device=device)
    last_seq_hidden = encoder_seq_output[:,maxNode].unsqueeze(0)
    decoder_hidden = decoder.get_first_hidden(encoder_tree_hc[0],last_seq_hidden,encoder_tree_hc[1],encoder_seq_hc[1])
    use_teacher_forcing = True
    c = torch.zeros(batch_size,decoder.hidden_size).to(device)
    for di in range(target_length):
      decoder_output, decoder_hidden,decoder_tanh_hidden, decoder_attention,c = decoder(
          decoder_input, decoder_hidden,tanh_hidden ,encoder_seq_output.transpose(0,1),encoder_tree_output.transpose(0,1),numNode,c)
      '''
        decoder is in shape (B,H)
        mean first word of each sentence.
      '''
      tanh_hidden = decoder_tanh_hidden
      loss += criterion(decoder_output, target_tensor[:,di])
      topv, topi = decoder_output.topk(1)
      decoder_input = topi.squeeze().detach()  # detach from history as input
      if check_end(decoder_input,batch_size):
        target_length = di
        break
    loss = loss/target_length
    totalLoss += loss
    torch.cuda.empty_cache()
  print("validation step: Loss : %.4f"%(totalLoss))
  torch.cuda.empty_cache()
    