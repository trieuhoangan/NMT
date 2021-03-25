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
import encoder
import decoder
import time
import gensim
path_to_file_vi = '../vi_model.bin'
path_to_file_en = '../en_model.bin'
en_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_en,binary=True)
vi_model = gensim.models.KeyedVectors.load_word2vec_format(path_to_file_vi,binary=True)

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
  for i in range(batch):
    sum += int(lst[i])
  if sum == batch:
    return True
  else:
    return False
'''
  the input tensor and target tensor should be a batch of sentences
    shape of target and input are (B,T)

'''
def train(input_tensor, target_tensor ,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length,batch_size):
    # encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = 700
    loss = 0
    
    encoder_seq_output,encoder_seq_hc = encoder(input_tensor)

    
    word_input = []
    for i in range(batch_size):
      word_input.append(en_model.vocab['<start>'].index)
    decoder_input = torch.tensor(word_input, device=device)
    decoder_hidden = encoder_seq_hc[0].unsqueeze(0)

    use_teacher_forcing = True
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # for bi in range(batch_size)
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_seq_output.transpose(0,1))
          
            loss += criterion(decoder_output, target_tensor[:,di])
      
            if check_end(decoder_input,batch_size):
              # print(decoder_input)
              break
            decoder_input = target_tensor[:,di]  # Teacher forcing
            
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_seq_output.transpose(0,1))
            '''
              decoder is in shape (B,H)
              mean first word of each sentence.
            '''
            loss += criterion(decoder_output, target_tensor[:,di])
            decoder_input = torch.Tensor( target_tensor[:,di])
            # topv, topi = decoder_output.topk(1)

            # decoder_input = topi.squeeze().detach()  # detach from history as input
            # if decoder_input.item() == EOS_token:
            #     break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


'''
  split dataset into batch for training
  input_sentence is all sentence in the dataset
  input_forest is all trees that is parsed from sentences inside of the dataset
  target_sentence is all sentence that is translated from the other side
'''
def trainIters(encoder, decoder, input_sentence,target_sentence,batch_size,input_model,target_model, MAX_LENGTH,save_path,epoch,last_iter,encoder_optimizer,decoder_optimizer,print_every=400, plot_every=400):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    total_exp = len(input_sentence)
    n_iters = int(total_exp/batch_size)
    checkpoint = last_iter * batch_size

    for iter in range(last_iter+1, n_iters + 1):
        # print(iter)
        input_batch = get_k_elements(source_list=input_sentence,batch_size=batch_size,start_point=checkpoint)
        target_batch = get_k_elements(source_list=target_sentence,batch_size=batch_size,start_point=checkpoint)
        checkpoint += batch_size
        input_tensor,in_lengths = tensorFromSentence(model=input_model,sentences=input_batch,MAX_SEQUENCE_LENGTH=MAX_LENGTH)
        target_tensor,tar_lengths = tensorFromSentence(model=target_model,sentences=target_batch,MAX_SEQUENCE_LENGTH=MAX_LENGTH)
        loss = train(input_tensor, target_tensor ,encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,MAX_LENGTH,batch_size)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        if iter>0 and iter % 5000 == 0:
          enc_path = '{}/checkpoint.pt'.format(save_path)
          torch.save({
            'epoch':epoch,
            'iter': iter,
            'enc_state_dict': enc.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'loss': loss,
            }, enc_path)
    # showPlot(plot_losses)
    return print_loss_total


def trainEpoch(enc,dec,input_data_path,target_data_path,num_epoch,last_epoch,last_iter,save_path,learning_rate=0.01):
  import os
  encoder_optimizer = optim.SGD(enc.parameters(), lr=learning_rate)
  decoder_optimizer = optim.SGD(dec.parameters(), lr=learning_rate)
  for epoch in range(last_epoch,num_epoch):
    '''
      load data from file each time begin a new epoch
    '''
    epoch_path = '{}/checkpoint.pt'.format(save_path)
    # if os.path.exists(epoch_dir)== False:
    #   os.mkdir(epoch_dir)
    raw_input_text = open(input_data_path, 'rb').read().decode(encoding='utf-8')
    raw_target_text = open(target_data_path, 'rb').read().decode(encoding='utf-8')

    input_sentences = []
    target_sentences = []

    input_sentences.extend(raw_input_text.split('\n'))
    target_sentences.extend(raw_target_text.split('\n'))

    input_sent = preprocess_batch(input_sentences)
    target_sent = preprocessing_without_start(target_sentences)

    batch_size = 1
    max_length = 870
    loss = trainIters(enc, dec,input_sent,target_sent, batch_size,vi_model,en_model,max_length,save_path,epoch,last_iter,encoder_optimizer,decoder_optimizer)
    print('finish epoch {} - loss {}'.format(epoch+1,loss))
    # spath = '{}/epoch_{}.pt'.format(epoch_dir,epoch)
    torch.save({
            'epoch': epoch,
            'iter':0,
            'enc_state_dict': enc.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'loss': loss,
            }, epoch_path)
    last_iter = 0