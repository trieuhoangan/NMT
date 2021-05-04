# preprocessing data
import re
import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PhoNode import Tree_,PhoNode
def preprocessing(sentence):
  sentence = re.sub(r"&apos",r" ",sentence)
  sentence = re.sub(r"&quot;",r" ",sentence)
  sentence = re.sub(r";s",r"s",sentence)
  sentence = re.sub(r"([?.!,¿])", r" ", sentence)
  sentence = re.sub(r'[" "]', " ", sentence)
  sentence = re.sub(r'[","]', "", sentence)
  sentence = re.sub(r'&#93', "", sentence)
  sentence = re.sub(r'&#91', "", sentence)
  sentence = re.sub(r'"', "", sentence)
  sentence = re.sub(r"'", "", sentence)
  sentence.strip()
  # sentence = '<start> '+sentence+' <end>'
  return sentence
def preprocess_batch(sentences):
  real_sent = []
  for sentence in sentences:
    sentence = preprocessing(sentence)
    sentence = '<start> '+sentence+' <end>'
    # real_sent.append(sentence.split(' '))
    real_sent.append(sentence)
  return real_sent
def preprocessing_without_start(sentences):
  real_sent = []
  for sentence in sentences:
    sentence = preprocessing(sentence)
    sentence = sentence+' <end>'
    # real_sent.append(sentence.split(' '))
    real_sent.append(sentence)
  return real_sent
def split_and_preprocessing_fortesting(sentences):
  real_sent = []
  for sentence in sentences:
    sentence = preprocessing(sentence)
    sentence = sentence+' <end>'
    # real_sent.append(sentence.split(' '))
    real_sent.append(sentence)
  return real_sent

def indexesFromSentence(model, sentence,MAX_SEQUENCE_LENGTH):
    # actuall_indices = [model.vocab[word].index for word in sentence if word!='']
    indices = np.zeros(MAX_SEQUENCE_LENGTH)
    indices = indices+1
    pos = 0
    # length = 0
    for word in sentence.split(' '):
      if word!='':
        indices[pos] = model.vocab[word].index
        pos=pos+1
    return indices,pos

def tensorFromSentence(model, sentences,MAX_SEQUENCE_LENGTH):
  # print(len(sentences))
  indexesList = []
  lengths = []
  for sentence in sentences:
    indexes,length = indexesFromSentence(model, sentence,MAX_SEQUENCE_LENGTH)
    indexesList.append(indexes)
    lengths.append(length)
    # indexes.append(EOS_token)
  return torch.tensor(indexesList, dtype=torch.long, device=device),lengths

#prepare forest
def save_forest_to_file(sentence_batch,parsing_model,file_path,error_path):
  error_parsing = []
  add_error = open(error_path,'a',encoding='utf-8')
  with open(file_path,'a',encoding='utf-8') as f:
    for sentence in sentence_batch:
      try:
        token_list =  parsing_model.annotate(text=sentence)
      except:
        # add_error = open(error_path,'a',encoding='utf-8')
        add_error.write(sentence+'\n')
        # add_error.close()
        # error_parsing.append(sentence)
      text_form = ''
      for word in token_list[0][0]:
        text_form=text_form+word+','
      text_form= text_form+'|'
      for pos in token_list[1][0]:
        text_form = text_form + pos[0] + ','
      text_form= text_form+'|'
      for ner in token_list[2][0]:
        text_form=text_form+ner+','
      text_form= text_form+'|'
      for head_index in token_list[3][0]:
        text_form = text_form + head_index[0]+':'+head_index[1]+','
      f.write(text_form+'\n')
    return error_parsing
def load_token_list_from_file(file_path):
  lines =  open(file_path,'r',encoding='utf-8').read().split('\n')
  token_list = []
  for line in lines:
    if line =='':
      continue
    if line =='None':
      token_list.append([])
      continue
    words = []
    poss = []
    ners = []
    head_indexes = []
    feature = line.split('|')
    for word in feature[0].split(','):
      if word !='':
        words.append(word)
    for pos in feature[1].split(','):
      if pos !='':
        poss.append([pos])
    for ner in feature[2].split(','):
      if ner!='':
        ners.append(ner)
    for index in feature[3].split(','):
      if index!='':
        # part = index.split(':')
        head_indexes.append(index.split(':'))
    listone = []
    listone.append([words])
    listone.append([poss])
    listone.append([ners])
    listone.append([head_indexes])
    token_list.append(listone)
  return token_list
def create_forest(list_token_list,embedding_model):
  forest = []
  for token_list in list_token_list:
    if len(token_list) == 0:
      forest.append(None)
      continue
    tree = Tree_(token_list)
    if len(tree.nodeList) == 0:
      forest.append(None)
      continue
    bin_tree = tree.make_binary_tree(tree.nodeList)
    bin_tree.clear_bin_tree()
    bin_tree.convert_bin_tree_to_word_index(embedding_model)
    # bin_tree.print_word_indices()
    forest.append(bin_tree)
  return forest

def get_k_elements(source_list,batch_size,start_point):
  result = []
  for i in range(0,batch_size):
    result.append(source_list[start_point+i])
  return result
