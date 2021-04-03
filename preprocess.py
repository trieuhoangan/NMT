# preprocessing data
import re
import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def preprocessing(sentence):
  sentence = re.sub(r"&apos",r" ",sentence)
  sentence = re.sub(r"&quot;",r" ",sentence)
  sentence = re.sub(r";s",r"s",sentence)
  sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]', " ", sentence)
  sentence = re.sub(r'[","]', "", sentence)
  sentence = re.sub(r'&#93', "", sentence)
  sentence = re.sub(r'&#91', "", sentence)
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


def get_k_elements(source_list,batch_size,start_point):
  result = []
  for i in range(0,batch_size):
    result.append(source_list[start_point+i])
  return result
