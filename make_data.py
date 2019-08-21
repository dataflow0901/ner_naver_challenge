import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import doc2vec
from sklearn.model_selection import train_test_split

data_path = './data/'
#data_save = './data_out/'

tf.reset_default_graph()

dataset = pd.read_excel(data_path + 'gs.xlsx', header=0)

dic = {}
ejeol = []
ner_tag = []
idx = []

for c, n in zip(dataset.chat, dataset.NER):
    chat_s = c.split(" ")
    ner_s = str(n).split(" ")
    # if len(chat_s) != len(ner_s) and len(ner_s)>1:
    #     print(chat_s)
    #     print(ner_s)

    for i in range(len(chat_s)):
        ejeol.append(chat_s[i])
        idx.append(i)

        if i >= len(ner_s):
            ner_tag.append("-")
        else:
            ner_tag.append(ner_s[i])

    ejeol.append("\n")
    idx.append("\n")
    ner_tag.append("\n")

file = open('./data/test.txt', 'w', encoding = 'utf8')
for i,j,k in zip(idx, ejeol, ner_tag):
  if i =='\n':
    line = (str(t+1)+"\t"+"."+"\t"+"-"+"\n"+"\n")
  else:
    line = (str(i+1)+"\t"+j+"\t"+k+"\n")
    t = i+1
  file.write(line)
file.close()

''' # shell에서  test_data만들 때
for i,j,k in zip(idx, ejeol, ner_tag):
  if i =='\n':
    print(str(t+1)+"\t"+"."+"\t"+"-"+"\n")
  else:
    print(str(i+1)+"\t"+j+"\t"+k)
    t = i+1
'''