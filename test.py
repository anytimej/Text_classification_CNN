import tensorflow as tf
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import time
import datetime
import json

# train = pd.read_csv("./review_list_lotte_super_sample.csv", sep='\t')
# print(train.shape)
# print('train.head(10)', train.head(10))

# import csv
#
# f = open('review_list_lotte_super_sample.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
# content_list =[]
# for line in rdr:
#     content_list.append(line[4])
# f.close()
# content_list.pop(0)
# print('content', content_list)


char = [0, 0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3, 4,4, 5,5,5,5,5,]
doc = ['한편', '1890', '어느날까', '모름', '그', '사랑한다']

answer_offset = 0
answer_lenth = 2
start_position = char[answer_offset]
end_position = char[answer_offset+answer_lenth-1]
print('aa', start_position)
print('bb', end_position)

#print('chr', char[start_position + end_position - 1])

print('chr', char[start_position + end_position - 1])
print('doc', doc[start_position: (end_position + 1)])


def aaa(values):
    feature = tf.train.Feature(int64_list = tf.train.Int64List(value = list(values)))
    return feature

import collections
features = collections.OrderedDict
features["start_position"] = aaa([feature.unique_id])
print('features', features)

