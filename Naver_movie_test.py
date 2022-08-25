import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
import time
import random
from gensim.models import Word2Vec
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
from sklearn.metrics import roc_curve
import csv
from konlpy.tag import Okt
okt = Okt()
import statistics

def savefiles(save_result_path, Result):
    #header_write = False
    #file_exist = os.path.exists(os.path.join(save_result_path, name + '_' + 'gpu_number_' + str(gpu_number) + '.csv'))## check the same name // write name one time)
    filename = os.path.join(save_result_path,'result_' + '.csv')
    file = open(filename, 'a', encoding='euc_kr', newline='') ### 형식 w, a, r
    wr = csv.writer(file)
    wr.writerow(Result)
    file.close()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def read_data(filename):
    with open(filename, 'r', encoding='utf8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

print('train_data len', len(train_data))
print('test_data len', len(test_data))
#print((train_data[0]))

def tokenize(doc, num):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    if num == 1:
        return [''.join(t) for t in okt.morphs(doc, norm=True, stem=True)]
    elif num == 2:
        return [''.join(t) for t in okt.nouns(doc)]
    else:
        return [''.join(t) for t in okt.pos(doc, norm=True, stem=True)]

##############################################################################################################################################################################
gpu_number = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
###########################################################################


file_name1 = '_docs_morphs.json'
file_name2 = '_docs_nouns.json'
file_name3 = '_docs_pos.json'

PATH_json = 'C:/PycharmProjects/cnn-text-classification'
json_names = file_name1 ##### 1: morph // 2: nouns // 3: pos
if json_names==file_name1:
    token_num = 1 ##### 1: morph
elif json_names==file_name2:
    token_num = 2  ##### 2: nouns
else:
    token_num = 3  ##### // 3: pos
print(token_num)

if os.path.isfile(PATH_json + '/train' + str(json_names)):
    print('Json is exited')
    with open(PATH_json + '/train'+ json_names, encoding='utf8') as f:
        train_docs = json.load(f)
    with open(PATH_json + '/test'+ json_names, encoding='utf8') as f:
        test_docs = json.load(f)
else:
    print('Json is not existed')
    train_docs = [(tokenize(row[1], token_num), row[2]) for row in train_data] ### row[1] 감상평 내용 row[2] GT
    test_docs = [(tokenize(row[1], token_num), row[2]) for row in test_data] ### row[1] 감상평 내용 row[2] GT

    # JSON 파일로 저장
    with open(PATH_json + '/train' + json_names, 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open(PATH_json  + '/test' +json_names, 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")

print('train_docs[0]', train_docs[0])
print('train_docs[0][0]', train_docs[0][0])
print('train_docs[0][1]', train_docs[0][1])
##########################################################################################
##########################################################################################
#train_docs = train_docs[ : 100]
#test_docs = test_docs[ : 100]
#########################################################################################################

train_docs_sen = []
test_docs_sen = []
train_docs_sen_lable = []
test_docs_sen_lable = []
for i in range(len(train_docs)):
    train_docs_sen.append(train_docs[i][0])
    train_docs_sen_lable.append(train_docs[i][1]) ### train label
for i in range(len(test_docs)):
    test_docs_sen.append(test_docs[i][0])
    test_docs_sen_lable.append(test_docs[i][1])  #### test label
print('train_docs_sen[0]', train_docs_sen[0])
print('test_docs_sen[0]', test_docs_sen[0])
print('train_docs_sen_lable', train_docs_sen_lable[0])

########## word2vec train ####################
if not os.path.isfile(PATH_json + '/Word2_train_model'):
    model = Word2Vec(size = 300, window=3, min_count=1, workers=1, iter=100) ### train model
    model.build_vocab(train_docs_sen)
    model.train(train_docs_sen, total_examples=model.corpus_count, epochs=100)
    model.save(PATH_json + '/Word2_train_model')
    print('---word embedding finished---')
######### Load trained model ########
else:
    word2_train_model = Word2Vec.load(PATH_json + '/Word2_train_model')
    print('---word2vec_train_model load ---')

### Max sentence
max_length = max([len(x) for x in train_docs_sen]) # max
median_value=[]
median_value.extend([len(x)for x in train_docs_sen]) #median
AVG_length = int(sum(median_value) / len(train_docs_sen)) # Average
median = int(statistics.median(median_value))
print('Max_length :',max_length)
print('AVG_length :',AVG_length)
print('Median_length', median)

max_document_length = median
print('max_document_length', median)

def zero_padding_check(sentence_temp, max_document_length):
    z_padding = np.zeros(300) ### word embedding dimension
    sentence_temp_list = []
    sentence_temp_list.extend(sentence_temp)
    if len(sentence_temp) < max_document_length: ### fit max length
        for i in range(max_document_length - len(sentence_temp)):
            sentence_temp_list.append(z_padding)
    elif len(sentence_temp) >= max_document_length:
        sentence_temp_list= sentence_temp[0 : max_document_length]
    

    return sentence_temp_list


################ 단어 임베딩 리스트
sentence_vector_train_list = []
oov_word__train_num = []
for i in range(len(train_docs_sen)): ####  word embedding each word
    sentence_temp = []
    for j in range(len(train_docs_sen[i])):
        if train_docs_sen[i][j] in word2_train_model.wv.vocab: ############ OOV check
            sentence_temp.append(word2_train_model[train_docs_sen[i][j]])
        else:
            oov_word__train_num.append([train_docs_sen[i][j]])
    #if sentence_temp: ### 빈 리스트 저장 X
    zero_padding_temp = zero_padding_check(sentence_temp, max_document_length)
    sentence_vector_train_list.append(zero_padding_temp)
print('sentence_vector_train_list', len(sentence_vector_train_list))
print('sentence_vector', sentence_vector_train_list[0])
print('sentence_vector_train_list', len(sentence_vector_train_list[99]))
print('train_embedding finished')

################## test
sentence_vector_list = []
oov_word_num = []
for i in range(len(test_docs_sen)): ####  word embedding each word
    sentence_temp = []
    for j in range(len(test_docs_sen[i])):
        if test_docs_sen[i][j] in word2_train_model.wv.vocab: ############ OOV check
            sentence_temp.append(word2_train_model[test_docs_sen[i][j]])
        else:
            oov_word_num.append([test_docs_sen[i][j]])
    #if sentence_temp: ### 빈 리스트 저장 X
    zero_padding_temp = zero_padding_check(sentence_temp, max_document_length)
    sentence_vector_list.append(zero_padding_temp)
print('sentence_vector_list', len(sentence_vector_list))
print('sentence_vector_list1', len(sentence_vector_list[1]))
print('sentence_vector_list1', (sentence_vector_list[1]))
print('sentence_vector_list2', len(sentence_vector_list[0][0]))
print('oov_word_num', len(oov_word_num))



x_train_input= sentence_vector_train_list ## embedding
x_test_input = sentence_vector_list ## embedding
# y_train_label = train_docs_sen_lable ## integer
# y_test_label =test_docs_sen_lable ## integer

#x_train_input  = np.squeeze(sentence_vector_train_list)
#x_test_input  = np.squeeze(sentence_vector_list)
y_train_label = np.squeeze(train_docs_sen_lable).astype(int)
y_test_label = np.squeeze(test_docs_sen_lable).astype(int)

y_train_onehot = np.zeros((len(train_docs), 2))
for i in range(len(y_train_label)):
    if y_train_label[i] == 0:
        y_train_onehot[i] = [1, 0]
    elif y_train_label[i] == 1:
        y_train_onehot[i] = [0, 1]
y_train_onehot = y_train_onehot.astype(int)

y_test_onehot = np.zeros((len(test_docs), 2))
for i in range(len(y_test_label)):
    if y_test_label[i] == 0:
        y_test_onehot[i] = [1, 0]
    elif y_test_label[i] == 1:
        y_test_onehot[i] = [0, 1]
y_test_onehot = y_test_onehot.astype(int)

x_train_input= np.array(x_train_input).astype(float)
x_train_input = x_train_input.reshape(len(train_docs), -1)
print('x_train_input.shape', x_train_input.shape)

x_test_input = np.array(x_test_input).astype(float)
x_test_input = x_test_input.reshape(len(test_docs), -1)
print('x_test_input.shape', x_test_input.shape)



# sequence_length: 최대 문장 길이
# num_classes: 클래스 개수
# vocab_size: 등장 단어 수
# embedding_size: 각 단어에 해당되는 임베디드 벡터의 차원
# filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
# num_filters: 각 filter size 별 filter 수
# l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("model_type", "multichannel", "Model type (default: static), 'rand, static, non-static, multichannel'")
tf.flags.DEFINE_string("word2vec", "./data/word2vec/GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings (default: None)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


ops.reset_default_graph() ############# graph reset ----
num_classes = 2
embedding_size = 300
num_filters = 128
filter_sizes = [3, 4, 5]


input_x = tf.placeholder(tf.float32, [None, max_document_length*embedding_size], name="input_x")
input_x_reshape= tf.reshape(input_x, [-1, max_document_length, embedding_size, 1])
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

dropout_rate_input = 0.5
#filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters] # 3, 300, 1, 128
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    print('w', W)
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

    conv = tf.nn.conv2d(input_x_reshape, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
    conv_r = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    pooled = tf.nn.max_pool(conv_r,  ksize=[1, max_document_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
    print('conv_r', conv)


    # filter_shape = [filter_size, 1, num_filters, num_filters]  # 3, 1, 128, 128
    # W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, name="W2"))
    # conv2 = tf.nn.conv2d(conv_r, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
    # pooled2 = tf.nn.max_pool(conv2, ksize=[1, max_document_length - filter_size*2 + 2, 1, 1], strides=[1, 1, 1, 1],  padding='VALID', name="pool2") ## 두번 필터링 했으므로 filter_size*2 + 2
    # pooled_outputs.append(pooled2)

    pooled_outputs.append(pooled)
    print('pooled_outputs', pooled_outputs)

num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
print('h_pool', h_pool.shape)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

l2_loss = tf.constant(0.0)
#num_filters_total = num_filters *3*4*5
#W = tf.compat.v1.get_variable ("W",  shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
W = tf.get_variable ("W",  shape=[num_filters_total, num_classes],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
l2_loss += tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)

hypothesis = tf.layers.dense(h_drop, num_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())

#hypothesis = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
#predictions = tf.argmax(hypothesis, 1, name="predictions")

l2_reg_lambda=0.1
#losses = tf.nn.softmax_cross_entropy_with_logits_v2(hypothesis, input_y)
#loss = tf.reduce_mean(losses) #+ l2_reg_lambda * l2_loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=input_y))

correct_predictions = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(input_y, 1))
#correct_predictions = tf.equal(tf.argmax(predictions, 1) tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

########################################################################
#cv = StratifiedKFold(n_splits=Fold, shuffle=True, random_state=permutation_num)
########################################################################
learning_rate = 0.001
global_step = tf.Variable(0, name="global_step", trainable=False)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
# grads_and_vars = optimizer.compute_gradients(loss)
# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)


#sess = tf.compat.v1.Session()
#sess.run(tf.compat.v1.global_variables_initializer())

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=10)
sess.run(tf.global_variables_initializer())

train_epoch = 10
batch_size = 64

train_size = len(train_docs)
print('trainsize', train_size)
fpr1_list=[]
for epoch in range(train_epoch):
    start_time = time.time()
    cm_list = []
    avg_loss = 0
    total_batch = int(train_size / batch_size)
    randidx = random.sample(range(0, train_size), train_size)  ### 중복없는 랜덤 선택
    for i in range(total_batch):
        randidx2 = randidx[i * batch_size: (i + 1) * batch_size]
        x_batch = x_train_input[randidx2]
        y_batch = y_train_onehot[randidx2]

        feed_dict = {input_x: x_batch, input_y: y_batch, dropout_keep_prob: FLAGS.dropout_keep_prob}
        _, step, loss_val, accuracy_val = sess.run([train_op, global_step, loss, accuracy], feed_dict)
        avg_loss += loss_val / total_batch
        #print('h_pool', sess.run(h_pool_flat, feed_dict=feed_dict))
    print('train---', "epoch {}, avg_loss {:g}".format(epoch, avg_loss))

    feed_dict = {input_x: x_test_input, input_y: y_test_onehot, dropout_keep_prob: 1.0 }
    loss_test, accuracy_test = sess.run([loss, accuracy],  feed_dict)
    print("epoch {}, acc {:g}".format(epoch, accuracy_test))

    correct_temp = sess.run(tf.cast(correct_predictions, "float"), feed_dict=feed_dict)
    savefiles(PATH_json, correct_temp)

    saver.save(sess, 'C:/PycharmProjects/cnn-text-classification/train_model.ckpt')

################### Check CM
    y_pred = sess.run(hypothesis, feed_dict={input_x: x_test_input, input_y: y_test_onehot, dropout_keep_prob: 1})
    y_pred_softmax = sess.run(tf.nn.softmax(y_pred))
    y_pred_cat = sess.run(tf.argmax(y_pred, 1))
    y_true_cat = sess.run(tf.argmax(y_test_onehot, 1))

    fpr1, tpr1, threshold1 = roc_curve(y_test_onehot[:, 1], y_pred_softmax[:, 1])
    fpr1_list.append(fpr1)

    cm = confusion_matrix(y_true_cat, y_pred_cat)
    cm_list.append(cm)
    #print('cm', cm)
    print("cm {}" .format(cm))
    PPV = cm[0][0] / (cm[0][0] + cm[1][0])  # PPV
    NPV = cm[1][1] / (cm[0][1] + cm[1][1])  # NPV
    SEN = cm[0][0] / (cm[0][1] + cm[0][0])  # SEN
    SPE = cm[1][1] / (cm[1][1] + cm[1][0])  # SPE

    #print('PPV', PPV)
    #print('SEN', SEN)
    #print('SPE', SPE)
    print("PPV {}, SEN {}, SPE {}" .format(PPV, SEN, SPE))

    accuracy_test = (cm[0][0] + cm[1][1]) / sum(sum(cm))
    #print('accuracy_test', accuracy_test)
    print("accuracy_test {}".format(accuracy_test))












#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#vocab_size=len(vocab_processor.vocabulary_)
# x = np.array(list(vocab_processor.fit_transform(test_docs_sen)))
#print('vocab_size', vocab_size)





###### 테스트 시 OOV 검사
# word = 'lotte'
# for word in test_docs:
#     if word in word2_train_model.wv.vocab:
#         print('ex')
#     else:
#         sentence_vector_result.append(word2_train_model[train_docs[i]])


### 품사별 저장 --- 1
# print(train_docs[0][0])
# for i in train_docs[0][0]:
#     temp = [i.split('/')]
#     for j in temp:
#         if j[1] in ['Noun', 'Adjective', 'Exclamation']:
#             print('tag', j)

### 품사별 저장 --- 2
# noun_adj_list = []
# for sentence1 in sentences_tag:
#     for word, tag in sentence1:
#         if tag in ['Noun','Adjective']:
#             noun_adj_list.append(word)





