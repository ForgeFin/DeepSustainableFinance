import os
import pandas as pd
import numpy as np
import re
import time
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import tensorflow as tf
# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Sequential, Model

import tensorflow.keras.backend as K #
from tensorflow.keras.layers import Layer, InputSpec #
from tensorflow.keras import initializers

from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy

from tensorflow.keras import regularizers


#################################################################################
''' 
Define paths containing the label data and paths for storage.
Set params for the neural network
'''
#################################################################################

## define path containing files (see folder "Files" for files "validation_data_rs0.txt","train_data_rs0.txt","test_data_rs0.txt")
path = 'C:/Users/.../Train_test_split/'    

## define path with pre-trained glove embeddings
glove_path = 'C:/Users/Armbrust/Desktop/glove_embeddings/'

## define path containing the documents / files
bertdocs = 'C:/Users/.../files/'

## define folder for saving glove models
savemodel = 'C:/Users/.../saved_models/glove/'

## define file containing nltk stopwords
stopwords = 'C:/Users/.../nltk_stopwords.txt'

## define according environmental label
label = 'EP_label'    ## << HERE 'ep90', 'EP_label'

''' Set params for neural networks'''
epoch = 50
batch = 32
denser = 100
regu = 0.1 
drop = 0.5

###########################
''' Define Metrics ''' 
###########################

def calc_metrics(yprediction, ytrue):
    ''' calculating precision '''
    #print('using K.epsilon since otherwise devision through zero!')
    TruePositive = 0
    for i in range(len(yprediction)):
        if yprediction[i] == ytrue[i] and yprediction[i] == 1:
            TruePositive += 1
    
    FalsePositive = 0 # outcome where the model incorrectly predicts the positive class
    for i in range(len(yprediction)):
        if yprediction[i] != ytrue[i] and yprediction[i] == 1:
            FalsePositive += 1
    
    FalseNegative = 0 # outcome where the model incorrectly predicts the negative class
    for i in range(len(yprediction)):
        if ytrue[i] != yprediction[i] and yprediction[i] == 0:
            FalseNegative += 1
            
    precision = TruePositive / (TruePositive + FalsePositive + K.epsilon())
    
    recall = TruePositive / (FalseNegative + TruePositive + K.epsilon())
    
    Fscore = 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    return Fscore, precision , recall, TruePositive, FalsePositive, FalseNegative#,


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
        
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # clip (x, min_value, max_value)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def TP_FP_FN_forBootstrap(yprediction, ytrue):
    ''' calculating precision '''
    TruePositive_list = []
    for i in range(len(yprediction)):
        if yprediction[i] == ytrue[i] and yprediction[i] == 1:
            TruePositive = 1
        else:
            TruePositive =0
        TruePositive_list.append(TruePositive)
    
    FalsePositive_list = []
    for i in range(len(yprediction)):
        if yprediction[i] != ytrue[i] and yprediction[i] == 1:
            FalsePositive = 1
        else:
            FalsePositive = 0
        FalsePositive_list.append(FalsePositive)
    
    FalseNegative_list = []
    for i in range(len(yprediction)):
        if ytrue[i] != yprediction[i] and yprediction[i] == 0:
            FalseNegative = 1
        else:
            FalseNegative = 0
        FalseNegative_list.append(FalseNegative)
            
    diction = {'TP':TruePositive_list,'FP':FalsePositive_list,'FN':FalseNegative_list}
    df = pd.DataFrame(diction, columns=['TP','FP','FN'])
    
    return df

#from sklearn.metrics import classification_report
from sklearn.metrics import f1_score # (y_true, y_pred)


class F1History(tf.keras.callbacks.Callback):
    ''' See Marco Cerliani https://stackoverflow.com/questions/61683829/calculating-fscore-for-each-epoch-using-keras-not-batch-wise '''
    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):

        logs['F1_score_train'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        y_pred = (self.model.predict(X_train).ravel()>0.5)+0
        score = f1_score(y_train, y_pred)       

        if (self.validation):
            logs['F1_score_val'] = float('-inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = (self.model.predict(X_valid).ravel()>0.5)+0
            val_score = f1_score(y_valid, y_val_pred)
            logs['F1_score_train'] = np.round(score, 5)
            logs['F1_score_val'] = np.round(val_score, 5)
        else:
            logs['F1_score_train'] = np.round(score, 5)
            


###########################
## only 10-K and 10_Q
validation_all = pd.read_csv(path+'validation_data_rs0.txt', sep=',', index_col=0, encoding='utf-8')
train_all = pd.read_csv(path+'train_data_rs0.txt', sep=',', index_col=0, encoding='utf-8')
test_all = pd.read_csv(path+'test_data_rs0.txt', sep=',', index_col=0, encoding='utf-8')
###########################

print("Note: to_categorical starts with 0, so need to substract 1 from industry codification")
industry_train = tf.keras.utils.to_categorical(np.array(train_all["Industry_numeric"]-1))
print("See shape industry_train",industry_train.shape)
industry_val = tf.keras.utils.to_categorical(np.array(validation_all["Industry_numeric"]-1))
industry_test = tf.keras.utils.to_categorical(np.array(test_all["Industry_numeric"]-1))

mktcap_test = np.array(test_all["Log_MarketCap"])
mktcap_train = np.array(train_all["Log_MarketCap"])
mktcap_val = np.array(validation_all["Log_MarketCap"])

print("Test size %s, Train size %s, Val size %s" % (len(test_all),len(train_all),len(validation_all)))

validation_all['path'] = bertdocs
train_all['path'] = bertdocs
test_all['path'] = bertdocs

validation_all['fileload'] = validation_all[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)
train_all['fileload'] = train_all[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)
test_all['fileload'] = test_all[['path', 'fname']].apply(lambda x: ''.join(x), axis=1)

''' save these to list and check unique counts'''
x_test, y_test_EP, y_test_bhar, y_test_ep90, y_test_eps, y_test_bhar_short = test_all['fileload'].to_list(), test_all['EP_label'].to_list(), test_all['BHAR_label'].to_list(), test_all['label_env'].to_list(), test_all['label_eps'].to_list(), test_all['BHAR_Short_label'].to_list()
x_train, y_train_EP, y_train_bhar, y_train_ep90, y_train_eps, y_train_bhar_short = train_all['fileload'].to_list(), train_all['EP_label'].to_list(), train_all['BHAR_label'].to_list(), train_all['label_env'].to_list(), train_all['label_eps'].to_list(), train_all['BHAR_Short_label'].to_list()
x_val, y_val_EP, y_val_bhar, y_val_ep90, y_val_eps, y_val_bhar_short = validation_all['fileload'].to_list(), validation_all['EP_label'].to_list(), validation_all['BHAR_label'].to_list(), validation_all['label_env'].to_list(), validation_all['label_eps'].to_list(), validation_all['BHAR_Short_label'].to_list()


############
############
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

print("Loading text to memory")
with open(stopwords, encoding='utf-8') as document:
    stopwords = document.read().split(",")
stopwords = set(stopwords)

def cleanstr(file_list):
    text = []
    # translator = str.maketrans(punctuation, ' '*len(punctuation)) ## splits thi-s as "thi" "s"
    for f in file_list:
        with open(f, encoding='utf-8') as document:
            doc = document.read().lower()
            # tok = doc.translate(translator)
            tokens = doc.split()
            # remove punctuation from each token
#            table = str.maketrans('', '', punctuation)
#            tokens = [w.translate(table) for w in doc]
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            tokens = [w for w in tokens if not w in stopwords]
            docs = " ".join(tokens)
            text.append(docs)
    return text

## documents as str in list
doc_training = cleanstr(x_train)
doc_test = cleanstr(x_test)
doc_val = cleanstr(x_val)


'''
### KERAS tokenizer
'''
## if num_words, less words are considered. Only most commmon words considered
t = Tokenizer(num_words=None, filters='', split=' ')
t.fit_on_texts(doc_training)
# summarize what was learned
t_wordcount = t.word_counts # ordered dict of word counts
t_doccount = t.document_count # document count
t_wordidx = t.word_index # index by frequency (most frequent = lowest no.)
t_worddocs = t.word_docs # in how many docs a word occured

# ‘binary‘: Whether or not each word is present in the document.
#           This is the default.
# ‘count‘: The count of each word in the document.
# ‘tfidf‘: The Text Frequency-Inverse DocumentFrequency (TF-IDF) 
#          scoring for each word in the document.
# ‘freq‘: The frequency of each word as a ratio of words
#          within each document.

''' For tfidf and counts'''
### freq and tfidf make most sense for shap
## integer encode documents

# maxlen: Int, maximum length of all sequences.
# ONLY small fraction of words is longer than 20000 words
MAX_SEQUENCE_LENGTH = 20000 ## Note documents with more words than max_seq are cuttet!


# num_words: maximum number of words to keep, based on word frequency.
#            Only the most common num_words-1 words will be kept.

sparse_x = t.texts_to_sequences(doc_training)
sparse_x = pad_sequences(sparse_x, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
sparse_x_val = t.texts_to_sequences(doc_val)
sparse_x_val = pad_sequences(sparse_x_val, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
sparse_x_test = t.texts_to_sequences(doc_test)
sparse_x_test = pad_sequences(sparse_x_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

sparse_x = sparse_x.astype('float32')
sparse_x_val = sparse_x_val.astype('float32')
sparse_x_test = sparse_x_test.astype('float32')


# Load word vectors from pre-trained dataset
embeddings_index = {}
f = open(glove_path+'glove.6B.100d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# Embedding
EMBED_SIZE = 100
min_wordCount = 2
absent_words = 0
small_words = 0

word_index = t.word_index 
print('Found %s unique tokens.' % len(word_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBED_SIZE))
word_counts = t.word_counts
for word, i in word_index.items():
    if word_counts[word] > min_wordCount:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            absent_words += 1
    else:
        small_words += 1
        
print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)),
      '% of total words') ## absent words = words not in GloVe embedding matrix
print('Words with '+str(min_wordCount)+' or less mentions', small_words, 'which is', "%0.2f" % (small_words * 100 / len(word_index)),
      '% of total words')
## only according number of words to proceed are accounted for; for others does not exist a word vector OR occurance is to small
print(str(len(word_index)-small_words-absent_words) + ' words to proceed.')


embedding_layer = Embedding(len(word_index) + 1,
                            EMBED_SIZE,
                            weights=[embedding_matrix], # by choosing embedding matrix only 27037 words are used
                            input_length=MAX_SEQUENCE_LENGTH, # input_length: Length of input sequences, when it is constant
                            trainable=False) # trainable is False, so is not updated while training

if label == 'ep90':
    y_train = y_train_ep90
    y_test = y_test_ep90
    y_val = y_val_ep90
    print(label)

else:
    y_train = y_train_EP
    y_test = y_test_EP
    y_val = y_val_EP
    print(label)

label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)


input_dim = sparse_x.shape[1] #1000
industry_dim = industry_train.shape[1]

y_train = np.array(y_train, dtype='float32')
y_test = np.array(y_test, dtype='float32')
y_val = np.array(y_val, dtype='float32')



def create_model(loadweight=None):
    ##### MODEL #####
    sequence_input = Input(shape=(input_dim,), dtype='float32')
    embedded_sequences = embedding_layer(sequence_input)
    market_cap = Input(shape=(1,), dtype='float32')
    industry = Input(shape=(industry_dim,), dtype='float32')
    
    x = Dense(denser,# kernel_regularizer=l1(0.0001),
              kernel_regularizer=regularizers.l1(regu),                   # kernel_regularizer = weight regularization
              bias_regularizer=regularizers.l1(regu),
              #activity_regularizer=regularizers.l2(0.1)
              )(embedded_sequences) # activity_regularizer = on outout of the layer 
    drp = Dropout(0.5)(x)
    flat = Flatten()(drp)
    combined = Concatenate()([flat, market_cap, industry])
    xx = Dropout(0.5)(combined)
    preds = Dense(1, activation='sigmoid',name='output',
                  #kernel_regularizer=regularizers.l2(1),
                  #bias_regularizer=regularizers.l2(1)
                  )(xx)
    model = Model([sequence_input,market_cap,industry], preds)
    
    if loadweight is not None:
        model.load_weights(loadweight)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', # opt
                  metrics=[TruePositives(name='true_positives'),
                          TrueNegatives(name='true_negatives'),
                          FalseNegatives(name='false_negatives'),
                          FalsePositives(name='false_positives'),
                          #Accuracy(name='accuracy')
                          BinaryAccuracy(name='binary_accuracy',
                                         dtype=None, threshold=0.5)
                          ])

    return model


model = create_model()

# checkpoint
filepath="weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(savemodelpath,filepath), verbose=1, save_best_only=False, save_weights_only=True, mode='auto')#, save_freq='epoch')
callbacks_list = [checkpoint]

history = model.fit([sparse_x,mktcap_train,industry_train], y_train,                ## << HERE
                    validation_data=([sparse_x_val,mktcap_val,industry_val], y_val), ## << HERE
                    epochs=epoch, batch_size=batch,
                    callbacks=[checkpoint]
                    )


### save accuracy and training tp, fn, fp
tp, fp, fn = [], [], []
tp_val, fp_val, fn_val = [], [], []
tp.append(history.history['true_positives'])
fp.append(history.history['false_positives'])
fn.append(history.history['false_negatives'])

## flatten list of lists without converting to numpy
tp = [item for sublist in tp for item in sublist]
fp = [item for sublist in fp for item in sublist]
fn = [item for sublist in fn for item in sublist]

tp_val.append(history.history['val_true_positives'])
fp_val.append(history.history['val_false_positives'])
fn_val.append(history.history['val_false_negatives'])

## flatten list of lists without converting to numpy
tp_val = [item for sublist in tp_val for item in sublist]
fp_val = [item for sublist in fp_val for item in sublist]
fn_val = [item for sublist in fn_val for item in sublist]

diction = {'tp':tp,'fp':fp,'fn':fn,'tp_val':tp_val,'fp_val':fp_val,'fn_val':fn_val}
df2 = pd.DataFrame(diction)
df2.to_csv(os.path.join(savemodelpath,'tp_fp_fn_from_training.txt'), sep=',', encoding='utf-8')


accuracy, val_accuracy = [], []
accuracy.append(history.history['binary_accuracy'])
val_accuracy.append(history.history['val_binary_accuracy'])
## flatten list of lists without converting to numpy
accuracy = [item for sublist in accuracy for item in sublist]
val_accuracy = [item for sublist in val_accuracy for item in sublist]

loss, val_loss = [], []
fscores, f_m, f_m_val = [], [], []
epochen = []
precision_sc, precision_sc_val = [], []
recall_sc, recall_sc_val = [], []
f1_sc, f1_sc_val = [], []
for i in range(0,epoch):
    epochen.append(i+1)
    pr = tp[i] / (tp[i] + fp[i] + K.epsilon())
    precision_sc.append(pr)
    pr_val = tp_val[i] / (tp_val[i] + fp_val[i] + K.epsilon())
    precision_sc_val.append(pr_val)
    rec = tp[i] / (tp[i] + fn[i] + K.epsilon())
    recall_sc.append(rec)
    rec_val = tp_val[i] / (tp_val[i] + fn_val[i] + K.epsilon())
    recall_sc_val.append(rec_val)


loss.append(history.history['loss'])
val_loss.append(history.history['val_loss'])
loss = np.array(loss).flatten()
val_loss = np.array(val_loss).flatten()
loss = list(loss)
val_loss = list(val_loss)

diction = {'epoch':epochen,
           'pre_train':precision_sc,'pre_val':precision_sc_val,
           'rec_train':recall_sc,'rec_val':recall_sc_val,
           'loss_train':loss,'loss_val':val_loss,'acc_train':accuracy,
           'acc_val':val_accuracy}
df3 = pd.DataFrame(diction)
df3.to_csv(os.path.join(savemodelpath,'f_p_r_loss_acc_from_training.txt'), sep=',', encoding='utf-8')


### baseline counts
unique, counts = np.unique(y_train,                 
                           return_counts=True)
baseline_trains = dict(zip(unique, counts))
total = baseline_trains[1] + baseline_trains[0]
total_precision_train = baseline_trains[1] / total
f_baseline_train = 2*(total_precision_train)/(1+total_precision_train)
print("Baseline_train:",baseline_trains)

unique_val, counts_val = np.unique(y_val,
                                   return_counts=True)
baseline_vals = dict(zip(unique_val, counts_val))
total_val = baseline_vals[1] + baseline_vals[0]
total_precision_val = baseline_vals[1] / total_val
f_baseline_val = 2*(total_precision_val)/(1+total_precision_val)
print("Baseline_val:",baseline_vals)

unique_test, counts_test = np.unique(y_test,
                                     return_counts=True)
baseline_test = dict(zip(unique_test, counts_test))
total_test = baseline_test[1] + baseline_test[0]
total_precision_test = baseline_test[1] / total_test
f_baseline_test = 2*(total_precision_test)/(1+total_precision_test)
print("Baseline_test:",baseline_test)

baseline = {'f_baseline_train':f_baseline_train,
            'f_baseline_val':f_baseline_val,
            'f_baseline_test':f_baseline_test,
            'p_baseline_train':total_precision_train,
            'p_baseline_val':total_precision_val,
            'p_baseline_test':total_precision_test,
            'r_baseline_train':1,
            'r_baseline_val':1,
            'r_baseline_test':1}
baseline = pd.DataFrame(baseline, index=[0])
baseline.to_csv(os.path.join(savemodelpath,'baseline.txt'), sep=',', encoding='utf-8')




''' Reloading the models '''

model_loads = []
for root, dirs, files in os.walk(savemodelpath):
    for file in files:
        if file.endswith(".hdf5"):  
            model_loads.append(os.path.join(root,file))

Fscore_val, Fscore_train = [], []

fscorepredict_val, fscorepredict_train = [], []
fscorepredict_val_custom, fscorepredict_val_sklearn = [], []
tp_train_total, fp_train_total, fn_train_total = [], [], []
tp_val_total, fp_val_total, fn_val_total = [], [], []
recall_train, recall_val = [], []
precision_train, precision_val = [], []
loss_tr, loss_ev = [], []
cnt = 0

for i in model_loads:
    model.load_weights(i)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', # opt
              metrics=[TruePositives(name='true_positives'),
                       TrueNegatives(name='true_negatives'),
                       FalseNegatives(name='false_negatives'),
                       FalsePositives(name='false_positives'),
              #         #Accuracy(name='accuracy')
                       BinaryAccuracy(name='binary_accuracy',
                                      dtype=None, threshold=0.5)
                      ])
    ## evaluate gives different results!
    # model.evaluate([sparse_x,mktcap_train,industry_train], y_train, batch_size=batch, verbose=0)

    # for training data
    y_pred_train =  model.predict([sparse_x,mktcap_train,industry_train], verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred_train]
    
    cm, p_tr, r_tr, tp_total, fp_total, fn_total = calc_metrics(train_preds, y_train)
    
    Fscore_train.append(cm)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    
    tp_train_total.append(tp_total)
    fp_train_total.append(fp_total)
    fn_train_total.append(fn_total)
    ff = f1_score(y_train, train_preds)
    fscorepredict_train.append(ff)
    
    ## for validation data
    y_pred_val = model.predict([sparse_x_val,mktcap_val,industry_val])
    val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
    
    cm, p, r, tp_total, fp_total, fn_total = calc_metrics(val_preds, y_val)
    Fscore_val.append(cm)
    precision_val.append(p)
    recall_val.append(r)    
    tp_val_total.append(tp_total)
    fp_val_total.append(fp_total)
    fn_val_total.append(fn_total)
    ff_val = f1_score(y_val, val_preds)
    fscorepredict_val_sklearn.append(ff_val)
    cnt += 1


fscorepredict_val_custom = np.array(Fscore_val)
best_validation_model = np.argmax(fscorepredict_val_custom)

print("")
print("fscore",fscorepredict_train)
print("loss",loss)
print("")
print("fscore val",fscorepredict_val_custom)
print("loss val",val_loss)
print("Best Val model",best_validation_model)

# using loss of model fit because model loss from evaluate
# is loss from forward pass without regularization and dropout
diction = {'epoch':epochen,'f_train':Fscore_train,'f_val':Fscore_val,
           'pre_train':precision_train,'pre_val':precision_val,
           'rec_train':recall_train,'rec_val':recall_val,
           'loss_train':loss,'loss_val':val_loss,
           'tp_train':tp_train_total,'fp_train':fp_train_total,'fn_train':fn_train_total,
           'tp_val':tp_val_total,'fp_val':fp_train_total,'fn_val':fn_val_total
           }
df4 = pd.DataFrame(diction)
df4.to_csv(os.path.join(savemodelpath,'f_p_r_loss_from_reloading.txt'), sep=',', encoding='utf-8')


fscorepredict_train = np.array(fscorepredict_train)


best_trainings_fscore = fscorepredict_train[np.argsort(fscorepredict_train)[-1:]]
best_validation_fscore = fscorepredict_val_custom[np.argsort(fscorepredict_val_custom)[-1:]]
best_val_index = np.argmax(fscorepredict_val_custom)
best_train_index = np.argmax(fscorepredict_train)


model.load_weights(model_loads[best_val_index])
model.compile(loss='binary_crossentropy',
              optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
          #         #Accuracy(name='accuracy')
                   BinaryAccuracy(name='binary_accuracy',
                                  dtype=None, threshold=0.5)
                  ])

# for training data
y_pred_val =  model.predict([sparse_x_val,mktcap_val,industry_val],
                              verbose=0)  # gives probabilities
saved_val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
    

## save bootstraps for best validation models
bootstrap_val = TP_FP_FN_forBootstrap(saved_val_preds,y_val)
bootstrap_val.to_csv(os.path.join(savemodelpath,'_bootstrap_val.txt'),sep=',',encoding='utf-8')

saved_val_preds = pd.DataFrame(saved_val_preds, columns=['y_val_preds'])
saved_val_preds.to_csv(os.path.join(savemodelpath,'_saved_val_preds.txt'),sep=',',encoding='utf-8')


y_pred_test =  model.predict([sparse_x_test,mktcap_test,industry_test],
                              verbose=0)  # gives probabilities
test_preds = [1 if x > 0.5 else 0 for x in y_pred_test]

f_test,pr_test, re_test, tp_test, fp_test, fn_test = calc_metrics(test_preds, y_test)


bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test)
bootstrap_test.to_csv(os.path.join(savemodelpath,'_bootstrap_test.txt'),sep=',',encoding='utf-8')

test_scores = {'f_test':f_test,'pr_test':pr_test,'re_test':re_test,
               'tp_test':tp_test, 'fp_test':fp_test, 'fn_test':fn_test, 'best_model':model_loads[best_val_index]}
df5 = pd.DataFrame(test_scores, index=[0])
df5.to_csv(os.path.join(savemodelpath,'best_val_model_on_test_set.txt'), sep=',', encoding='utf-8')



##################
''' Lime '''
##################

import seaborn as sns


model.load_weights(model_loads[best_val_index])
model.compile(loss='binary_crossentropy',
              optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
          #         #Accuracy(name='accuracy')
                   BinaryAccuracy(name='binary_accuracy',
                                  dtype=None, threshold=0.5)
                  ])

# for training data
y_pred_val =  model.predict([sparse_x_val,mktcap_val,industry_val],
                              verbose=0)  # gives probabilities




from collections import OrderedDict
from lime.lime_text import LimeTextExplainer

class_names = ['negative', 'positive']
''' BOW must be false if word order is taken into account'''
explainer = LimeTextExplainer(class_names=class_names,bow=False)



def weigth_mtx_test(data, sparse_data):
    ''' runs several lime explainers
    
    holds mktcap and industry constant!
    
    and return local lime values for each instance
    and predicted class probability of each instance '''    
    sample_indices = np.arange(len(data))
    # Generate Explanations
    explanations = []
    class_one_predictions = []
    
    for i in sample_indices:
        print(i)
        def predict_prob(string):
            ''' must take list of d strings and output (d,k) numpy array
                with prediction probabilities, where k is the number of classes
            
                mkt must be mktcap_train[instance_index]
                ind must be industry_train[instance_index]
            '''
            global model
            x_temp = t.texts_to_sequences(string)
            x_temp = pad_sequences(x_temp, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            x_temp = x_temp.astype('float32')
            
            shape_n = x_temp.shape[0]
            market = []
            indust = []
            for n in range(0,shape_n):
                ## for length of string (shape_n), create market
                ## filled with instance i
                market.append(mktcap_test[i])
                indust.append(list(industry_test[i]))
            market = np.array(market, dtype='float32')
            indust = np.array(indust, dtype='float32')
            
            prediction = model.predict([x_temp,market,indust])
            class_zero = 1-prediction
            probability= np.append(class_zero, prediction, axis=1)
        
            return probability ## array [1-p, p]
        
        explanations.append(explainer.explain_instance(data[i],
                                                       predict_prob, num_features=10))
        ## bring everything into shape for instance prediction        
        temp = sparse_data[i]
        temp = temp.reshape((1,len(temp)))
        temp = temp.astype('float32')
        
        mrkt = np.array(mktcap_test[i], dtype='float32')
        mrkt = mrkt.reshape((1,1))
        mrkt = mrkt.astype('float32')
        
        indu = industry_test[i]
        indu = indu.reshape((1,len(indu)))
        indu = indu.astype('float32')
        
        prediction = model.predict([temp,mrkt,indu])
        ## prediction[0][0] since array
        class_one_predictions.append(prediction[0][0])

    # Find all the explanation model features used. Defines the dimension d'
    features_dict = {}
    feature_iter = 0
    for exp in explanations:
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, _ in exp.as_list(label=label):
                if feature not in features_dict.keys():
                    features_dict[feature] = (feature_iter)
                    feature_iter += 1
    d_prime = len(features_dict.keys())
    
    def getList(dict):
        return list(dict.keys())
    
    feature_list = getList(features_dict)
    
    # Create the n x d' dimensional 'explanation matrix', W
    W = np.zeros((len(explanations), d_prime))
    for i, exp in enumerate(explanations):
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, value in exp.as_list(label):
                # print(i, feature, value)
                # order of W (nxd) with n being explainer and d being words
                # is d according to features dictionary (0 to end)
                W[i, features_dict[feature]] += value 
    
    ## include here prediction and actual y value for each instance into W
    
    df_w = pd.DataFrame(W, columns=feature_list)
    
    return df_w, class_one_predictions




''' For test data'''

percentage = round(len(doc_test) * 0.1)
print(percentage) ## 10 percent of data

y_test_list = list(y_test)
y_mkt_test = list(mktcap_test)
temp = {'doc':doc_test,'label':y_test_list,'mkt':y_mkt_test}
df_explainer = pd.DataFrame(temp)

df_zero_class = df_explainer[df_explainer['label']==0]
df_one_class = df_explainer[df_explainer['label']==1]

one = df_one_class['label'].iloc[:percentage] ### <<< HERE select 10% of zero
one = np.array(one, dtype='float32')
one_doc = df_one_class['doc'].iloc[:percentage] ### <<< HERE select 10% of zero
one_doc = list(one_doc)
one_doc_sparse = t.texts_to_sequences(one_doc)
one_doc_sparse = pad_sequences(one_doc_sparse, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
one_doc_sparse = one_doc_sparse.astype('float32')
mkt_one = df_one_class['mkt'].iloc[:percentage]
mkt_one = np.array(mkt_one, dtype='float32')
index_one = df_one_class.index.tolist() ## get indices from df
ind_one = industry_test[index_one,:] ## selects rows according to indices from df


zero = df_zero_class['label'].iloc[:percentage] ### <<< HERE select 10% of zero
zero = np.array(zero, dtype='float32')
zero_doc = df_zero_class['doc'].iloc[:percentage] ### <<< HERE select 10% of zero
zero_doc = list(zero_doc)
zero_doc_sparse = t.texts_to_sequences(zero_doc)
zero_doc_sparse = pad_sequences(zero_doc_sparse, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
zero_doc_sparse = zero_doc_sparse.astype('float32')
mkt_zero = df_zero_class['mkt'].iloc[:percentage]
mkt_zero = np.array(mkt_zero, dtype='float32')
index_zero = df_zero_class.index.tolist() ## get indices from df
ind_zero = industry_test[index_zero,:] ## selects rows according to indices from df


both = df_explainer['label'].iloc[:percentage] ### <<< HERE select 10% of zero
both = np.array(both, dtype='float32')
both_doc = df_explainer['doc'].iloc[:percentage] ### <<< HERE select 10% of zero
both_doc = list(both_doc)
both_doc_sparse = t.texts_to_sequences(both_doc)
both_doc_sparse = pad_sequences(both_doc_sparse, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
both_doc_sparse = both_doc_sparse.astype('float32')


''' both classes '''
df_w, prob_one = weigth_mtx_test(both_doc,both_doc_sparse)

df_abs_importance = df_w.abs().sum()
df_importance = df_w.sum()

df_w['label_prob'] = prob_one
df_w['label_true'] = both

df_w.to_csv(os.path.join(savemodelpath,'lime_weight_matrix_test_set_both_classes.txt'),
            sep=',', encoding='utf-8')
df_abs_importance.to_csv(os.path.join(savemodelpath,'lime_abs_feature_importance_test_set_both_classes.txt'),
            sep=',', encoding='utf-8')
df_importance.to_csv(os.path.join(savemodelpath,'lime_feature_importance_test_set_both_classes.txt'),
            sep=',', encoding='utf-8')




def weigth_mtx_neg_test(data, sparse_data):
    ''' runs several lime explainers
    
    holds mktcap and industry constant!
    
    and return local lime values for each instance
    and predicted class probability of each instance '''    
    sample_indices = np.arange(len(data))
    # Generate Explanations
    explanations = []
    class_one_predictions = []
    
    for i in sample_indices:
        print(i)
        def predict_prob(string):
            ''' must take list of d strings and output (d,k) numpy array
                with prediction probabilities, where k is the number of classes
            
                mkt must be mktcap_train[instance_index]
                ind must be industry_train[instance_index]
            '''
            global model
            x_temp = t.texts_to_sequences(string)
            x_temp = pad_sequences(x_temp, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            x_temp = x_temp.astype('float32')
            
            shape_n = x_temp.shape[0]
            market = []
            indust = []
            for n in range(0,shape_n):
                ## for length of string (shape_n), create market
                ## filled with instance i
                market.append(mkt_zero[i])
                indust.append(list(ind_zero[i]))
            market = np.array(market, dtype='float32')
            indust = np.array(indust, dtype='float32')
            
            prediction = model.predict([x_temp,market,indust])
            class_zero = 1-prediction
            probability= np.append(class_zero, prediction, axis=1)
        
            return probability ## array [1-p, p]
        
        explanations.append(explainer.explain_instance(data[i],
                                                       predict_prob, num_features=10))
        ## bring everything into shape for instance prediction        
        temp = sparse_data[i]
        temp = temp.reshape((1,len(temp)))
        temp = temp.astype('float32')
        
        mrkt = np.array(mkt_zero[i], dtype='float32')
        mrkt = mrkt.reshape((1,1))
        mrkt = mrkt.astype('float32')
        
        indu = ind_zero[i]
        indu = indu.reshape((1,len(indu)))
        indu = indu.astype('float32')
        
        prediction = model.predict([temp,mrkt,indu])
        ## prediction[0][0] since array
        class_one_predictions.append(prediction[0][0])

    # Find all the explanation model features used. Defines the dimension d'
    features_dict = {}
    feature_iter = 0
    for exp in explanations:
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, _ in exp.as_list(label=label):
                if feature not in features_dict.keys():
                    features_dict[feature] = (feature_iter)
                    feature_iter += 1
    d_prime = len(features_dict.keys())
    
    def getList(dict):
        return list(dict.keys())
    
    feature_list = getList(features_dict)
    
    # Create the n x d' dimensional 'explanation matrix', W
    W = np.zeros((len(explanations), d_prime))
    for i, exp in enumerate(explanations):
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, value in exp.as_list(label):
                # print(i, feature, value)
                # order of W (nxd) with n being explainer and d being words
                # is d according to features dictionary (0 to end)
                W[i, features_dict[feature]] += value 
    
    ## include here prediction and actual y value for each instance into W
    
    df_w = pd.DataFrame(W, columns=feature_list)
    
    return df_w, class_one_predictions



''' negative class '''
df_w, prob_one = weigth_mtx_neg_test(zero_doc,zero_doc_sparse)

df_abs_importance = df_w.abs().sum()
df_importance = df_w.sum()

df_w['label_prob'] = prob_one
df_w['label_true'] = zero

df_w.to_csv(os.path.join(savemodelpath,'lime_weight_matrix_test_set_negative_class.txt'),
            sep=',', encoding='utf-8')
df_abs_importance.to_csv(os.path.join(savemodelpath,'lime_abs_feature_importance_test_set_negative_class.txt'),
            sep=',', encoding='utf-8')
df_importance.to_csv(os.path.join(savemodelpath,'lime_feature_importance_test_set_negative_class.txt'),
            sep=',', encoding='utf-8')




def weigth_mtx_pos_test(data, sparse_data):
    ''' runs several lime explainers
    
    holds mktcap and industry constant!
    
    and return local lime values for each instance
    and predicted class probability of each instance '''    
    sample_indices = np.arange(len(data))
    # Generate Explanations
    explanations = []
    class_one_predictions = []
    
    for i in sample_indices:
        print(i)
        def predict_prob(string):
            ''' must take list of d strings and output (d,k) numpy array
                with prediction probabilities, where k is the number of classes
            
                mkt must be mktcap_train[instance_index]
                ind must be industry_train[instance_index]
            '''
            global model
            x_temp = t.texts_to_sequences(string)
            x_temp = pad_sequences(x_temp, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            x_temp = x_temp.astype('float32')
            
            shape_n = x_temp.shape[0]
            market = []
            indust = []
            for n in range(0,shape_n):
                ## for length of string (shape_n), create market
                ## filled with instance i
                market.append(mkt_one[i])
                indust.append(list(ind_one[i]))
            market = np.array(market, dtype='float32')
            indust = np.array(indust, dtype='float32')
            
            prediction = model.predict([x_temp,market,indust])
            class_zero = 1-prediction
            probability= np.append(class_zero, prediction, axis=1)
        
            return probability ## array [1-p, p]
        
        explanations.append(explainer.explain_instance(data[i],
                                                       predict_prob, num_features=10))
        ## bring everything into shape for instance prediction        
        temp = sparse_data[i]
        temp = temp.reshape((1,len(temp)))
        temp = temp.astype('float32')
        
        mrkt = np.array(mkt_one[i], dtype='float32')
        mrkt = mrkt.reshape((1,1))
        mrkt = mrkt.astype('float32')
        
        indu = ind_one[i]
        indu = indu.reshape((1,len(indu)))
        indu = indu.astype('float32')
        
        prediction = model.predict([temp,mrkt,indu])
        ## prediction[0][0] since array
        class_one_predictions.append(prediction[0][0])

    # Find all the explanation model features used. Defines the dimension d'
    features_dict = {}
    feature_iter = 0
    for exp in explanations:
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, _ in exp.as_list(label=label):
                if feature not in features_dict.keys():
                    features_dict[feature] = (feature_iter)
                    feature_iter += 1
    d_prime = len(features_dict.keys())
    
    def getList(dict):
        return list(dict.keys())
    
    feature_list = getList(features_dict)
    
    # Create the n x d' dimensional 'explanation matrix', W
    W = np.zeros((len(explanations), d_prime))
    for i, exp in enumerate(explanations):
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, value in exp.as_list(label):
                # print(i, feature, value)
                # order of W (nxd) with n being explainer and d being words
                # is d according to features dictionary (0 to end)
                W[i, features_dict[feature]] += value 
    
    ## include here prediction and actual y value for each instance into W
    
    df_w = pd.DataFrame(W, columns=feature_list)
    
    return df_w, class_one_predictions


''' positive class '''
df_w, prob_one = weigth_mtx_pos_test(one_doc,one_doc_sparse)

df_abs_importance = df_w.abs().sum()
df_importance = df_w.sum()

df_w['label_prob'] = prob_one
df_w['label_true'] = one

df_w.to_csv(os.path.join(savemodelpath,'lime_weight_matrix_test_set_positive_class.txt'),
            sep=',', encoding='utf-8')
df_abs_importance.to_csv(os.path.join(savemodelpath,'lime_abs_feature_importance_test_set_positive_class.txt'),
            sep=',', encoding='utf-8')
df_importance.to_csv(os.path.join(savemodelpath,'lime_feature_importance_test_set_positive_class.txt'),
            sep=',', encoding='utf-8')



########################
''' For training data'''
########################


model.load_weights(model_loads[best_train_index])
model.compile(loss='binary_crossentropy',
              optimizer='adam', # opt
          metrics=[TruePositives(name='true_positives'),
                   TrueNegatives(name='true_negatives'),
                   FalseNegatives(name='false_negatives'),
                   FalsePositives(name='false_positives'),
          #         #Accuracy(name='accuracy')
                   BinaryAccuracy(name='binary_accuracy',
                                  dtype=None, threshold=0.5)
                  ])

# for training data
y_pred_val =  model.predict([sparse_x_val,mktcap_val,industry_val],
                              verbose=0)  # gives probabilities



def weigth_mtx_train(data, sparse_data):
    ''' runs several lime explainers
    
    holds mktcap and industry constant!
    
    and return local lime values for each instance
    and predicted class probability of each instance '''    
    sample_indices = np.arange(len(data))
    # Generate Explanations
    explanations = []
    class_one_predictions = []
    
    for i in sample_indices:
        print(i)
        def predict_prob(string):
            ''' must take list of d strings and output (d,k) numpy array
                with prediction probabilities, where k is the number of classes
            
                mkt must be mktcap_train[instance_index]
                ind must be industry_train[instance_index]
            '''
            global model
            x_temp = t.texts_to_sequences(string)
            x_temp = pad_sequences(x_temp, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            x_temp = x_temp.astype('float32')
            
            shape_n = x_temp.shape[0]
            market = []
            indust = []
            for n in range(0,shape_n):
                ## for length of string (shape_n), create market
                ## filled with instance i
                market.append(mktcap_train[i])
                indust.append(list(industry_train[i]))
            market = np.array(market, dtype='float32')
            indust = np.array(indust, dtype='float32')
            
            prediction = model.predict([x_temp,market,indust])
            class_zero = 1-prediction
            probability= np.append(class_zero, prediction, axis=1)
        
            return probability ## array [1-p, p]
        
        explanations.append(explainer.explain_instance(data[i],
                                                       predict_prob, num_features=10))
        ## bring everything into shape for instance prediction        
        temp = sparse_data[i]
        temp = temp.reshape((1,len(temp)))
        temp = temp.astype('float32')
        mrkt = np.array(mktcap_train[i], dtype='float32')
        mrkt = mrkt.reshape((1,1))
        mrkt = mrkt.astype('float32')
        indu = industry_train[i]
        indu = indu.reshape((1,len(indu)))
        indu = indu.astype('float32')
        
        prediction = model.predict([temp,mrkt,indu])
        ## prediction[0][0] since array
        class_one_predictions.append(prediction[0][0])

    # Find all the explanation model features used. Defines the dimension d'
    features_dict = {}
    feature_iter = 0
    for exp in explanations:
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, _ in exp.as_list(label=label):
                if feature not in features_dict.keys():
                    features_dict[feature] = (feature_iter)
                    feature_iter += 1
    d_prime = len(features_dict.keys())
    
    def getList(dict):
        return list(dict.keys())
    
    feature_list = getList(features_dict)
    
    # Create the n x d' dimensional 'explanation matrix', W
    W = np.zeros((len(explanations), d_prime))
    for i, exp in enumerate(explanations):
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, value in exp.as_list(label):
                # print(i, feature, value)
                # order of W (nxd) with n being explainer and d being words
                # is d according to features dictionary (0 to end)
                W[i, features_dict[feature]] += value 
    
    ## include here prediction and actual y value for each instance into W
    
    df_w = pd.DataFrame(W, columns=feature_list)
    
    return df_w, class_one_predictions


percentage = round(len(doc_training) * 0.01)
print(percentage) ## 10 percent of data

y_train_list = list(y_train)
y_mkt_train = list(mktcap_train)
temp = {'doc':doc_training,'label':y_train_list,'mkt':y_mkt_train}
df_explainer = pd.DataFrame(temp)

df_zero_class = df_explainer[df_explainer['label']==0]
df_one_class = df_explainer[df_explainer['label']==1]

one = df_one_class['label'].iloc[:percentage] ### <<< HERE select 10% of zero
one = np.array(one, dtype='float32')
one_doc = df_one_class['doc'].iloc[:percentage] ### <<< HERE select 10% of zero
one_doc = list(one_doc)
one_doc_sparse = t.texts_to_sequences(one_doc)
one_doc_sparse = pad_sequences(one_doc_sparse, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
one_doc_sparse = one_doc_sparse.astype('float32')
mkt_one = df_one_class['mkt'].iloc[:percentage]
mkt_one = np.array(mkt_one, dtype='float32')
index_one = df_one_class.index.tolist()
ind_one = industry_train[index_one,:]

zero = df_zero_class['label'].iloc[:percentage] ### <<< HERE select 10% of zero
zero = np.array(zero, dtype='float32')
zero_doc = df_zero_class['doc'].iloc[:percentage] ### <<< HERE select 10% of zero
zero_doc = list(zero_doc)
zero_doc_sparse = t.texts_to_sequences(zero_doc)
zero_doc_sparse = pad_sequences(zero_doc_sparse, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
zero_doc_sparse = zero_doc_sparse.astype('float32')
mkt_zero = df_zero_class['mkt'].iloc[:percentage]
mkt_zero = np.array(mkt_zero, dtype='float32')
index_zero = df_zero_class.index.tolist() ## get indices from df
ind_zero = industry_train[index_zero,:] ## selects rows according to indices from df


both = df_explainer['label'].iloc[:percentage] ### <<< HERE select 10% of zero
both = np.array(both, dtype='float32')
both_doc = df_explainer['doc'].iloc[:percentage] ### <<< HERE select 10% of zero
both_doc = list(both_doc)
both_doc_sparse = t.texts_to_sequences(both_doc)
both_doc_sparse = pad_sequences(both_doc_sparse, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
both_doc_sparse = both_doc_sparse.astype('float32')


''' both classes '''
df_w, prob_one = weigth_mtx_train(both_doc,both_doc_sparse)

df_abs_importance = df_w.abs().sum()
df_importance = df_w.sum()

df_w['label_prob'] = prob_one
df_w['label_true'] = both

df_w.to_csv(os.path.join(savemodelpath,'lime_weight_matrix_training_set_both_classes.txt'),
            sep=',', encoding='utf-8')
df_abs_importance.to_csv(os.path.join(savemodelpath,'lime_abs_feature_importance_training_set_both_classes.txt'),
            sep=',', encoding='utf-8')
df_importance.to_csv(os.path.join(savemodelpath,'lime_feature_importance_training_set_both_classes.txt'),
            sep=',', encoding='utf-8')



def weigth_mtx_neg_train(data, sparse_data):
    ''' runs several lime explainers
    
    holds mktcap and industry constant!
    
    and return local lime values for each instance
    and predicted class probability of each instance '''    
    sample_indices = np.arange(len(data))
    # Generate Explanations
    explanations = []
    class_one_predictions = []
    
    for i in sample_indices:
        print(i)
        def predict_prob(string):
            ''' must take list of d strings and output (d,k) numpy array
                with prediction probabilities, where k is the number of classes
            
                mkt must be mktcap_train[instance_index]
                ind must be industry_train[instance_index]
            '''
            global model
            x_temp = t.texts_to_sequences(string)
            x_temp = pad_sequences(x_temp, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            x_temp = x_temp.astype('float32')
            
            shape_n = x_temp.shape[0]
            market = []
            indust = []
            for n in range(0,shape_n):
                ## for length of string (shape_n), create market
                ## filled with instance i
                market.append(mkt_zero[i])
                indust.append(list(ind_zero[i]))
            market = np.array(market, dtype='float32')
            indust = np.array(indust, dtype='float32')
            
            prediction = model.predict([x_temp,market,indust])
            class_zero = 1-prediction
            probability= np.append(class_zero, prediction, axis=1)
        
            return probability ## array [1-p, p]
        
        explanations.append(explainer.explain_instance(data[i],
                                                       predict_prob, num_features=10))
        ## bring everything into shape for instance prediction        
        temp = sparse_data[i]
        temp = temp.reshape((1,len(temp)))
        temp = temp.astype('float32')
        mrkt = np.array(mkt_zero[i], dtype='float32') ###########
        mrkt = mrkt.reshape((1,1))
        mrkt = mrkt.astype('float32')
        indu = ind_zero[i] ###########
        indu = indu.reshape((1,len(indu)))
        indu = indu.astype('float32')
        
        prediction = model.predict([temp,mrkt,indu])
        ## prediction[0][0] since array
        class_one_predictions.append(prediction[0][0])

    # Find all the explanation model features used. Defines the dimension d'
    features_dict = {}
    feature_iter = 0
    for exp in explanations:
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, _ in exp.as_list(label=label):
                if feature not in features_dict.keys():
                    features_dict[feature] = (feature_iter)
                    feature_iter += 1
    d_prime = len(features_dict.keys())
    
    def getList(dict):
        return list(dict.keys())
    
    feature_list = getList(features_dict)
    
    # Create the n x d' dimensional 'explanation matrix', W
    W = np.zeros((len(explanations), d_prime))
    for i, exp in enumerate(explanations):
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, value in exp.as_list(label):
                # print(i, feature, value)
                # order of W (nxd) with n being explainer and d being words
                # is d according to features dictionary (0 to end)
                W[i, features_dict[feature]] += value 
    
    ## include here prediction and actual y value for each instance into W
    
    df_w = pd.DataFrame(W, columns=feature_list)
    
    return df_w, class_one_predictions



''' negative class '''
df_w, prob_one = weigth_mtx_neg_train(zero_doc,zero_doc_sparse)

df_abs_importance = df_w.abs().sum()
df_importance = df_w.sum()

df_w['label_prob'] = prob_one
df_w['label_true'] = zero

df_w.to_csv(os.path.join(savemodelpath,'lime_weight_matrix_training_set_negative_class.txt'),
            sep=',', encoding='utf-8')
df_abs_importance.to_csv(os.path.join(savemodelpath,'lime_abs_feature_importance_training_set_negative_class.txt'),
            sep=',', encoding='utf-8')
df_importance.to_csv(os.path.join(savemodelpath,'lime_feature_importance_training_set_negative_class.txt'),
            sep=',', encoding='utf-8')




def weigth_mtx_pos_train(data, sparse_data):
    ''' runs several lime explainers
    
    holds mktcap and industry constant!
    
    and return local lime values for each instance
    and predicted class probability of each instance '''    
    sample_indices = np.arange(len(data))
    # Generate Explanations
    explanations = []
    class_one_predictions = []
    
    for i in sample_indices:
        print(i)
        def predict_prob(string):
            ''' must take list of d strings and output (d,k) numpy array
                with prediction probabilities, where k is the number of classes
            
                mkt must be mktcap_train[instance_index]
                ind must be industry_train[instance_index]
            '''
            global model
            x_temp = t.texts_to_sequences(string)
            x_temp = pad_sequences(x_temp, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            x_temp = x_temp.astype('float32')
            
            shape_n = x_temp.shape[0]
            market = []
            indust = []
            for n in range(0,shape_n):
                ## for length of string (shape_n), create market
                ## filled with instance i
                market.append(mkt_one[i])
                indust.append(list(ind_one[i]))
            market = np.array(market, dtype='float32')
            indust = np.array(indust, dtype='float32')
            
            prediction = model.predict([x_temp,market,indust])
            class_zero = 1-prediction
            probability= np.append(class_zero, prediction, axis=1)
        
            return probability ## array [1-p, p]
        
        explanations.append(explainer.explain_instance(data[i],
                                                       predict_prob, num_features=10))
        ## bring everything into shape for instance prediction        
        temp = sparse_data[i]
        temp = temp.reshape((1,len(temp)))
        temp = temp.astype('float32')
        mrkt = np.array(mkt_one[i], dtype='float32') ###########
        mrkt = mrkt.reshape((1,1))
        mrkt = mrkt.astype('float32')
        indu = ind_one[i] ###########
        indu = indu.reshape((1,len(indu)))
        indu = indu.astype('float32')
        
        prediction = model.predict([temp,mrkt,indu])
        ## prediction[0][0] since array
        class_one_predictions.append(prediction[0][0])

    # Find all the explanation model features used. Defines the dimension d'
    features_dict = {}
    feature_iter = 0
    for exp in explanations:
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, _ in exp.as_list(label=label):
                if feature not in features_dict.keys():
                    features_dict[feature] = (feature_iter)
                    feature_iter += 1
    d_prime = len(features_dict.keys())
    
    def getList(dict):
        return list(dict.keys())
    
    feature_list = getList(features_dict)
    
    # Create the n x d' dimensional 'explanation matrix', W
    W = np.zeros((len(explanations), d_prime))
    for i, exp in enumerate(explanations):
        labels = exp.available_labels() # if exp.mode == 'classification' else [1]
        for label in labels:
            for feature, value in exp.as_list(label):
                # print(i, feature, value)
                # order of W (nxd) with n being explainer and d being words
                # is d according to features dictionary (0 to end)
                W[i, features_dict[feature]] += value 
    
    ## include here prediction and actual y value for each instance into W
    
    df_w = pd.DataFrame(W, columns=feature_list)
    
    return df_w, class_one_predictions


''' positive class '''
df_w, prob_one = weigth_mtx_pos_train(one_doc,one_doc_sparse)

df_abs_importance = df_w.abs().sum()
df_importance = df_w.sum()

df_w['label_prob'] = prob_one
df_w['label_true'] = one

df_w.to_csv(os.path.join(savemodelpath,'lime_weight_matrix_training_set_positive_class.txt'),
            sep=',', encoding='utf-8')
df_abs_importance.to_csv(os.path.join(savemodelpath,'lime_abs_feature_importance_training_set_positive_class.txt'),
            sep=',', encoding='utf-8')
df_importance.to_csv(os.path.join(savemodelpath,'lime_feature_importance_training_set_positive_class.txt'),
            sep=',', encoding='utf-8')

