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
bertdocs = 'C:/Users/.../NumpyBERT/'

## define folder for saving bert models
savemodel = 'C:/Users/.../saved_models/bert/'

## define file containing nltk stopwords
stopwords = 'C:/Users/.../nltk_stopwords.txt'

## define according financial label
label = 'joint_ep90_eps'    ## << HERE 'joint_EP_bhar', 'joint_EP_bhar_short',
                           ##         'joint_ep90_eps'

''' Set params for neural networks'''
epoch = 50
batch = 32
denser = 100
regu = 0.1 # 0.01 fitted nicht
drop = 0.5

no_sentences_per_doc = 1000 #1000
sentence_embedding = 768  #1000

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
            
# Basic Attention
class Attention(Layer):
    '''
    Source:
    https://github.com/stevewyl/comparative-reviews-classification/blob/master/layers.py
    '''
    def __init__(self, attention_size, return_coefficients=False, **kwargs):
        self.attention_size = attention_size
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)
        
        if self.return_coefficients:
            return [output, atx]
        else:
            return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])



#from tensorflow.keras.utils import Sequence
class multi_input_generator_env(Sequence):
    ''' 
    Generates data for Keras
    
    list_IDs =  a list of npy. files to load
    labels   =  a dictionary of labels {'filename1.npy':1,'filename1.npy':0,...etc}
    filepath =  for example 'C:/Users/ac129731/Desktop/Dissertation_Doktor_Arbeit/Edgar_Files/SP500_merged/Needed_for_Bert/testing_generator/'
    '''
    
    def __init__(self, list_IDs, env, labels, filepath, batch_size=32, sentence_length=1000, features=768, ind_dim=8, shuffle=True, to_fit=True):
        ''' initialization '''
        self.list_IDs = list_IDs
        self.env = env
        self.labels = labels
        self.batch_size = batch_size
        self.sentence_length = sentence_length 
        self.features = features
        self.ind_dim = ind_dim
        self.filepath = filepath
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()
        
    def __len__(self):
        ''' Denotes the number of batches per epoch '''
        
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        ''' 
        Generate one batch of data
        :param index: index of the batch; is created when called!
        :return: X and y when fitting. X only when predicting
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self._generate_data(list_IDs_temp)
        

        if self.to_fit:
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes) # suffles list IN PLACE! so does NOT create new list 
            
            
    def _generate_data(self, list_IDs_temp):
        '''
        Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        
        list_IDs_temp is created when __getitem__ is called
        
        '''
        # Initialization
        X = np.empty((self.batch_size, self.sentence_length, self.features))
        y = np.empty((self.batch_size), dtype=int)
        
        environmental = np.empty((self.batch_size), dtype=int)
        
        for i, ID in enumerate(list_IDs_temp):
            # i is a number;
            # ID is the file-name
               
            # load single file
            single_file = np.load(os.path.join(self.filepath,ID))
            ## create empty array to contain batch of features and labels
            batch_features = np.zeros((self.sentence_length, self.features))
            
            #####
            # to allow for shorter than 1000-sentences
            single_file = single_file[:self.sentence_length,:self.features]
            
            #####
            
            # pad loaded array to same length        
            shape = np.shape(single_file)
            batch_features[:shape[0],:shape[1]] = single_file 
            
            ## append to sequence
            X[i,] = batch_features
            
            environmental[i] = self.env[ID]
            
            y[i] = self.labels[ID] ### this looks-up according rating (Note ID = file name.npy)
            
        return [X, environmental], y

      
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

## Replace txt with npy
validation_all['np_name'] = validation_all['fname'].str.replace('txt','npy', n=1) # need to replace npy ending with txt for merging
train_all['np_name'] = train_all['fname'].str.replace('txt','npy', n=1) # need to replace npy ending with txt for merging
test_all['np_name'] = test_all['fname'].str.replace('txt','npy', n=1) # need to replace npy ending with txt for merging

validation_all['fileload'] = validation_all[['path', 'np_name']].apply(lambda x: ''.join(x), axis=1)
train_all['fileload'] = train_all[['path', 'np_name']].apply(lambda x: ''.join(x), axis=1)
test_all['fileload'] = test_all[['path', 'np_name']].apply(lambda x: ''.join(x), axis=1)


''' save these to list and check unique counts'''
x_test, y_test_EP, y_test_bhar, y_test_ep90, y_test_eps, y_test_bhar_short = test_all['fileload'].to_list(), test_all['EP_label'].to_list(), test_all['BHAR_label'].to_list(), test_all['label_env'].to_list(), test_all['label_eps'].to_list(), test_all['BHAR_Short_label'].to_list()
x_train, y_train_EP, y_train_bhar, y_train_ep90, y_train_eps, y_train_bhar_short = train_all['fileload'].to_list(), train_all['EP_label'].to_list(), train_all['BHAR_label'].to_list(), train_all['label_env'].to_list(), train_all['label_eps'].to_list(), train_all['BHAR_Short_label'].to_list()
x_val, y_val_EP, y_val_bhar, y_val_ep90, y_val_eps, y_val_bhar_short = validation_all['fileload'].to_list(), validation_all['EP_label'].to_list(), validation_all['BHAR_label'].to_list(), validation_all['label_env'].to_list(), validation_all['label_eps'].to_list(), validation_all['BHAR_Short_label'].to_list()




if label == 'joint_EP_bhar':
    ## bhar + EP
    y_train = y_train_bhar 
    y_test = y_test_bhar   
    y_val = y_val_bhar
    
    golden_env_train = y_train_EP 
    golden_env_test = y_test_EP
    golden_env_val = y_val_EP
    print(label)

elif label == 'joint_EP_bhar_short':
    ## bhar_short + EP
    y_train = y_train_bhar_short 
    y_test = y_test_bhar_short
    y_val = y_val_bhar_short
    
    golden_env_train = y_train_EP 
    golden_env_test = y_test_EP
    golden_env_val = y_val_EP
    print(label)

else:
    ## eps + ep90
    y_train = y_train_eps 
    y_test = y_test_eps
    y_val = y_val_eps
    
    golden_env_train = y_train_ep90 
    golden_env_test = y_test_ep90
    golden_env_val = y_val_ep90
    print(label)
    
    
label_ = label+'/'
savemodelpath = os.path.join(savemodel,label_)


# industry_dim = industry_train.shape[1]

y_train = np.array(y_train, dtype='float32')
y_test = np.array(y_test, dtype='float32')
y_val = np.array(y_val, dtype='float32')

golden_env_train = np.array(golden_env_train, dtype='float32')
golden_env_test = np.array(golden_env_test, dtype='float32')
golden_env_val = np.array(golden_env_val, dtype='float32')


X_train, X_test, X_val = np.array(x_train, dtype='str'), np.array(x_test, dtype='str'), np.array(x_val, dtype='str')

y_label_train = dict(zip(X_train,y_train)) # labels is actual input for generator
y_label_val = dict(zip(X_val,y_val)) # labels is actual input for generator
y_label_test = dict(zip(X_test,y_test)) # labels is actual input for generator


golden_env_train = dict(zip(X_train,golden_env_train)) # labels is actual input for generator
golden_env_test = dict(zip(X_test,golden_env_test)) # labels is actual input for generator
golden_env_val = dict(zip(X_val,golden_env_val)) # labels is actual input for generator



def create_model(loadweight=None):
    ##### MODEL #####
    sequence_input  = Input(shape=(no_sentences_per_doc, sentence_embedding))

    ### environmental performance
    env_perf = Input(shape=(1,), dtype='float32')
    
    gru_layer = Bidirectional(GRU(50, #activation='tanh',
                              return_sequences=True#True
                              ))(sequence_input)
    
    ### consider putting here a TimeDistributed Dense layer... l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    ### TimeDistributed applies a layer to every temporal slice of an input; input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension
    sent_dense = Dense(100, activation='relu', name='sent_dense')(gru_layer)  # make signal stronger
    
    sent_att,sent_coeffs = Attention(100,return_coefficients=True,name='sent_attention')(sent_dense)
    sent_at = Dropout(0.5,name='sent_dropout')(sent_att)

    ## combine market cap and industry with cnn
    combined = Concatenate()([sent_at, env_perf])

    preds = Dense(1, activation='sigmoid',name='output')(combined)  # NOTE: SIGMOID FOR 2-CLASS PROBLEM; softmax for multiclass problem
    model = Model([sequence_input,env_perf], preds)

    
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


training_generator = multi_input_generator_env(X_train, golden_env_train, y_label_train,
                                               bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                               features=sentence_embedding, shuffle=False, to_fit=True)
validation_generator = multi_input_generator_env(X_val, golden_env_val, y_label_val,
                                                 bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                 features=sentence_embedding, shuffle=False, to_fit=True)  # to_fit returns X and y


history = model.fit(x=training_generator, epochs=epoch,
                    validation_data=validation_generator,
                    #use_multiprocessing=True,
                    #workers=num_workers, # check if workers should be set to 1 or other on colab
                    callbacks=callbacks_list)



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

f1_sc, f1_sc_val = [], []
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


training_generator_predict = multi_input_generator_env(X_train, golden_env_train, y_label_train,
                                                       bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                       features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y

validation_generator_predict = multi_input_generator_env(X_val, golden_env_val, y_label_val,
                                                         bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                         features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y


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
    percent = cnt / epoch
    print("Progress: ",percent)
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
    # for validation data
    y_pred = model.predict(validation_generator_predict, verbose=0)
    # y_pred = model.predict_generator(validation_generator_predict, verbose=0)
    val_preds = [1 if x > 0.5 else 0 for x in y_pred]
    
    ## see Careful-note
    yy = len(val_preds)
    cm, p, r, tp_total, fp_total, fn_total = calc_metrics(val_preds, y_val[0:yy])#y_val)
    Fscore_val.append(cm)
    precision_val.append(p)
    recall_val.append(r)    
    tp_val_total.append(tp_total)
    fp_val_total.append(fp_total)
    fn_val_total.append(fn_total)
    ff_val = f1_score(y_val[0:yy], val_preds)
    fscorepredict_val_sklearn.append(ff_val)
    
    
    # for training data 
    y_pred =  model.predict(training_generator_predict, verbose=0)  # gives probabilities
    # y_pred =  model.predict_generator(training_generator_predict, verbose=0)  # gives probabilities
    train_preds = [1 if x > 0.5 else 0 for x in y_pred]
    yy = len(train_preds)
    
    cm, p_tr, r_tr, tp_total, fp_total, fn_total = calc_metrics(train_preds, y_train[0:yy])
    Fscore_train.append(cm)
    precision_train.append(p_tr)
    recall_train.append(r_tr)
    
    tp_train_total.append(tp_total)
    fp_train_total.append(fp_total)
    fn_train_total.append(fn_total)
    ff = f1_score(y_train[0:yy], train_preds)
    fscorepredict_train.append(ff)
    
    fscorepredict_train.append(cm)
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

y_pred_val =  model.predict(validation_generator_predict, verbose=0)  # gives probabilities
saved_val_preds = [1 if x > 0.5 else 0 for x in y_pred_val]
yy = len(saved_val_preds)

## save bootstraps for best validation models
bootstrap_val = TP_FP_FN_forBootstrap(saved_val_preds,y_val[0:yy])
bootstrap_val.to_csv(os.path.join(savemodelpath,'_bootstrap_val.txt'),sep=',',encoding='utf-8')

saved_val_preds = pd.DataFrame(saved_val_preds, columns=['y_val_preds'])
saved_val_preds.to_csv(os.path.join(savemodelpath,'_saved_val_preds.txt'),sep=',',encoding='utf-8')


test_generator_predict = multi_input_generator_env(X_test, golden_env_test, y_label_test,
                                                   bertdocs, batch_size=batch, sentence_length=no_sentences_per_doc,
                                                   features=sentence_embedding, shuffle=False, to_fit=False)  # to_fit returns X and y


y_pred_test =  model.predict(test_generator_predict,
                              verbose=0)  # gives probabilities
test_preds = [1 if x > 0.5 else 0 for x in y_pred_test]
yy = len(test_preds)
f_test,pr_test, re_test, tp_test, fp_test, fn_test = calc_metrics(test_preds, y_test[0:yy])


bootstrap_test = TP_FP_FN_forBootstrap(test_preds,y_test[0:yy])
bootstrap_test.to_csv(os.path.join(savemodelpath,'_bootstrap_test.txt'),sep=',',encoding='utf-8')

test_scores = {'f_test':f_test,'pr_test':pr_test,'re_test':re_test,
               'tp_test':tp_test, 'fp_test':fp_test, 'fn_test':fn_test, 'best_model':model_loads[best_val_index]}
df5 = pd.DataFrame(test_scores, index=[0])
df5.to_csv(os.path.join(savemodelpath,'best_val_model_on_test_set.txt'), sep=',', encoding='utf-8')

















