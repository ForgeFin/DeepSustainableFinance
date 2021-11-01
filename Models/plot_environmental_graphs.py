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

## define path with environmental word list
env_words = 'C:/.../env_wordlist.txt'

## define folder for saving bert models
savemodel = 'C:/Users/.../saved_models/bert/'

## define file containing nltk stopwords
stopwords = 'C:/Users/.../nltk_stopwords.txt'

## define according financial label
label = 'BHAR_Short_label'    ## << HERE 'BHAR_label', 'label_eps', BHAR_Short_label


#################################################################################
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


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


print("Loading text to memory")
with open(stopwords, encoding='utf-8') as document:
    stopwords = document.read().split(",")
stopwords = set(stopwords)


env_word_list = pd.read_csv(env_words, sep=',', index_col=0, encoding='utf-8')
env_word_list = env_word_list.reset_index(level='words')
env_word_list = list(env_word_list['words'])


def cleanstr(file_list):
    text = []
    word_count = []
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
            doc_len = len(tokens)
            word_count.append(doc_len)
            tokens = [w for w in tokens if not w in stopwords]
            docs = " ".join(tokens)
            text.append(docs)
    return text, word_count

## documents as str in list
doc_training, count_train = cleanstr(x_train)
doc_test, count_test = cleanstr(x_test)
doc_val, count_val = cleanstr(x_val)



vectorizer = CountVectorizer(ngram_range=(1,2))
sparse_x = vectorizer.fit_transform(doc_training)
featurenames = vectorizer.get_feature_names()
df_sparse_x = pd.DataFrame(sparse_x.toarray(),columns=featurenames)
df_sparse_x = df_sparse_x[env_word_list]
''' Following deletes double counts of climate and warming'''
df_sparse_x['climate'] = df_sparse_x['climate'] - df_sparse_x['climate change']
df_sparse_x['warming'] = df_sparse_x['warming'] - df_sparse_x['global warming']
df_sparse_x['greenhouse'] = df_sparse_x['greenhouse'] - df_sparse_x['greenhouse gas']
df_sparse_x['greenhouse'] = df_sparse_x['greenhouse'] - df_sparse_x['greenhouse gases']
sparse_x = np.array(df_sparse_x, dtype='float32')


sparse_x_test = vectorizer.transform(doc_test)
df_sparse_x_test = pd.DataFrame(sparse_x_test.toarray(),columns=featurenames)
df_sparse_x_test = df_sparse_x_test[env_word_list]
df_sparse_x_test['climate'] = df_sparse_x_test['climate'] - df_sparse_x_test['climate change']
df_sparse_x_test['warming'] = df_sparse_x_test['warming'] - df_sparse_x_test['global warming']
df_sparse_x_test['greenhouse'] = df_sparse_x_test['greenhouse'] - df_sparse_x_test['greenhouse gas']
df_sparse_x_test['greenhouse'] = df_sparse_x_test['greenhouse'] - df_sparse_x_test['greenhouse gases']
sparse_x_test = np.array(df_sparse_x_test, dtype='float32')


sparse_x_val = vectorizer.transform(doc_val)
df_sparse_x_val = pd.DataFrame(sparse_x_val.toarray(),columns=featurenames)
df_sparse_x_val = df_sparse_x_val[env_word_list]
df_sparse_x_val['climate'] = df_sparse_x_val['climate'] - df_sparse_x_val['climate change']
df_sparse_x_val['warming'] = df_sparse_x_val['warming'] - df_sparse_x_val['global warming']
df_sparse_x_val['greenhouse'] = df_sparse_x_val['greenhouse'] - df_sparse_x_val['greenhouse gas']
df_sparse_x_val['greenhouse'] = df_sparse_x_val['greenhouse'] - df_sparse_x_val['greenhouse gases']
sparse_x_val = np.array(df_sparse_x_val, dtype='float32')



## total word count, % of document

dat_train = train_all.merge(data_all_rank[['fname','Comp_name','Date_filing','Form','yy','Subclass','Industry']], on=['fname'], how='inner') # An inner merge, (or inner join) keeps only the common values in both the left and right dataframes for the result.
dat_test = test_all.merge(data_all_rank[['fname','Comp_name','Date_filing','Form','yy','Subclass','Industry']], on=['fname'], how='inner') # An inner merge, (or inner join) keeps only the common values in both the left and right dataframes for the result.
dat_val = validation_all.merge(data_all_rank[['fname','Comp_name','Date_filing','Form','yy','Subclass','Industry']], on=['fname'], how='inner') # An inner merge, (or inner join) keeps only the common values in both the left and right dataframes for the result.

dat = dat_train.append(dat_test)
dat = dat.append(dat_val)
wrd = df_sparse_x.append(df_sparse_x_test)
wrd = wrd.append(df_sparse_x_val)

result = pd.concat([dat, wrd], axis=1)

## new df for relative cnt
df_sparse_x['twrds_doc'] = count_train # total words in document
df_sparse_x_test['twrds_doc'] = count_test # total words in document
df_sparse_x_val['twrds_doc'] = count_val # total words in document
wrd_rel = df_sparse_x.append(df_sparse_x_test)
wrd_rel = wrd_rel.append(df_sparse_x_val)
result3 = pd.concat([dat, wrd_rel], axis=1)

## mentiones in percentage % per doc
wrd_rel_per_doc = wrd_rel[env_word_list].div(wrd_rel['twrds_doc'], axis=0) * 100
result2 = pd.concat([dat, wrd_rel_per_doc], axis=1)


df_wrd_sum = wrd.sum()
summe = wrd.sum()
summe_terms = summe.sum()
rel_terms = (summe / summe_terms) * 100
# summe = (df_wrd_sum / df_wrd_sum['twrds_doc']) * 100

envwords = pd.DataFrame(df_wrd_sum) 
envwords = envwords.reset_index()
envwords = envwords.rename(columns={0:"cnt","index":"words"})
zahl = df_wrd_sum.sum()
envwords['relative'] = envwords['cnt'] / zahl
envwords['relative'] = envwords['relative'] * 100
envwords = envwords.sort_values('relative',ascending=True)


''' Term used most often: No.Term.occurs / TotalNumber all Terms occur'''
envwords_cnt = list(envwords['words'])
density = list(envwords['relative'])
year = np.arange(len(envwords_cnt))
fig = plt.figure(figsize=(8.5,4.5))
plt.rcParams['font.size'] = '22'
plt.bar(year, density, tick_label=envwords_cnt)
ax = plt.axes()


plt.xticks(year, envwords_cnt, rotation=45, ha="right")#, size='small')
plt.yticks(np.arange(0, 45, 5.0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')

plt.show() # Show graphic



#
''' 
group result2 by year yy 
and then by industry
'''
# wrd[env_word_list]  = wrd[env_word_list].div(wrd[env_word_list].sum(axis=1), axis=0)


### Corpus
## x - year, y - no.climate-words / totalnumber of words (ex. numbers)


result['cnt_occ'] = 1
no_doc = result.groupby(['yy'])['cnt_occ'].sum() 
no_doc = list(no_doc)



''' Companies that report '''
result3['sum_env_wrd'] = result3[env_word_list].sum(axis=1)
result3['env_wrd_rel'] = result3['sum_env_wrd'] / result3['twrds_doc']
result3['cnt_env'] = result3['sum_env_wrd'] / result3['sum_env_wrd']
result3['cnt_env']  = result3['cnt_env'].fillna(0)
exists = result3.groupby(['yy'])['cnt_env'].sum() 
exists = list(exists)
dic = {'exists':exists,'no_doc':no_doc} 
dic = pd.DataFrame(dic)
dic['perc'] = dic['exists'] / dic['no_doc']
dic['perc'] = dic['perc'] * 100
perc = list(dic['perc'])
# xxx[env_word_list]  = xxx[env_word_list].div(xxx[['twrds_doc']].sum(axis=1), axis=0)
# xxx[env_word_list] = xxx[env_word_list] * 100

result3['ten_or_more'] = result3['sum_env_wrd'].where(result3['sum_env_wrd'] >= 10, 0)
result3['ten_or_more'] = result3['ten_or_more'] / result3['ten_or_more']
result3['ten_or_more']  = result3['ten_or_more'].fillna(0)
ten_or_more = result3.groupby(['yy'])['ten_or_more'].sum() 
ten_or_more = list(ten_or_more)
ten_or_more = {'exists':ten_or_more,'no_doc':no_doc} 
ten_or_more = pd.DataFrame(ten_or_more)
ten_or_more['perc'] = ten_or_more['exists'] / ten_or_more['no_doc']
ten_or_more['perc'] = ten_or_more['perc'] * 100
ten_perc = list(ten_or_more['perc'])

diff = {'ten_per':ten_perc,'one_per':perc}
diff = pd.DataFrame(diff)
diff['diff'] = diff['one_per'] - diff['ten_per']
difference = list(diff['diff'])
# set width of bars
barWidth = 0.45
plt.rcParams['font.size'] = '22'

''' 
Percentage of documents with 1 or more mentiones of keywords
cnt(keyword) / doc_in_given_year
'''

year = ['2014','2015','2016','2017','2018']
# Set position of bar on X axis
r1 = np.arange(len(year))

r2 = [x + barWidth for x in r1]
# fig = plt.figure(1, (7,4))
# ax = fig.add_subplot(1,1,1)
fig = plt.figure(figsize=(8.5,4.5))
ax = plt.axes()
# Make the plot
plt.bar(r1, ten_perc, color='goldenrod', width=barWidth, edgecolor='white', label='10 or more')
plt.bar(r2, perc, color='darkblue', width=barWidth, edgecolor='white', label='1 or more')

for i in ax.patches:
    tt = "%s%%" % round((i.get_height()), 1)
    ax.text(i.get_x(),i.get_height(), tt, 
            color='black', ha="left", fontsize=22) #ha="center"  #rotation=40
# Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
# Text on the top of each bar
tz = barWidth / 2
plt.xticks([r + tz for r in range(len(year))], ['2014','2015','2016','2017','2018'])
# plt.xticks(r1, ['2014','2015','2016','2017','2018'])
# Create legend & Show graphic
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.ylim(0,100)
plt.legend(loc="upper right")
# plt.ylabel("%", fontsize=18) 
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()



## x - industry, y - no.climate-words / totalnumber of words (ex. numbers)
test = result3.groupby(['yy', 'Industry_numeric'])['cnt_env'].sum()
test3 = result3.groupby(['yy', 'Industry_numeric'])['ten_or_more'].sum()

test2 = result.groupby(['yy', 'Industry_numeric'])['cnt_occ'].sum()
result4 = pd.concat([test, test3, test2], axis=1)
result4['perc_ten_more'] = result4['ten_or_more'] / result4['cnt_occ']
result4['perc_one_more'] = result4['cnt_env'] / result4['cnt_occ']
result4['perc_ten_more'] = result4['perc_ten_more'] * 100
result4['perc_one_more'] = result4['perc_one_more'] * 100

result4 = result4.reset_index(level='Industry_numeric')
industry1 = result4[result4['Industry_numeric'] == 1]
industry1_ten = list(industry1['perc_ten_more'])
industry1_one = list(industry1['perc_one_more'])

industry2 = result4[result4['Industry_numeric'] == 2]
industry2_ten = list(industry2['perc_ten_more'])
industry2_one = list(industry2['perc_one_more'])

industry3 = result4[result4['Industry_numeric'] == 3]
industry3_ten = list(industry3['perc_ten_more'])
industry3_one = list(industry3['perc_one_more'])

industry4 = result4[result4['Industry_numeric'] == 4]
industry4_ten = list(industry4['perc_ten_more'])
industry4_one = list(industry4['perc_one_more'])

industry5 = result4[result4['Industry_numeric'] == 5]
industry5_ten = list(industry5['perc_ten_more'])
industry5_one = list(industry5['perc_one_more'])

industry6 = result4[result4['Industry_numeric'] == 6]
industry6_ten = list(industry6['perc_ten_more'])
industry6_one = list(industry6['perc_one_more'])

industry7 = result4[result4['Industry_numeric'] == 7]
industry7_ten = list(industry7['perc_ten_more'])
industry7_one = list(industry7['perc_one_more'])

industry8 = result4[result4['Industry_numeric'] == 8]
industry8_ten = list(industry8['perc_ten_more'])
industry8_one = list(industry8['perc_one_more'])


'''  One ore more mentiones '''
year = ['2014','2015','2016','2017','2018']

# barWidth = 0.1
barWidth = 0.11
# Set position of bar on X axis
r1 = np.arange(len(year))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
r8 = [x + barWidth for x in r7]

plt.rcParams['font.size'] = '24'
fig = plt.figure(1, (13,6))
ax = fig.add_subplot(1,1,1)
# Make the plot
# plt.bar(r1, industry1_one, color='steelblue', width=barWidth, edgecolor='white', label='Finance, Insurance, And Real Estate')
# plt.bar(r2, industry2_one, color='cornflowerblue', width=barWidth, edgecolor='white', label='Manufacturing')
# plt.bar(r3, industry3_one, color='royalblue', width=barWidth, edgecolor='white', label='Transp., Comm., Electric, Gas, & Sanitary')
# plt.bar(r4, industry4_one, color='blue', width=barWidth, edgecolor='white', label='Services')
# plt.bar(r5, industry5_one, color='mediumblue', width=barWidth, edgecolor='white', label='Mining')
# plt.bar(r6, industry6_one, color='darkblue', width=barWidth, edgecolor='white', label='Construction')
# plt.bar(r7, industry7_one, color='midnightblue', width=barWidth, edgecolor='white', label='Wholesale Trade')
# plt.bar(r8, industry8_one, color='indigo', width=barWidth, edgecolor='white', label='Retail Trade')

plt.bar(r1, industry1_one, color='grey', width=barWidth, edgecolor='white', label='Finance, Insurance, Real Estate')
plt.bar(r2, industry2_one, color='firebrick', width=barWidth, edgecolor='white', label='Manufacturing')
plt.bar(r3, industry3_one, color='royalblue', width=barWidth, edgecolor='white', label='Transportation & Public Utilities')
plt.bar(r4, industry4_one, color='magenta', width=barWidth, edgecolor='white', label='Services')
plt.bar(r5, industry5_one, color='darkorange', width=barWidth, edgecolor='white', label='Mining')
plt.bar(r6, industry6_one, color='darkblue', width=barWidth, edgecolor='white', label='Construction')
plt.bar(r7, industry7_one, color='green', width=barWidth, edgecolor='white', label='Wholesale Trade')
plt.bar(r8, industry8_one, color='indigo', width=barWidth, edgecolor='white', label='Retail Trade')

ti = barWidth * 3.5
plt.xticks([r + ti for r in range(len(year))], ['2014','2015','2016','2017','2018'])
# plt.xticks(r1, ['2014','2015','2016','2017','2018'])
# Create legend & Show graphic
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
# plt.title("Environmental Search Terms: One or More Mentions", fontsize=20)
plt.ylim(0,100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0., fontsize=20)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
# plt.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=18)
# plt.ylabel("%", fontsize=18) 
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()





''' 10 or more mentiones '''
year = ['2014','2015','2016','2017','2018']

barWidth = 0.11
# Set position of bar on X axis
r1 = np.arange(len(year))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]
r8 = [x + barWidth for x in r7]


fig = plt.figure(1, (13,6))
ax = fig.add_subplot(1,1,1)

# avrg = sum(industry8_ten) / len(industry5_ten)
# print(avrg)
plt.bar(r1, industry1_ten, color='grey', width=barWidth, edgecolor='white', label='Finance, Insurance, Real Estate')
plt.bar(r2, industry2_ten, color='firebrick', width=barWidth, edgecolor='white', label='Manufacturing')
plt.bar(r3, industry3_ten, color='royalblue', width=barWidth, edgecolor='white', label='Transportation & Public Utilities')
plt.bar(r4, industry4_ten, color='magenta', width=barWidth, edgecolor='white', label='Services')
plt.bar(r5, industry5_ten, color='darkorange', width=barWidth, edgecolor='white', label='Mining')
plt.bar(r6, industry6_ten, color='darkblue', width=barWidth, edgecolor='white', label='Construction')
plt.bar(r7, industry7_ten, color='green', width=barWidth, edgecolor='white', label='Wholesale Trade')
plt.bar(r8, industry8_ten, color='indigo', width=barWidth, edgecolor='white', label='Retail Trade')

plt.xticks([r + 0.35 for r in range(len(year))], ['2014','2015','2016','2017','2018'])
# plt.xticks(r1, ['2014','2015','2016','2017','2018'])
# Create legend & Show graphic
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.ylim(0,70)
# plt.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=18)
plt.yticks([0,10,20,30,40,50,60,70])
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0., fontsize=20)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
# plt.title("Environmental Search Terms: Ten or More Mentions", fontsize=20)
# plt.ylabel("%", fontsize=18) 
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()



'''
## x - 10K & 10Q, y - no.climate-words occurs more than 1(10) / totalnumber of reports (ex. numbers)
'''
new_result = result3.groupby(['yy', 'Form_x'])[['cnt_env','ten_or_more']].sum()
new_result2 = result.groupby(['yy', 'Form_x'])['cnt_occ'].sum()
new_result = pd.concat([new_result, new_result2], axis=1)

new_result = new_result.reset_index(level='Form_x')
new_result['ten_or_more'] = new_result['ten_or_more'] / new_result['cnt_occ']
new_result['cnt_env'] = new_result['cnt_env'] / new_result['cnt_occ']
new_result['ten_or_more'] = new_result['ten_or_more'] * 100
new_result['cnt_env'] = new_result['cnt_env'] * 100

ten_K = new_result[new_result['Form_x'] == '10-K']
ten_K_ten = list(ten_K['ten_or_more'])
ten_K_one = list(ten_K['cnt_env'])
ten_Q = new_result[new_result['Form_x'] == '10-Q']
ten_Q_ten = list(ten_Q['ten_or_more'])
ten_Q_one = list(ten_Q['cnt_env'])

year = ['2014','2015','2016','2017','2018']
# barWidth = 0.20
barWidth = 0.23
# Set position of bar on X axis
r1 = np.arange(len(year))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# fig = plt.figure(1, (9,5))
# ax = fig.add_subplot(1,1,1)
fig = plt.figure(figsize=(9,5))
ax = plt.axes()
# Make the plot
plt.bar(r1, ten_K_one, color='limegreen', width=barWidth, edgecolor='white',
        label='10-K (one or more)')
plt.bar(r2, ten_K_ten, color='green', width=barWidth, edgecolor='white',
        label='10-K (ten or more)')
plt.bar(r3, ten_Q_one, color='darkblue', width=barWidth, edgecolor='white',
        label='10-Q (one or more)')
plt.bar(r4, ten_Q_ten, color='royalblue', width=barWidth, edgecolor='white',
        label='10-Q (ten or more)')

# avrg = sum(ten_Q_ten)/5

# for i in ax.patches:
#     tt = "%s%%" % round((i.get_height()), 1)
#     ax.text(i.get_x(),i.get_height(), tt, 
#             color='black', ha="left", fontsize=18) #ha="center"  #rotation=40

tu = barWidth * 1.5
plt.xticks([r + tu for r in range(len(year))], ['2014','2015','2016','2017','2018'])
# plt.xticks(r1, ['2014','2015','2016','2017','2018'])
# Create legend & Show graphic
# plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.ylim(0,100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.legend(loc="upper right", fontsize=20)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
# plt.ylabel("%", fontsize=18) 
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

