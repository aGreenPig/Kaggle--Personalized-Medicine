# -*- coding: utf-8 -*-
"""
@author: Jiachen Wang
this is my current and best model on this project: my current rank is top 20 percent
"""

from sklearn import pipeline,feature_extraction,decomposition,model_selection,metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os
from matplotlib import pyplot
os.chdir("D:\py2\medicine")

train = pd.read_csv('training_variants.csv')
test = pd.read_csv('test_variants.csv')
trainx = pd.read_csv('training_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
testx = pd.read_csv('test_text', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_size=train.shape[0]
test_size=test.shape[0]

train = pd.merge(train, trainx, how='left', on='ID').fillna('')
y = train['Class'].values
train_y = train['Class'].values
train = train.drop(['Class'], axis=1)


test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values

df_all = pd.concat((train, test), axis=0, ignore_index=True)
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)

del trainx,testx
for i in range(10):
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: str(x[i]) if len(x)>i else '')
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: str(x[i]) if len(x)>i else '')


gen_var_lst = sorted(list(train.Gene.unique()))
print(len(gen_var_lst))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
print(len(gen_var_lst))
i_ = 0
for gen_var_lst_itm in gen_var_lst:
    if i_ % 100 == 0: print(i_)
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
    i_ += 1

for c in df_all.columns:
    if df_all[c].dtype == 'object':
        if c in ['Gene','Variation']:
            lbl = LabelEncoder()
            df_all[c+'_lbl_enc'] = lbl.fit_transform(df_all[c].values)  
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' ')))
        elif c != 'Text':
            lbl = LabelEncoder()
            df_all[c] = lbl.fit_transform(df_all[c].values)
        if c=='Text': 
            df_all[c+'_len'] = df_all[c].map(lambda x: len(str(x)))
            df_all[c+'_words'] = df_all[c].map(lambda x: len(str(x).split(' '))) 

AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'
df_all["simple_variation_pattern"] =df_all.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)
df_all["simple_variation_pattern"]=pd.get_dummies(df_all["simple_variation_pattern"])
df_all['location_number'] = df_all.Variation.str.extract('(\d+)')
df_all['variant_letter_first'] = df_all.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else 'U',axis=1)
df_all['variant_letter_last'] = df_all.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else 'U' ,axis=1)
df_all.loc[df_all.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = df_all
df_all['Text_len'] = df_all['Text'].map(lambda x: len(str(x)))
df_all['Text_words'] = df_all['Text'].map(lambda x: len(str(x).split(' '))) 

def TransDict_from_list(groups):
   result = {}
   for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            result[c] = str(g_members[0]) #K:V map, use group's first letter as represent.
   return result
ofer8=TransDict_from_list(["C", "G", "P", "FYW", "AVILM", "RKH", "DE", "STNQ"])
sdm12 =TransDict_from_list(
    ["A", "D", "KER", "N",  "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"] )
pc5 = {"I": "A", # Aliphatic
         "V": "A",         "L": "A",
         "F": "R", # Aromatic
         "Y": "R",         "W": "R",         "H": "R",
         "K": "C", # Charged
         "R": "C",         "D": "C",         "E": "C",
         "G": "T", # Tiny
         "A": "T",         "C": "T",         "S": "T",
         "T": "D", # Diverse
         "M": "D",         "Q": "D",         "N": "D",
         "P": "D"}

df_all['AAGroup_ofer8_letter_first'] = df_all["variant_letter_first"].map(ofer8).fillna('U')
df_all['AAGroup_ofer8_letter_last'] = df_all["variant_letter_last"].map(ofer8).fillna('U')

df_all['AAGroup_ofer8_equiv'] = df_all['AAGroup_ofer8_letter_first'] == df_all['AAGroup_ofer8_letter_last']
df_all['AAGroup_ofer8_equiv']=pd.get_dummies(df_all['AAGroup_ofer8_equiv'])
df_all['AAGroup_m12_equiv'] =df_all['variant_letter_last'].map(sdm12) == df_all['variant_letter_first'].map(sdm12)
df_all['AAGroup_m12_equiv']=pd.get_dummies(df_all['AAGroup_m12_equiv'])
df_all['AAGroup_p5_equiv'] =df_all['variant_letter_last'].map(pc5) ==df_all['variant_letter_first'].map(pc5)
df_all['AAGroup_p5_equiv']=pd.get_dummies(df_all['AAGroup_p5_equiv'])

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Input, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization, Embedding
from keras.utils import np_utils
from keras.preprocessing import text, sequence
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences
def cleanup(line):
    line = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", line)
    line = line.lower()
    line = line.replace('.', '').replace(',', '').replace(':', '').replace('(','').replace(')','').replace(';','').replace('=','').replace('/','')
    line = line.split(" ")
    line=[x for x in line if not x in stopwords.words('english')]
    line=[x for x in line if not x.isdigit()]
    line=[x for x in line if not (x=='' or x==' ')]
    line=' '.join(line)
    wordnet_lemmatizer.lemmatize(line,pos='n')
    wordnet_lemmatizer.lemmatize(line,pos='v')
    wordnet_lemmatizer.lemmatize(line,pos='a')
    wordnet_lemmatizer.lemmatize(line,pos='s')
    wordnet_lemmatizer.lemmatize(line,pos='r')
    return line

label_encoder = LabelEncoder()
label_encoder.fit(train_y)
encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))

aa=[0,0,0,0,0,0,0,0,0,0]
bb=[1,1,1,1,1,1,1,1,1,1]
for i in train_y:
    aa[i]+=1
for i in range(1,10):
    bb[i]*=aa[7]/aa[i]
weight=[bb[i] for i in train_y]
del aa,bb
weight=np.asarray(weight)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
l1=[]
for i in df_all['Text']:
    l1.append(i) 
vectorizer = TfidfVectorizer(stop_words='english',max_features=20000)
f1 = np.squeeze(np.asarray(vectorizer.fit_transform(l1).todense())) 
del l1
pca = PCA(n_components=150)
reduced=pca.fit_transform(f1)
del f1
for i in range(150):
    df_all['tfidf'+str(i)] = [row[i] for row in reduced]

model=None
filename='docEmbeddings.d2v'
if os.path.isfile(filename):
    model = Doc2Vec.load(filename) 
else:
    allText = df_all['Text'].apply(cleanup)
    sentences = constructLabeledSentences(allText)
    model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=4,iter=10,seed=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save(filename)

train_arrays = np.zeros((train_size, 300))
test_arrays = np.zeros((test_size, 300))

for i in range(train_size):
    train_arrays[i] = model.docvecs['Text_'+str(i)]
j=0
for i in range(train_size,train_size+test_size):
    test_arrays[j] = model.docvecs['Text_'+str(i)]
    j=j+1
    
pred=np.concatenate((train_arrays,test_arrays),axis=0)
for i in range(9):
    df_all['kerasmodel_'+str(i)] = [row[i] for row in pred]

del df_all['variant_letter_first'],df_all['variant_letter_last'],df_all['AAGroup_ofer8_letter_first'],df_all['AAGroup_ofer8_letter_last']
del df_all['location_number'],df_all['Gene'], df_all['Variation'], df_all['Text']
train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]
del df_all,pred,train_arrays,test_arrays,encoded_y
print('df_all is ready')

y = y - 1 #fix for zero bound array
denom = 0
preds=0
fold = 10
for i in range(fold):
    
    params = {
        'eta': 0.05,
        'max_depth': 5,
        'objective': 'multi:softprob',
        'eval_metric': 'merror',
        'num_class': 9,
        'seed': i,
        'silent': True,
        'subsample':0.8,
        'colsample_bytree':0.8,
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.15, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1,missing=0), 'train'), (xgb.DMatrix(x2, y2,missing=0), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1,missing=0), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=120)
    #score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2,missing=0), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    #print("score:: ",score1)
    print('processing..')
    
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test,missing=0), ntree_limit=model.best_ntree_limit)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test,missing=0), ntree_limit=model.best_ntree_limit)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_fold_'  + str(i) + '.csv', index=False)
    #print(model.feature_importances_)

preds/=denom
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)
