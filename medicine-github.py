# -*- coding: utf-8 -*-
"""
Created by Jiachen Wang
PS: this approach is still updating, not the perfect version yet
PPS: the main purpose of this program is to give you some insights of this competition
"""

'''import all the pakages needed
the keras package imported here is based on tensorflow backend
you should have your GPU-version tensorflow package installed
'''
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')

import pandas as pd
import numpy as np
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
import seaborn
import xgboost

INPUT_DIM=300
'''
use pandas to read in the four csv files and do some tricks
'''
train_variant=pd.read_csv("training_variants.csv")
test_variant = pd.read_csv("test_variants.csv")
train_text = pd.read_csv("training_text", sep="\|\|", 
                         engine='python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("test_text", sep="\|\|", 
                        engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how='left', on='ID')

train_y = train['Class'].values 
train_x = train.drop('Class', axis=1)
train_size=train_x.shape[0]
test_x = pd.merge(test_variant, test_text, how='left', on='ID')
test_size=test_x.shape[0]
test_index = test_x['ID'].values
all_data = np.concatenate((train_x, test_x), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]

'''word stemmers for NLP analysis'''
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

'''feature extractions'''
all_data['Gene_Share'] = all_data.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
all_data['Variation_Share'] =all_data.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)
AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'
all_data["simple_variation_pattern"] =all_data.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)
all_data["simple_variation_pattern"]=pd.get_dummies(all_data["simple_variation_pattern"])
all_data['location_number'] = all_data.Variation.str.extract('(\d+)')
all_data['variant_letter_first'] = all_data.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else 'U',axis=1)
all_data['variant_letter_last'] = all_data.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] 
    in (AA_VALID)) else 'U' ,axis=1)
all_data.loc[all_data.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = all_data
all_data['Text_len'] = all_data['Text'].map(lambda x: len(str(x)))
all_data['Text_words'] = all_data['Text'].map(lambda x: len(str(x).split(' '))) 

'''some helper function'''
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

all_data['AAGroup_ofer8_letter_first'] = all_data["variant_letter_first"].map(ofer8).fillna('U')
all_data['AAGroup_ofer8_letter_last'] = all_data["variant_letter_last"].map(ofer8).fillna('U')
all_data['AAGroup_ofer8_equiv'] = all_data['AAGroup_ofer8_letter_first'] == all_data['AAGroup_ofer8_letter_last']
all_data['AAGroup_ofer8_equiv']=pd.get_dummies(all_data['AAGroup_ofer8_equiv'])
all_data['AAGroup_m12_equiv'] =all_data['variant_letter_last'].map(sdm12) == all_data['variant_letter_first'].map(sdm12)
all_data['AAGroup_m12_equiv']=pd.get_dummies(all_data['AAGroup_m12_equiv'])
all_data['AAGroup_p5_equiv'] =all_data['variant_letter_last'].map(pc5) ==all_data['variant_letter_first'].map(pc5)
all_data['AAGroup_p5_equiv']=pd.get_dummies(all_data['AAGroup_p5_equiv'])

'''encode features into one-hot'''
a=pd.get_dummies(all_data['AAGroup_ofer8_letter_first'])
b=pd.get_dummies(all_data['AAGroup_ofer8_letter_last'])
c=pd.get_dummies(all_data['variant_letter_first'])
d=pd.get_dummies(all_data['variant_letter_last'])

col=all_data[['Gene_Share','Variation_Share','simple_variation_pattern','location_number',
    'Text_len','Text_words','AAGroup_ofer8_equiv','AAGroup_m12_equiv','AAGroup_p5_equiv']]
arr=col.values
a=a.values
b=b.values
c=c.values
d=d.values

'''free up some memory'''
del train_variant,test_variant,train_text,test_text,train_x,test_x

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences
 
'''clean up the texts'''
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

'''encode target into one-hot'''
label_encoder = LabelEncoder()
label_encoder.fit(train_y)
encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))

'''the weight deals with imbalance distribution of train target classes'''
aa=[0,0,0,0,0,0,0,0,0,0]
bb=[1,1,1,1,1,1,1,1,1,1]
for i in train_y:
    aa[i]+=1
for i in range(1,10):
    bb[i]*=aa[7]/aa[i]
weight=[bb[i] for i in train_y]
del aa,bb

#l1=[]
#for i in allText:
#    l1.append(i) 
#vectorizer = TfidfVectorizer(stop_words='english')
#f1 = np.squeeze(np.asarray(vectorizer.fit_transform(l1).todense())) 
#del l1

'''train doc-to-vec model'''
model=None
filename='docEmbeddings.d2v'
if os.path.isfile(filename):
    model = Doc2Vec.load(filename) 
else:
    allText = all_data['Text'].apply(cleanup)
    sentences = constructLabeledSentences(allText)
    model = Doc2Vec(min_count=1, window=10, size=INPUT_DIM, sample=1e-4, negative=5, workers=4,iter=10,seed=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save(filename)

'''construct final train arrays'''
del all_data
train_arrays = np.zeros((train_size, INPUT_DIM))
test_arrays = np.zeros((test_size, INPUT_DIM))
for i in range(train_size):
    train_arrays[i] = model.docvecs['Text_'+str(i)]
j=0
for i in range(train_size,train_size+test_size):
    test_arrays[j] = model.docvecs['Text_'+str(i)]
    j=j+1
    
#train_arrays=np.concatenate((train_arrays,f1[:train_size]),axis=1)
#test_arrays=np.concatenate((test_arrays,f1[train_size:]),axis=1)
train_arrays=np.concatenate((train_arrays,arr[:train_size]),axis=1)
test_arrays=np.concatenate((test_arrays,arr[train_size:]),axis=1)
train_arrays=np.concatenate((train_arrays,a[:train_size]),axis=1)
test_arrays=np.concatenate((test_arrays,a[train_size:]),axis=1)
train_arrays=np.concatenate((train_arrays,b[:train_size]),axis=1)
test_arrays=np.concatenate((test_arrays,b[train_size:]),axis=1)
train_arrays=np.concatenate((train_arrays,c[:train_size]),axis=1)
test_arrays=np.concatenate((test_arrays,c[train_size:]),axis=1)
train_arrays=np.concatenate((train_arrays,d[:train_size]),axis=1)
test_arrays=np.concatenate((test_arrays,d[train_size:]),axis=1)
#del count
weight=np.asarray(weight)
del a,b,c,d

'''keras deep leaning model'''
def baseline_model():
    model = Sequential()
    model.add(Dense(1000, input_dim=train_arrays.shape[1],init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1000, input_dim=train_arrays.shape[1],init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(500, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(500, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(200, init='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(9, init='normal', activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
#for i in range (5668):
#    for j in range (9):
#        if y_pred[i][j]==max(y_pred[i]):y_pred[i][j]=1
#        else: y_pred[i][j]=0

'''train the model'''
estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=32)
estimator.fit(train_arrays, encoded_y, validation_split=0.10,sample_weight =weight)

'''predict test data based on keras model'''
y_pred = estimator.predict_proba(test_arrays)
submission = pd.DataFrame(y_pred)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv("submission.csv",index=False)
