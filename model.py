#File: BestModel.py
#Author: Jiachen Wang

print("Importing packages in need")
from  sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import os
import re
import matplotlib.pyplot as plt
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Input, LSTM, dot, Flatten, Dense, Reshape, add, Dropout, BatchNormalization, Embedding
from keras.utils import np_utils
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

rootdir="D:\py2\medicine"
os.chdir(rootdir)

print("start loading docs and creating dataframes")
oldtest = pd.read_csv('test_variants.csv')
oldtestx = pd.read_csv('test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
oldtest = pd.merge(oldtest, oldtestx, how='left', on='ID').fillna('') 
oldtestsolution = pd.read_csv('stage1_solution_filtered.csv')
oldtest = pd.merge(oldtest, oldtestsolution, how='left', on='ID').fillna('')
del oldtestsolution, oldtestx
train = pd.read_csv('training_variants.csv')
trainx = pd.read_csv('training_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train, trainx, how='left', on='ID').fillna('')
train=pd.concat([train,oldtest.loc[oldtest["Class"] != '']])
test = pd.read_csv('stage2_test_variants.csv')
testx = pd.read_csv('stage2_test_text.csv', sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
test = pd.merge(test, testx, how='left', on='ID').fillna('')
pid = test['ID'].values
train_y = train['Class'].values
train = train.drop(['Class'], axis=1)
train_size=train.shape[0]
test_size=test.shape[0]
df_all = pd.concat((train, test), axis=0, ignore_index=True)
del trainx,testx,test
print(str(train_size)+" rows and training data and "+str(test_size)+" rows of testing data in total.")
print("loading docs and creating dataframes completed")

print("start extracting biomedical features")
df_all['Gene_Share'] = df_all.apply(lambda r: sum([1 for w in r['Gene'].split(' ') if w in r['Text'].split(' ')]), axis=1)
df_all['Variation_Share'] = df_all.apply(lambda r: sum([1 for w in r['Variation'].split(' ') if w in r['Text'].split(' ')]), axis=1)
for i in range(5):
    df_all['Gene_'+str(i)] = df_all['Gene'].map(lambda x: ord(x[i]) if len(x)>i else 0)
    df_all['Variation'+str(i)] = df_all['Variation'].map(lambda x: ord(x[i]) if len(x)>i else 0)
gen_var_lst = sorted(list(train.Gene.unique()))
gen_var_lst = [x for x in gen_var_lst if len(x.split(' '))==1]
for gen_var_lst_itm in gen_var_lst:
    df_all['GV_'+str(gen_var_lst_itm)] = df_all['Text'].map(lambda x: str(x).count(str(gen_var_lst_itm)))
AA_VALID='ACDEFGHIKLMNPQRSTVWY'
df_all["simple_variation_pattern"] =df_all.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)
df_all["simple_variation_pattern"]=pd.get_dummies(df_all["simple_variation_pattern"])
df_all['location_number'] = df_all.Variation.str.extract('(\d+)').fillna(0)
df_all['variant_letter_first'] = df_all.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else 'U',axis=1)
df_all['variant_letter_last'] = df_all.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else 'U' ,axis=1)
df_all.loc[df_all.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = df_all
df_all['Text_len'] = df_all['Text'].map(lambda x: len(str(x)))
df_all['Text_words'] = df_all['Text'].map(lambda x: len(str(x).split(' ')))
def TransDict_from_list(groups):
   result = {}
   for group in groups:
        g_members = sorted(group)
        for c in g_members:
            result[c] = str(g_members[0])
   return result
ofer8=TransDict_from_list(["C", "G", "P", "FYW", "AVILM", "RKH", "DE", "STNQ"])
sdm12 =TransDict_from_list(["A", "D", "KER", "N",  "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"])
pc5 = {"I": "A","V": "A","L": "A","F": "R","Y": "R","W": "R","H": "R",
  "K": "C","R": "C","D": "C","E": "C","G": "T","A": "T","C": "T","S": "T",
  "T": "D","M": "D","Q": "D", "N": "D","P": "D"}
df_all['AAGroup_ofer8_letter_first'] = df_all["variant_letter_first"].map(ofer8).fillna('U')
df_all['AAGroup_ofer8_letter_last'] = df_all["variant_letter_last"].map(ofer8).fillna('U')
df_all['AAGroup_ofer8_equiv'] = df_all['AAGroup_ofer8_letter_first'] == df_all['AAGroup_ofer8_letter_last']
df_all['AAGroup_ofer8_equiv']=pd.get_dummies(df_all['AAGroup_ofer8_equiv'])
df_all['AAGroup_m12_equiv'] =df_all['variant_letter_last'].map(sdm12) == df_all['variant_letter_first'].map(sdm12)
df_all['AAGroup_m12_equiv']=pd.get_dummies(df_all['AAGroup_m12_equiv'])
df_all['AAGroup_p5_equiv'] =df_all['variant_letter_last'].map(pc5) ==df_all['variant_letter_first'].map(pc5)
df_all['AAGroup_p5_equiv']=pd.get_dummies(df_all['AAGroup_p5_equiv'])
print("biomedical feature extraction completed")

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
    line=[x for x in line if not (x=='' or x==' ')]
    line=[porter_stemmer.stem(w) for w in line]
    line=' '.join(line)
    return line
label_encoder = LabelEncoder()
label_encoder.fit(train_y)
encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))
print("cleaning up text words")
df_all['Text'] = df_all['Text'].apply(cleanup)
print("creating natural language tf-idf feature")
vectorizer = TfidfVectorizer(stop_words='english',max_features=200)
f1 = np.squeeze(np.asarray(vectorizer.fit_transform(list(df_all['Text'])).todense()))
for i in range(100):
    df_all['tfidf'+str(i)] = [row[i] for row in f1]
del f1
print("creating document-to-vector word vector feature")
model=None
filename='docEmbeddings.d2v'
if os.path.isfile(filename):
    model = Doc2Vec.load(filename) 
else:
    sentences = constructLabeledSentences(df_all['Text'])
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
dtov=np.concatenate((train_arrays,test_arrays),axis=0)
for i in range(300):
    df_all['dtov_'+str(i)] = [row[i] for row in dtov]

'''
num_words = 2000
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(df_all['Text'].values)
X = tokenizer.texts_to_sequences(df_all['Text'].values)
X = pad_sequences(X, maxlen=2000)
'''

'''
embed_dim = 128
lstm_out = 196
ckpt_callback = ModelCheckpoint('keras_model', 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')
model = Sequential()
model.add(Embedding(num_words, embed_dim, input_length = train_arrays.shape[1]))
model.add(LSTM(lstm_out, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(9,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['categorical_crossentropy'])
print(model.summary())
Y = pd.get_dummies(train_y).values
X_train, X_test, Y_train, Y_test = train_test_split(train_arrays, Y, test_size = 0.2, random_state = 42, stratify=Y)
batch_size = 32
model.fit(X_train, Y_train, epochs=8, batch_size=batch_size, validation_split=0.2, callbacks=[ckpt_callback])
model = load_model('keras_model')
pred1 = model.predict_proba(train_arrays)
pred2 = model.predict_proba(test_arrays)
pred=np.concatenate((pred1,pred2),axis=0)
del pred1,pred2
for i in range(9):
    df_all['kerasmodel_'+str(i)] = [row[i] for row in pred]
'''

del df_all['variant_letter_first'],df_all['variant_letter_last'],df_all['AAGroup_ofer8_letter_first'],df_all['AAGroup_ofer8_letter_last']
del df_all['location_number'],df_all['Gene'], df_all['Variation'], df_all['Text']
train = df_all.iloc[:len(train)]
test = df_all.iloc[len(train):]
del df_all,train_arrays,test_arrays,encoded_y

print('all feature extraction completed')
print("XGBoost random tree classifier starts")
train_y =  train_y - 1 #fix for zero bound array
denom = 0
preds=0
for i in range(10):
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
    x1, x2, y1, y2 = model_selection.train_test_split(train, train_y, test_size=0.15, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1,missing=0), 'train'), (xgb.DMatrix(x2, y2,missing=0), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1,missing=0), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test,missing=0), ntree_limit=model.best_ntree_limit+80)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test,missing=0), ntree_limit=model.best_ntree_limit+80)
        preds = pred.copy()
    denom += 1
preds/=denom
print("generating final result")
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb.csv', index=False)

'''
References to some other participants:

https://www.kaggle.com/the1owl/redefining-treatment-0-57456
https://www.kaggle.com/reiinakano/basic-nlp-bag-of-words-tf-idf-word2vec-lstm
https://www.kaggle.com/danofer/genetic-variants-to-protein-features
'''
