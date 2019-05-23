# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:31:44 2018

@author: Administrator
"""

from sklearn.svm import SVC
from sklearn.externals import joblib
from data_process import loadfile , get_train_vecs ,get_data ,\
tokenizer,word2vec_train

import yaml
import sys
import multiprocessing
import numpy as np

#from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
#from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
#import pandas as pd
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()


def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC(kernel='rbf',verbose=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/svm_model/model.pkl')
    print(clf.score(test_vecs,y_test))

    
    
##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test),show_accuracy=True)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_data/lstm.h5')
    print('Test score:', score)

#训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
def train():
    print('Loading Data...')
    combined,y=loadfile()
    print(len(combined),len(y))
    print('Tokenising...')
    combined = tokenizer(combined)#对句子经行分词，并去掉换行符
    print('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,\
    x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print(x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test) # 训练模型，并保存

    
if __name__=='__main__':
  #以下是使用SVM模型训练
    x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
    get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
    train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
    svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型
#    print->[LibSVM]0.6823027718550106
    
  #以下是使用LSTM模型训练
    #
    train()

