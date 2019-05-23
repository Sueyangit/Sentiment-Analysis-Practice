# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:22:09 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
import jieba
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import configurations as config
#import multiprocessing

from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
#cpu_count = multiprocessing.cpu_count()
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100

#以下是使用SVM模型训练准备的util
# 加载文件，导入数据,分词
#def loadfile():
#    neg=pd.read_excel(config.NEG_PATH,header=None,index=None)
#    pos=pd.read_excel(config.POS_PATH,header=None,index=None)
#
#    cw = lambda x: list(jieba.cut(x))
#    pos['words'] = pos[0].apply(cw)
#    neg['words'] = neg[0].apply(cw)
#
#    #print pos['words']
#    #use 1 for positive sentiment, 0 for negative
#    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
#
#    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)
#    
#    np.save('./svm_data/y_train.npy',y_train)
#    np.save('./svm_data/y_test.npy',y_test)
#    return x_train,x_test
#
#def get_data():
#    train_vecs=np.load('./svm_data/train_vecs.npy')
#    y_train=np.load('./svm_data/y_train.npy')
#    test_vecs=np.load('./svm_data/test_vecs.npy')
#    y_test=np.load('./svm_data/y_test.npy') 
#    return train_vecs,y_train,test_vecs,y_test
#


#对每个句子的所有词向量取均值
def buildWordVector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
    
#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #Initialize model and build vocab
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)
    
    #Train the model over train_reviews (this may take several minutes)
#    imdb_w2v.train(x_train)
    imdb_w2v.train(x_train,total_examples=imdb_w2v.corpus_count,epochs=2)
    
    train_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)
    
    np.save('./svm_data/train_vecs.npy',train_vecs)
    print(train_vecs.shape)
    #Train word2vec on test tweets
    imdb_w2v.train(x_test,total_examples=imdb_w2v.corpus_count,epochs=2)
    imdb_w2v.save('./svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([buildWordVector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('./svm_data/test_vecs.npy',test_vecs)
    print(test_vecs.shape)


    
    
#以下是给LSTM模型训练准备的util
#加载训练文件
def loadfile():
    neg=pd.read_excel(config.NEG_PATH,header=None,index=None)
    pos=pd.read_excel(config.POS_PATH,header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

    return combined,y

#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

    
    
#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
#                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined)
    model.save('lstm_data/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined
    
    
    

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test



#if __name__=='__main__':
    
    
    #导入文件，处理保存为向量
#    x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
#    get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
#    train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
#    svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型
    