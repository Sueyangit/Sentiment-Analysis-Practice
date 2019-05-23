# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:49:51 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:05:30 2016
@author: ldy
"""
from gensim.models.word2vec import Word2Vec
import jieba
from sklearn.externals import joblib
from data_process import buildWordVector
#import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')


    
##得到待预测单个句子的词向量    
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('./svm_data/w2v_model/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = buildWordVector(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs
    
####对单个句子进行情感判断    
def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('./svm_data/svm_model/model.pkl')
     
    result=clf.predict(words_vecs)
    
    if int(result[0])==1:
        print(string,' positive')
    else:
        print(string,' negative')
    
if __name__=='__main__':
    
    
    ##导入文件，处理保存为向量
#    x_train,x_test=loadfile() #得到句子分词后的结果，并把类别标签保存为y_train。npy,y_test.npy
#    get_train_vecs(x_train,x_test) #计算词向量并保存为train_vecs.npy,test_vecs.npy
#    train_vecs,y_train,test_vecs,y_test=get_data()#导入训练数据和测试数据
#    svm_train(train_vecs,y_train,test_vecs,y_test)#训练svm并保存模型
    

##对输入句子情感进行判断
    string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    #string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'    
    svm_predict(string)