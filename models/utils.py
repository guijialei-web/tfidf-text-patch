import jieba
import logging
import pickle
from .config import *
jieba.setLogLevel(logging.INFO)


def load_stopwords():
    stopwords=open(STOPWORDS_PATH,encoding='utf-8-sig',).read().split('\n')
    stopwords.extend(['\n'])
    return stopwords

def text_cut(text,cut_all=False):
    words=[]
    stopwords=load_stopwords()
    for word in jieba.cut(text,cut_all=cut_all):
        if word not in stopwords:
            words.append(word)
    return words

def dump_cache(object,path):
    pickle.dump(object,open(path,'wb'))

def load_cache(path):
    return pickle.load(open(path,'rb'))