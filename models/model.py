import math
from tqdm import tqdm
from models.utils import text_cut

class TfidModel():
    def __init__(self):
        self.all_words=[]
        self.vocab=set()
        self.corpus_words=[]
    # 加载不同的分词
    def load_corpus(self,corpus):
        for sentence in tqdm(corpus,desc='load corpus'):
            words =text_cut(sentence)
            self.corpus_words.append(words)
            self.all_words+=words
            self.vocab.update(words)

    # 计算各个单词的tf
    def compute_tf(self):
        tf_list=[]
        for words in tqdm(self.corpus_words,desc='compute tf'):
            tf={}
            for word in words:
                # 计算每个word的tf
                tf[word]=words.count(word)/len(words)
            tf_list.append(tf)
        self.tf_list=tf_list

    # 计算各个单词的idf
    def compute_idf(self):
        idf_dict={}
        N =len(self.corpus_words)
        for word in tqdm(self.vocab,desc='compute idf'):
            num=sum([1 if word in words else 0 for words in self.corpus_words])
            #     计算每个word的idf
            idf_dict[word]=math.log(N/(num+1))

        self.idf_dict=idf_dict

    # 计算各个单词的tfidf
    def compute_tfidf(self):
        self.compute_tf()
        self.compute_idf()
        tfidf_list=[]
        for tf in tqdm(self.tf_list,desc='compute tfidf'):
            tfidf={}
            for word,tf_val in tf.items():
                tfidf[word]=tf_val * self.idf_dict[word]
            tfidf_list.append(tfidf)
            return tfidf_list
