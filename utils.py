from models.config import *
from models.utils import *
import numpy as np


def search(text, topK=5):
    words = text_cut(text, cut_all=True)
    corpus, idf_dict, tfidf_list = load_cache(CACHE_PATH)
    words = [word for word in words if word in idf_dict]
    score_list = []
    for tfidf in tfidf_list:
        score_list.append(sum([tfidf.get(word, 0) for word in words]))

    ids = [id for id in np.argsort(score_list)[::-1] if score_list[id] != 0]
    return ids[:topK], [corpus[id] for id in ids[:topK]]


if __name__ == '__main__':
    print(search('万科房地产公司怎么样?', 5))
