# https://radimrehurek.com/gensim/models/word2vec.html

from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read("./default.ini")
data_path = cfg['DATA']['data_path']

sents = []
N = 0
with open(data_path, 'r') as f:
    lines = f.readlines()
    sents = [l.strip().split('|') for l in lines[0:100000]]
# release memory
del lines

print('---- Start Training ----')
model = Word2Vec(sents[0:100000], size=256, window=5, min_count=50, workers=8)
print(model.wv.most_similar('小学', topn=10))
print(model.wv.most_similar('生活', topn=10))
print(model.wv.most_similar('记者', topn=10))
print(model.wv.most_similar('少儿', topn=10))
print(model.wv.most_similar('深圳', topn=10))
model.save("./models/word2vec_gensim.model")

# In gensim, the training is streamed, 
# meaning sentences can be a generator, reading input data from disk on-the-fly, without loading the entire corpus into RAM.
# model = Word2Vec.load("./models/word2vec_gensim.model")
# model.train(sents[50000:], total_examples=1, epochs=1)

