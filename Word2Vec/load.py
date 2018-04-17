# -*- coding: utf-8 -*-
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
sentences = word2vec.Text8Corpus("corpus.txt")
model = word2vec.Word2Vec(sentences, size=50, min_count=5)
model.wv.save_word2vec_format("embedding.bin", binary=True)
model = KeyedVectors.load_word2vec_format("embedding.bin", binary=True)
model.save_word2vec_format("embedding.txt", binary=False)
