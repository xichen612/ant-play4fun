# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:47:02 2018

@author: chenxi11
"""

import jieba
import os
import pandas as pd
#import numpy as np
from gensim.models import word2vec
# import uniout
jieba.add_word(u'借呗')
jieba.add_word(u'花呗')

path = '/data/chenxi11/others/Ant'
dat = pd.read_csv(path + '/data/atec_nlp_sim_train.csv', sep='\t', header=None)
print dat.shape
print dat.head(10)

# 分词
def process(inpath, outpath):
    with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
        wordList = []
        for line in fin:
            lineno, sen1, sen2, label = line.strip().split('\t')

            sen1 = sen1.strip().replace(' ', '')
            sen2 = sen2.strip().replace(' ', '')

            words1 = [w1 for w1 in jieba.cut(sen1) if w1!= u'\ufeff']
            words2 = [w2 for w2 in jieba.cut(sen2) if w2!= u'\ufeff']

            wordList += words1
            wordList += words2

            words1_copus = ' '.join(words1).encode('UTF-8')
            words2_copus = ' '.join(words2).encode('UTF-8')

            fout.write(words1_copus + '\n')
            fout.write(words2_copus + '\n')

    return wordList

inpath = path + '/data/atec_nlp_sim_train.csv'
outpath = path + '/result/r2.txt'

wordList = process(inpath, outpath)
wordList_uni = list(set(wordList))


# word2vec训练
myCorpus = word2vec.Text8Corpus(outpath)
mymodel = word2vec.Word2Vec(myCorpus, min_count=1)

mymodel.save(path +'/result/r2.model')

# 形成vector字典
# featureDict = {key:[] for key in wordList_uni}

def generate_embedding(inpath, wordList, model):
    featureDict = {key: [] for key in wordList}
    with open(inpath, 'r') as fin:
        for line in fin:
            print line
            lineno, sen1, sen2, label = line.strip().split('\t')
            sen1 = sen1.strip().replace(' ', '')
            sen2 = sen2.strip().replace(' ', '')
            print sen1
            print sen2

            words1 = list(set([w1 for w1 in jieba.cut(sen1) if w1!= u'\ufeff']))
            words2 = list(set([w2 for w2 in jieba.cut(sen2) if w2!= u'\ufeff']))

            print 'part1'
            word1List = wordList[:]

            top_w1 = []
            for w1 in words1:
                print("**********")
                print w1
                print("**********")

                top10 = mymodel.wv.most_similar(w1, topn=10)
                for i1 in top10:
                    print 'here'
                    print i1[0]
                    featureDict[i1[0]].append(i1[1])
                    top_w1.append(i1[0])

            top_w1 = set(top_w1)
            for j1 in set(word1List).difference(top_w1):
                featureDict[j1].append(0)
            print "-----"
            print 'part2'
            print "-----"
            word2List = wordList[:]
            top_w2 = []
            for w2 in words2:
                print("**********")
                print w2
                print("**********")
                top10_w2 = mymodel.wv.most_similar(w2, topn=10)
                for i2 in top10_w2:
                    print "there"
                    print i2[0]
                    featureDict[i2[0]].append(i2[1])
                    top_w2.append(i2[0])

            top_w2 = set(top_w2)
            for j in set(word2List).difference(top_w2):
                featureDict[j].append(0)

    return featureDict


featureDict = generate_embedding(inpath, wordList_uni, mymodel)

if u'开花' in wordList_uni:
    print "hello"
# if __name__ == '__main__':
#    process(sys.argv[1], sys.argv[2]`)