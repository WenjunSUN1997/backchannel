import sys
from ast import literal_eval
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData, ranges, minVals


def get_feature(length):
    length = str(length)
    length_list = [ '256']
    if length not in length_list:
        sentence_list_all = pd.read_csv('data_processed_' + length + '/data_all_' + length + '_pure.csv')['sentence']
    else:
        sentence_list_all = pd.read_csv('data_processed_' + length + '/data_all_' + length + '.csv')['sentence']
    document = []
    # sentence_list_all = pd.read_csv('data_processed_'+length+'/data_all_'+length+'_pure.csv')['sentence']
    for sentence_list in sentence_list_all:
        pure_sentence_list = literal_eval(sentence_list)
        document.append(' '.join(x for x in pure_sentence_list))
    print(document)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(document)

    print(X)
    lda = LatentDirichletAllocation(n_components=2,
        random_state=0)
    lda.fit(X)
    lda_result = np.array(lda.transform(X))
    lda_result, _, _ = noramlization(lda_result)
    print(lda_result)
    lda_result = lda_result.tolist()
    print(lda_result)
    lda_result_new = []
    for i in lda_result:
        x = [0]*len(i)
        index = i.index(max(i))
        x[index] = 1
        lda_result_new.append(x)

    for data in lda_result_new:
        with open('data_processed_'+length+'/lda_temp.txt', 'a+') as file:
            file.write(str(data) + '\n')

if __name__ == "__main__":
    length = '256'
    with open('data_processed_' + length + '/lda_temp.txt', 'w') as file:
        file.write('')
    get_feature(length)