import sys

import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

backchannel = ['mmh', 'ouais', 'accord', 'oui', 'voilà', 'hum', 'muh',
               'bon', 'bien', 'ah', 'ben', 'alors', 'euh', 'super', 'euh', 'ok']

def get_backchannel(length):
    length = str(length)
    if length in ['128', '256', '512']:
        df = pd.read_csv('data_processed_' + length + '/data_all_' + length + '.csv')
    else:
        df = pd.read_csv('data_processed_' + length + '/data_all_' + length + '_pure.csv')
    tf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)

    for index in range(len(df['sentence'])):
        weight = 0
        all = 0
        backchannel_count = dict(zip(backchannel, [0] * len(backchannel)))
        sentence_list = literal_eval(df['sentence'][index])
        for sentence in sentence_list:
            for sentence_token in sentence.split(' '):
                if sentence_token in backchannel:
                    backchannel_count[sentence_token] += 1
        print(backchannel_count)
        result = tf_vectorizer.fit_transform(sentence_list)
        tfidf_block = result.toarray()
        tokens = tf_vectorizer.get_feature_names()
        for tfidf in tfidf_block:
            tfidf_sentence = dict(zip(tokens, tfidf))
            for backchannel_token in backchannel:
                if tfidf_sentence.get(backchannel_token) != None:
                    weight += tfidf_sentence[backchannel_token]

        with open('tfidf_dict/'+df['source'][index]+'.pkl', 'rb') as file:
            entity_tfidf_doc = pickle.load(file)
        for backchannel_token in backchannel:
            try:
                all += backchannel_count[backchannel_token] * entity_tfidf_doc[backchannel_token]
            except:
                continue
        if all == 0:
            result = 0
        else:
            result = weight / all

        with open('data_processed_'+length+'/backchannel_temp.txt', 'a+') as file:
            file.write(str(result) + '\n')




def run():
    length = '30'
    get_backchannel(length)
