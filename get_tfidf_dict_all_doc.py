import os
import data_preprocess
import entity_weight_feature_extraction
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def get_file():
    file_list = os.listdir('original')
    file_contant_list = []
    for file_name in file_list:
        sentence_list, _ = data_preprocess.get_sentence_list(file_name)
        print('raw sentece')
        print(sentence_list)
        _,  pure_sentence= entity_weight_feature_extraction.sentence_process(sentence_list)
        pure_file = ' '.join(x for x in pure_sentence)
        print('pure sentece')
        print(pure_file)
        file_contant_list.append(pure_file)
    df = pd.DataFrame({'contant':file_contant_list, 'resource':file_list})
    df.to_csv('tfidf_dict/document_pure.csv')
    return file_contant_list

def get_tfidf(file):
    file_list = os.listdir('original')
    tf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    result = tf_vectorizer.fit_transform(file)
    tfidf = result.toarray()
    feature_names = tf_vectorizer.get_feature_names()
    for index in range(len(tfidf)):
        dic = dict(zip(feature_names, tfidf[index]))
        print(dic)
        with open('tfidf_dict/'+file_list[index]+'.pkl', 'wb') as file:
            pickle.dump(dic, file)



def run():
    if not (os.path.exists('tfidf_dict')):
        os.makedirs('tfidf_dict')

    get_tfidf(get_file())




