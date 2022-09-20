import sys

import pandas as pd
from ast import literal_eval
import torch
from sentence_transformers import SentenceTransformer, util
import math


if __name__ == "__main__":
    length_to_process = str(sys.argv[1])
    result_all = []
    if length_to_process not in ['128', '256', '512']:
        data_all = pd.read_csv('data_processed_'+length_to_process+'/data_all_'+length_to_process+'_pure.csv')
    else:
        data_all = pd.read_csv('data_processed_'+length_to_process+'/data_all_'+length_to_process+'.csv')

    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    for data in data_all.iterrows():
        length_semantic_sentence = 0
        length_semantic_block = 0
        result_temp = []
        sentence_pure_list = literal_eval(data[1]['sentence'])
        length = sum([len(x.split(' ')) for x in sentence_pure_list])
        sentence_embeddings = model.encode(sentence_pure_list)
        for index in range(len(sentence_embeddings)-1):
            length_semantic_sentence += math.sqrt(sum([(a - b)**2 for (a,b) in zip(sentence_embeddings[index], sentence_embeddings[index+1])]))
        length_semantic_block = math.sqrt(sum([(a - b)**2 for (a,b) in zip(sentence_embeddings[0], sentence_embeddings[-1])]))
        result = (length_semantic_block / length_semantic_sentence) * length
        print(result)
        print(length)
        with open('data_processed_'+length_to_process+'/semantic_length.txt', 'a+') as file:
            file.write(str(result) + '\n')




