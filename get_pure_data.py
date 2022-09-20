import pandas as pd
import entity_weight_feature_extraction
from ast import literal_eval
import sys

def run():
    length = str(30)
    print(length)
    df = pd.read_csv('data_processed_'+length+'/data_all_'+length+'.csv')
    for index in range(len(df['sentence'])):
        sentence_list = literal_eval(df['sentence'][index])
        entity_count, sentence_pure_list = entity_weight_feature_extraction.sentence_process(sentence_list)
        print(sentence_pure_list)
        df['sentence'][index] = sentence_pure_list

    df.to_csv('data_processed_'+length+'/data_all_'+ length+'_pure.csv')