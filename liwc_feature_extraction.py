import os

from nltk import cluster
import subprocess
from ast import literal_eval
import pandas as pd
# LIWC22-35A0DACE-16374BAC-B0BD842C-C0D1279A
def get_txt_file():
    length_list = ['128', '256', '512']
    for length in length_list:
        if len(os.listdir('data_processed_'+length+'/text'))!=0:
            continue
        df = pd.read_csv('data_processed_'+length+'/data_all_'+length+'.csv')
        for index in range(len(df['sentence'])):
            sentence_list = literal_eval(df['sentence'][index])
            sentence = ' '.join(x for x in sentence_list)
            with open('data_processed_'+length+'/text/'+str(index)+'.txt', 'w', encoding='utf-8') as file:
                file.write(sentence)

def analysis():

    pass

def trans_csv_to_txt():
    length_list = ['20']

    for length in length_list:
        try:
            with open('data_processed_' + length + '/liwc_temp.txt', 'w') as file:
                file.write('')
            df = pd.read_csv('data_processed_'+length+'/liwc.csv')
            col_name_list = ['mental','work','certitude','Affect', 'Social', 'Lifestyle','illness', 'emo_pos', 'emo_neg']
            print(col_name_list)
            for index in range(len(df['Filename'])):
                print(index)
                index_target = df[df['Filename']==str(index)+'.txt'].index.tolist()[0]
                print(index_target)
                result =[]
                for col_name in col_name_list:
                    result.append(df[col_name][index_target])
                print(result)
                with open('data_processed_'+length+'/liwc_temp.txt', 'a+') as file:
                    file.write(str(result) + '\n')
        except:
            continue

if __name__ == "__main__":
    # get_txt_file()
    trans_csv_to_txt()