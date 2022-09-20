import os
import pandas as pd
import entity_weight_feature_extraction
import nltk
from nltk.tag import StanfordPOSTagger


def process_tokens(word_list:list):
    for index in range(len(word_list)):
        if word_list[index] == 'j' or word_list[index] =='J':
            word_list[index] = 'je'
        if word_list[index] == 'c' or word_list[index] == 'J':
            word_list[index] = 'ce'
        if word_list[index] == 't' or word_list[index]=='T':
            word_list[index] = 'tu'
        if word_list[index] == 'd' or word_list[index] == 'T':
            word_list[index] = 'de'
        if word_list[index][0] == '\'':
            word_list[index] = word_list[index][1:]
        word_list[index] = word_list[index].replace('.', '')
        word_list[index] = word_list[index].replace('!', '')
        word_list[index] = word_list[index].replace('?', '')
        word_list[index] = word_list[index].replace(',', '')
        word_list[index] = word_list[index].replace('...', '')
    return word_list

def connect_token(result_parsr:list):
    pure_words_list = []
    pure_sentence = ''
    for token in result_parsr:
        if token[1] != 'PUNCT':
            pure_words_list.append(token[0])
            pure_sentence += token[0]+' '
    return pure_words_list, pure_sentence

def sentence_process(sentence: str):
    tokenizer_french = nltk.data.load('tokenizers/punkt/french.pickle')
    sentence_lower = sentence.lower()
    words = [str(word) for word in tokenizer_french._tokenize_words(sentence_lower)]
    words = process_tokens(words)
    st = StanfordPOSTagger('tool/french-ud.tagger',
                           'tool/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar')
    result_parser = st.tag(words)
    _, pure_sentence = connect_token(result_parser)
    return pure_sentence

def data_process():
    file_name_list = os.listdir('original')
    for file_name in file_name_list:
        target = []
        sentence_sub = []
        source = []
        with open('original/'+file_name, 'r', encoding='utf-8') as file:
            contant = file.readlines()
        for sentence in contant:
            sentence_to_process = sentence.replace('\n', '')
            if sentence_to_process[0] == 'P':
                continue
            if sentence_to_process[0] == 'S':
                target.append(1)
            if sentence_to_process[0] == 'T':
                target.append(0)
            sentence_to_process = sentence_to_process.split('-', 1)[1]
            sentence_sub.append(sentence_to_process)
            source.append(file_name)
        df = pd.DataFrame({'sentence':sentence_sub, 'target':target, 'source':source})
        df.to_csv('only_patient/'+file_name+'.csv')
        print(df)


def get_pure_data():
    file_name_list = os.listdir('only_patient/')
    for file_name in file_name_list:
        if file_name+'_pure.csv' in file_name_list or '_pure.csv' in file_name:
            continue
        print(file_name)
        df = pd.read_csv('only_patient/'+file_name)
        try:
            for index in range(len(df['sentence'])):
                sentence = df['sentence'][index]
                print(sentence)
                sentence_pure = sentence_process(sentence)
                print(sentence_pure)
                df['sentence'][index] = sentence_pure
        except:
            print(file_name)
            return False
        df.to_csv('only_patient/'+file_name+'_pure.csv')

def split_data(length):
    length_list = [128, 256, 512]
    length = int(length)
    if length not in length_list:
        print('wrong length')
        return False

    file_name_list  = [x for x in os.listdir('only_patient') if '_pure.csv' in x ]
    # file_name_list = ['002LOAout.txt.csv_pure.csv']

    for file_name in file_name_list:
        sentence_list = []
        target = []
        source = []
        df = pd.read_csv('only_patient/'+file_name)
        df = df.dropna().reset_index(drop=True)
        index = 0
        num_token = 0
        while index <= len(df['sentence'])-1 and num_token <= length-10:
            sentence_list_temp = []
            num_token = len(df['sentence'][index].split(' '))
            sentence_list_temp.append(df['sentence'][index])
            index_new = index+1
            while index_new <= len(df['sentence'])-1 and num_token <= length-10:
                num_token += len(df['sentence'][index_new].split(' '))
                sentence_list_temp.append(df['sentence'][index_new])
                index_new += 1
            sentence_list.append(sentence_list_temp)
            target.append(df['target'][0])
            source.append(df['source'][0])
            index = index_new
            num_token = 0
        df_to_save = pd.DataFrame({'sentence':sentence_list, 'target':target, 'source':source})
        df_to_save.to_csv('data_processed_'+str(length)+'/'+file_name+'.csv')

def get_all_data():
    length_list = ['128', '256', '512']
    # length_list = ['512']
    for length in length_list:
        file_name_list = [x for x in os.listdir('data_processed_'+length) if 'all' not in x]
        df_list = []
        for file_name in file_name_list:
            df_list.append(pd.read_csv('data_processed_'+length+'/'+file_name))
        df_all = pd.concat(df_list).dropna().reset_index(drop=True)
        df_all.to_csv('data_processed_'+length+'/data_all_'+length+'.csv')




if __name__ == "__main__":
    # split_data(256)
    get_all_data()