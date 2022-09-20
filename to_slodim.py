import pandas as pd
from ast import literal_eval

length = 20

df = pd.read_csv('data_processed_'+str(length)+'/data_all_'+str(length)+'.csv')['sentence']
for index in range(len(df)):
    sentence_list  = literal_eval(df[index])
    with open('slodim/raw/'+str(length)+'/'+str(index)+'.txt', 'w', encoding='utf-8') as file:
        a = 0
        for sentence in sentence_list:
            a += 1
            if sentence=='':
                continue
            if  a %2==0:
                file.write('s:'+sentence+'\n')
            else:
                file.write('p:' + sentence + '\n')
