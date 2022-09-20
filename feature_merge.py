import pandas as pd
from ast import literal_eval

def run():
    length_list = ['5', '10', '15', '20', '25', '30']
    # length_list = ['128', '512']

    feature_list = ['verb_tfidf_temp.txt','backchannel_temp.txt', 'semantic_length.txt', 'entity_tfidf_temp.txt', 'lda_temp.txt', 'liwc_temp.txt', 'prosociety_temp.txt', 'slodim_temp.txt']
    for length in length_list:
        for feature in feature_list:
            feature_all = []
            try:
                with open('data_processed_'+length+'/'+feature, 'r') as file:
                    print(length)
                    feature_contant = file.readlines()
                    if feature == 'lda_temp.txt':
                        print(feature_contant)
                for index in range(len(feature_contant)):
                    if feature_contant[index][0]=='[':
                        feature_contant[index] = literal_eval(feature_contant[index].replace('\n', ''))
                    else:
                        feature_contant[index] = float(feature_contant[index].replace('\n', ''))
                    feature_all.append(feature_contant[index])
                print('done')

                df = pd.read_csv('data_processed_'+length+'/data_all_feature_entity_semantic.csv')
                print(feature.split('_')[0])
                df[feature.split('_')[0]] = feature_all
                df = df.reset_index(drop=True)
                df.to_csv('data_processed_'+length+'/data_all_feature_entity_semantic.csv')
                print(length)
            except:
                continue