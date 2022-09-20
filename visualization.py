import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd
from ast import literal_eval
from sklearn.neighbors import KernelDensity
import random
from scipy.stats import wasserstein_distance

import matplotlib.patches as mpatches
def show():
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'lda']]
    print(df)
    type = []
    value = []
    for index in range(len(df['target'])):
        if df['target'][index]==1:
            type.append('patient')
        else:
            type.append('normal')
        x = literal_eval(df['lda'][index])
        if x[0]==1:
            value.append('topic_1')
        else:
            value.append('topic_2')
    df_new = pd.DataFrame({'type':type, 'value':value})
    print(df_new)
    sn.displot(x="value", hue="type", data=df_new,stat='probability')
    plt.show()
    fig, ax = plt.subplots()
    normal = df_new['value'][df_new['type']=='normal']
    patient = df_new['value'][df_new['type']=='patient']
    sn.histplot(patient, legend=True,kde=False, ax=ax, color='tab:orange', label="patient", stat='probability', cbar=True)
    sn.histplot(normal, legend=True,kde=False, ax=ax, color='tab:blue', label="noraml", stat='probability', cbar=True)
    plt.xlabel('LDA')
    plt.legend(loc="upper left")
    plt.savefig('G:\Onedrive\\fig\\' + 'LDA_distribution.png')

    plt.show()

    return
    # (e+b+li+l)
    li = [0.06212257, 0.06071308, 0.0628997,  0.06227726, 0.07250346, 0.10618836,
     0.01927543, 0.04824632, 0.04444002, 0.05372847, 0.05806867, 0.05819111,
     0.02950473, 0.05381164, 0.03862665, 0.08638647, 0.08301607]
    print(li[1:3])

    importance = [sum([0.06212257, 0.06071308, 0.0628997,  0.06227726, 0.07250346]), li[5],
                  sum([0.01927543, 0.04824632, 0.04444002, 0.05372847, 0.05806867, 0.05819111,
     0.02950473, 0.05381164, 0.03862665]), sum([0.08638647, 0.08301607])]
    print(importance)
    name = ['E0', 'E1','E2','E3','E4','back', 'mental','work','certitude','affect', 'social', 'lifestyle','illness', 'emo_pos', 'emo_neg', 'topic_1', 'topic_2']
    df = pd.DataFrame({'feature':name, 'importance':li})
    sn.barplot(y="feature", x="importance", data=df)
    plt.show()

    length_list = ['20', '25', '30', '512']
    for length in length_list:
        df = pd.read_csv('data_processed_'+length+'/data_all_feature_entity_semantic.csv')
        count = len(df['sentence'])
        word = []
        for x in df['sentence']:
            c= 0
            a = literal_eval(x)
            for b in a:
                # print(b)
                c+=len(b.split(' '))
            word.append(c)
        print(count, ' & ' ,min(word),' & ',max(word),' & ',int(sum(word)/count) )


    normal = []
    patient = []
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'entity']]
    for index in range(len(df['target'])):
        all = literal_eval(df['entity'][index])[0]
        if df['target'][index] == 0:
            normal.append(all)
        else:
            patient.append(all)
    data_all = normal+patient
    target_all = ['normal']*len(normal) + ['patient']*len(patient)
    df = pd.DataFrame({'entity_stability':data_all, 'target':target_all})
    sn.kdeplot(data=df, x='entity_stability', hue='target')
    plt.show()



def show_entity_test():
    fig, ax = plt.subplots()
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'entity']]
    normal = df['entity'][df['target']==0].reset_index(drop=True)
    patient = df['entity'][df['target'] == 1].reset_index(drop=True)
    # sn.kdeplot(normal,ax = ax,color='tab:blue',label="noraml")
    # sn.kdeplot(patient, ax = ax,color='tab:orange', label="patient")
    for index in range(len(normal)):
        normal[index] = float(sum(literal_eval(normal[index])))
    for index in range(len(patient)):
        patient[index] = float(sum(literal_eval(patient[index])))
    normal = normal.tolist()
    patient = patient.tolist()
    sn.histplot(normal, binwidth= 100, kde=True,ax=ax, color='tab:blue', stat='probability', label='normal')
    sn.histplot(patient, binwidth= 100,kde=True, ax=ax, color='tab:orange', stat='probability', label='patient')
    plt.xlabel('entity')
    plt.legend(loc="upper right")
    # plt.text(2000, 1, '— patient', color='#ffbe86', size=15)
    plt.savefig('G:\Onedrive\\fig\\'+'entity_distributation.png')
    plt.show()

def show_backchannel_test():
    fig, ax = plt.subplots()
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'backchannel']]
    normal = df['backchannel'][df['target'] == 0].reset_index(drop=True)
    patient = df['backchannel'][df['target'] == 1].reset_index(drop=True)
    # sn.kdeplot(normal,ax = ax,color='tab:blue',label="noraml")
    # sn.kdeplot(patient, ax = ax,color='tab:orange', label="patient")
    for index in range(len(normal)):
        normal[index] = float(normal[index])
    for index in range(len(patient)):
        patient[index] = float(patient[index])
    normal = normal.tolist()
    patient = patient.tolist()
    sn.histplot(normal, binwidth=0.005,kde=True, ax=ax, color='tab:blue', stat='probability', label='normal')
    sn.histplot(patient, binwidth=0.005,kde=True, ax=ax, color='tab:orange', stat='probability', label='patient')
    plt.xlabel('backchannel importance')
    plt.legend(loc="upper right")
    # plt.text(2000, 1, '— patient', color='#ffbe86', size=15)
    plt.savefig('G:\Onedrive\\fig\\' + 'backchannel_importance.png')
    plt.show()

def show_semantic_test():
    fig, ax = plt.subplots()
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'semantic']]
    normal = df['semantic'][df['target'] == 0].reset_index(drop=True)
    patient = df['semantic'][df['target'] == 1].reset_index(drop=True)
    # sn.kdeplot(normal,ax = ax,color='tab:blue',label="noraml")
    # sn.kdeplot(patient, ax = ax,color='tab:orange', label="patient")
    for index in range(len(normal)):
        normal[index] = float(normal[index])
    for index in range(len(patient)):
        patient[index] = float(patient[index])
    normal = normal.tolist()
    patient = patient.tolist()
    sn.histplot(normal, binwidth=4,kde=True, ax=ax, color='tab:blue', stat='probability', label='normal')
    sn.histplot(patient, binwidth=4,kde=True, ax=ax, color='tab:orange', stat='probability', label='patient')
    plt.xlabel('semantic density')
    plt.legend(loc="upper right")
    # plt.text(2000, 1, '— patient', color='#ffbe86', size=15)
    plt.savefig('G:\Onedrive\\fig\\' + 'semantic_density.png')
    plt.show()

def show_backchannel():
    normal = []
    patient = []
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'semantic']]
    for index in range(len(df['target'])):
        all = float(df['backchannel'][index])
        if df['target'][index] == 0:
            normal.append(all)
        else:
            patient.append(all)
    data_all = normal + patient
    target_all = ['normal'] * len(normal) + ['patient'] * len(patient)
    df = pd.DataFrame({'backchannel_importance': data_all, 'target': target_all})
    sn.kdeplot(data=df, x='backchannel_importance', hue='target')
    plt.show()

def show_semantic():
    normal = []
    patient = []
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'semantic']]
    for index in range(len(df['target'])):
        all = float(df['semantic'][index])
        if df['target'][index] == 0:
            normal.append(all)
        else:
            patient.append(all)
    data_all = normal+patient
    target_all = ['normal']*len(normal) + ['patient']*len(patient)
    df = pd.DataFrame({'semantic_density':data_all, 'target':target_all})
    sn.kdeplot(data=df, x='semantic_density', hue='target')
    plt.show()

def show_liwc():
    f = plt.figure()
    dict = {0:'mental',1:'work',2:'certitude',3:'Affect', 4:'Social', 5:'Lifestyle',6:'illness', 7:'emo_pos', 8:'emo_neg'}
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'liwc']]

    for index_to_show in list(dict.keys()):
        print(index_to_show)
        normal = []
        patient = []
        for index in range(len(df['target'])):

            all = literal_eval(df['liwc'][index])[index_to_show]
            if df['target'][index] == 0:
                normal.append(all)
            else:
                patient.append(all)
        fig, ax = plt.subplots()
        sn.histplot(normal,  binwidth=0.25, kde=True, ax=ax, color='tab:blue', stat='probability', label='normal')
        sn.histplot(patient, binwidth=0.25,kde=True, ax=ax, color='tab:orange', stat='probability', label='patient')
        plt.xlabel(dict[index_to_show])
        plt.legend(loc="upper right")
        plt.savefig('G:\Onedrive\\fig\\'+dict[index_to_show]+'.png')
        plt.show()

def show_slodim_test():
    f = plt.figure()
    dict = {0: 'stopwordRatio',
            1: 'lexicalDensity',
            2: 'verbalArity',
            3: 'clausesLength',
            4:'subordinatesRatio',
            5:'questions',
            6:'questionsRatio',
            7:'typesTokensRatio',
            8:'verbalRootsRatio',
            9:'flow',
            10:'longestDependencyString'}
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target', 'slodim']]
    width = [0.025,
             0.025,
             0.25,
             2.5,
             5,
             0.5,
             0.025,
             0.025,
             0.5,
             0.5,
             2.5
             ]
    for index_to_show in list(dict.keys()):
        print(index_to_show)

        normal = []
        patient = []
        for index in range(len(df['target'])):

            all = literal_eval(df['slodim'][index])[index_to_show]
            if df['target'][index] == 0:
                normal.append(all)
            else:
                patient.append(all)
        max_value = max([max(patient), max(normal)])
        fig, ax = plt.subplots()
        sn.histplot(normal, binrange=[0, max_value],binwidth=width[index_to_show], kde=True, ax=ax, color='tab:blue', stat='probability', label='normal')
        sn.histplot(patient, binrange=[0, max_value],binwidth=width[index_to_show], kde=True, ax=ax, color='tab:orange', stat='probability', label='patient')
        plt.xlabel(dict[index_to_show])
        plt.legend(loc="upper right")
        plt.savefig('G:\Onedrive\\fig\slo\\' + dict[index_to_show] + '.png')
        plt.show()

def show_importance_test():

    importance = [0.11269549, 0.05225694, 0.05077011, 0.05393537, 0.05568709, 0.05816056,
 0.01306515, 0.03111563, 0.02821049, 0.03870278, 0.0359821,  0.03887813,
 0.02348167, 0.0375692,  0.03297777, 0.09697767, 0.0998786,  0.02665055,
 0.0421873,  0.028863,   0.04195437]
    name = ['back.','E0', 'E1','E2','E3','E4','mental','work','certitude','affect', 'social', 'lifestyle','illness', 'emo_pos', 'emo_neg',
            'topic_1','topic_2','stopword.','verbal.','clauses.','longestD.'
            ]
    df = pd.DataFrame({'feature': name, 'importance': importance})
    sn.set(font_scale=1)
    sn.barplot(y="feature", x="importance", data=df, palette=sn.color_palette("muted",2))
    plt.show()




if __name__ == "__main__":
    show_importance_test()
    # show_liwc()
    # # show()
    # show_entity_test()
    # show_backchannel_test()
    # show_semantic_test()


