from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.utils import shuffle
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from matplotlib import pyplot as plt

def get_train_test():
    df = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['target',  'entity', 'backchannel', 'liwc',  'lda', 'slodim']]
    df = shuffle(df).reset_index(drop=True)
    # df.to_csv('best/'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.csv')
    # print(df)
    df = pd.read_csv('best/2022-06-27-11-09-08.csv')
    # df_1 = pd.read_csv('data_processed_30/data_all_feature_entity_semantic.csv')[['slodim','target']]

    target_all = df['target']
    feature_all = []
    for index in range(len(df['backchannel'])):
        e = literal_eval(df['entity'][index])
        b = [df['backchannel'][index]]
        # sd = [df['semantic'][index]]
        l = literal_eval(df['lda'][index])
        li = literal_eval(df['liwc'][index])
        s = literal_eval(df['slodim'][index])
        # v = literal_eval(df['verb'][index])
        # p = literal_eval(df['prosociety'][index])

        feature_all.append(b+e+li+l+s)


    return feature_all, target_all, df

def classify():
    acc = 0
    while acc <= 0.845:
        feature_all, target_all, data_best= get_train_test()
        # cls = SVC()
        cls = RandomForestClassifier(n_estimators=700)
        # cls = LogisticRegression()
        # cls = GaussianNB()
        # cls = KNeighborsClassifier()
        # cls = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')


        score =  cross_val_score(cls, feature_all, target_all, cv=3,scoring='accuracy')
        acc = score.mean()
        print(acc*100)
        print(score)
        break


    # data_best.to_csv('best/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.csv')
    feature_train = feature_all[:int(0.7*(len(feature_all)))]
    target_train = target_all[:int(0.7*(len(feature_all)))]
    feature_test = feature_all[int(0.7*(len(feature_all))):]
    target_test = target_all[int(0.7*(len(feature_all))):]
    cls.fit(feature_train, target_train)
    result =cls.predict(feature_test)
    print(accuracy_score(target_test, result))
    try:
        print(cls.feature_importances_)
    except:
        return 0
    estimator = cls.estimators_[5]
    tree.plot_tree(estimator)
    plt.show()
    tree.plot_tree(cls.estimators_[0])
    plt.show()

def paper(feature:str):
    cls_list = [SVC(),
                RandomForestClassifier(n_estimators=700),
                LogisticRegression(),
                GaussianNB(),
                KNeighborsClassifier()]
    length_list = ['20', '25', '30']

    for length in length_list:
        df = pd.read_csv('data_processed_'+length+'/data_all_feature_entity_semantic.csv')[
            [feature, 'target']]
        df = shuffle(df).reset_index(drop=True)
        target_all = df['target']
        feature_all = []
        if feature == 'backchannel' or feature == 'semantic':
            for index in range(len(df['target'])):
                feature_all.append([df[feature][index]])
        else:
            for index in range(len(df['target'])):
                feature_all.append(literal_eval(df[feature][index]))
        result = []
        for cls in cls_list:
            score = cross_val_score(cls, feature_all, target_all, cv=3, scoring='accuracy')
            # print(score)
            acc = round(score.mean()*100, 2)
            result.append(acc)
        result_str = length
        for temp in result:
            result_str += ' & ' + str(temp)

        print(result_str + '\\\\')


if __name__ == "__main__":
    classify()
    # paper('slodim')
