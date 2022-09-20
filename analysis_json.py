import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression,SGDClassifier

all = []
length = '30'
file_list = [x for x in os.listdir('slodim/result/'+length+'/') if 'conllu' not in x and 'ufo' not in x and 'json.json' not in x]
print(len(file_list))
for index in range(len(file_list)):
    try:
        with open('slodim/result/'+length+'/'+str(index)+'.txt.json', 'r') as file:
            data = json.load(file)
    except:
        print(index)
    data  = data["analysis"][0]
    # print(data.keys())
    # result_slodim = [data['verbalArity'], data['subordinatesRatio']]

    result_slodim = [
                     data['stopwordRatio'],
                     # data['lexicalDensity'],
                     data['verbalArity'],
                     data['clausesLength'],
                     # data['subordinatesRatio'],
                     # data['questions'],
                     # data['questionsRatio'],
                     # data['typesTokensRatio'],
                     # data['verbalRootsRatio'],
                     # data['flow'],
                     data['longestDependencyString']
                     ]
    all.append(result_slodim)
print(all)

tags = pd.read_csv('data_processed_'+length+'/data_all_feature_entity_semantic.csv')['target']
print(list(tags))

feature_train = all[:int(0.7*len(all))]
feature_test = all[int(0.7*len(all)):]
target_train = tags[:int(0.7*len(all))]
target_test = tags[int(0.7*len(all)):]
cls = RandomForestClassifier(n_estimators=700)
# cls = SGDClassifier(loss='log')
# cls = SVC()
# # cls = GaussianNB()
# cls = DecisionTreeClassifier()
cls.fit(feature_train, target_train)
result =cls.predict(feature_test)
print(accuracy_score(target_test, result))
print(cls.feature_importances_)

with open('slodim_temp.txt', 'w', encoding='utf-8') as file:
   file.write('')
#
with open('slodim_temp.txt', 'w', encoding='utf-8') as file:
    for x in all:
        file.write(str(x)+'\n')


