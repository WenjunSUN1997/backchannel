import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt

df = pd.read_csv('data_processed_25/data_all_25_pure.csv')

normal = []
patient = []

for index in range(len(df['sentence'])):
    if df['target'][index]==1:
        with open('data_processed_25/text_en/'+str(index)+'.txt', 'r', encoding='utf-8') as file:
            patient.append(file.read())
    else:
        with open('data_processed_25/text_en/'+str(index)+'.txt', 'r', encoding='utf-8') as file:
            normal.append(file.read())

print(normal)
print(patient)
text_normal = ' '.join(x for x in normal)
text_patient = ' '.join(x for x in patient)
stopwords = set(STOPWORDS)
stopwords.update(['see','going','things','think','make','bah','go','dacord','Ah','OK','suddenly','time','thing','fact','say','finally','MMH','little','good',"yeah", "uh", "well", "ye", "yes", 'know', 'will'])
wordcloud_normal = WordCloud(stopwords=stopwords).generate(text_normal)
wordcloud_patinet = WordCloud(stopwords=stopwords).generate(text_patient)
# Display the generated image:
plt.imshow(wordcloud_normal, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.imshow(wordcloud_patinet, interpolation='bilinear')
plt.axis("off")
plt.show()