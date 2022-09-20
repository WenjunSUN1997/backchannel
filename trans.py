import os
import time
from pygoogletranslation import Translator
def translate(lenghth):
    lenghth =str(lenghth)
    translator = Translator()
    file_list = os.listdir('data_processed_'+lenghth+'/text')
    try:
        for file_name in file_list:
            file_list_en = os.listdir('data_processed_'+lenghth+'/text_en')
            if file_name in file_list_en:
                continue
            time.sleep(0.3)
            with open('data_processed_'+lenghth+'/text/'+file_name, encoding='utf-8') as file:
                a = file.read()
                print(a)
                print(type(a))
                result = translator.translate(a, src='fr', dest='en')
                print(result.text)
            with open('data_processed_'+lenghth+'/text_en/'+file_name, 'w', encoding='utf-8') as file:
                file.write(result.text)
        return 1
    except:
        return 0
if __name__ == "__main__":
    length_list = ['20']
    for length in length_list:
        while True:
            if translate(length)==1:
                break