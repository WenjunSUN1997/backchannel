from ast import literal_eval
a = [1,2,3]
print(a.index(max(a)))
print(a[:1])
for index in range(0, 20, 5):
    print(index, index+20)

a = "[0.33440472 0.66559528]\n"
a= '[0, 0, 0, 0, 0]\n'
print(literal_eval(a.replace('\n', '')))
with open('G:\Onedrive\internship\data_extend\data_processed_5\entity_tfidf_temp.txt', 'r') as file:
    print(file.readlines())