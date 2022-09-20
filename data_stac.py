import pandas as pd

def count_class(length):
    length = str(length)
    df = pd.read_csv('data_processed_'+length+'/data_all_feature_entity_semantic.csv')['target']
    num_1 = 0
    num_0 = 0
    for x in df:
        if x==1:
            num_1+=1
        else:
            num_0+=1
    print(num_0, num_1)
if __name__ == "__main__":
    count_class(10)