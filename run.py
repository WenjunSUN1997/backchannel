import data_preprocess
import get_pure_data
import get_tfidf_dict_all_doc
import backchannel_feature_extraction
import entity_weight_feature_extraction
import feature_merge

'''创建data_all.csv'''
data_preprocess.run()

'''得到pure data'''
get_pure_data.run()

'''获得所有的tf-idf值'''
get_tfidf_dict_all_doc.run()

'''获得backchannel importance'''
backchannel_feature_extraction.run()

'''获得实体稳定度'''
entity_weight_feature_extraction.run()

'''最终'''
feature_merge.run()





