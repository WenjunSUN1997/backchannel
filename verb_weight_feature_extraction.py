import sys
from ast import literal_eval
import nltk
from nltk.tag import StanfordPOSTagger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

stop_words = ['aah','oui','mmh','ouais…','euhh','ai','euh','accord','ouais', "a","abord","absolument","afin","ah","ai","aie","ailleurs","ainsi","ait","allaient","allo","allons","allô","alors","anterieur","anterieure","anterieures","apres","après","as","assez","attendu","au","aucun","aucune","aujourd","aujourd'hui","aupres","auquel","aura","auraient","aurait","auront","aussi","autre","autrefois","autrement","autres","autrui","aux","auxquelles","auxquels","avaient","avais","avait","avant","avec","avoir","avons","ayant","b","bah","bas","basee","bat","beau","beaucoup","bien","bigre","boum","bravo","brrr","c","car","ce","ceci","cela","celle","celle-ci","celle-là","celles","celles-ci","celles-là","celui","celui-ci","celui-là","cent","cependant","certain","certaine","certaines","certains","certes","ces","cet","cette","ceux","ceux-ci","ceux-là","chacun","chacune","chaque","cher","chers","chez","chiche","chut","chère","chères","ci","cinq","cinquantaine","cinquante","cinquantième","cinquième","clac","clic","combien","comme","comment","comparable","comparables","compris","concernant","contre","couic","crac","d","da","dans","de","debout","dedans","dehors","deja","delà","depuis","dernier","derniere","derriere","derrière","des","desormais","desquelles","desquels","dessous","dessus","deux","deuxième","deuxièmement","devant","devers","devra","different","differentes","differents","différent","différente","différentes","différents","dire","directe","directement","dit","dite","dits","divers","diverse","diverses","dix","dix-huit","dix-neuf","dix-sept","dixième","doit","doivent","donc","dont","douze","douzième","dring","du","duquel","durant","dès","désormais","e","effet","egale","egalement","egales","eh","elle","elle-même","elles","elles-mêmes","en","encore","enfin","entre","envers","environ","es","est","et","etant","etc","etre","eu","euh","eux","eux-mêmes","exactement","excepté","extenso","exterieur","f","fais","faisaient","faisant","fait","façon","feront","fi","flac","floc","font","g","gens","h","ha","hein","hem","hep","hi","ho","holà","hop","hormis","hors","hou","houp","hue","hui","huit","huitième","hum","hurrah","hé","hélas","i","il","ils","importe","j","je","jusqu","jusque","juste","k","l","la","laisser","laquelle","las","le","lequel","les","lesquelles","lesquels","leur","leurs","longtemps","lors","lorsque","lui","lui-meme","lui-même","là","lès","m","ma","maint","maintenant","mais","malgre","malgré","maximale","me","meme","memes","merci","mes","mien","mienne","miennes","miens","mille","mince","minimale","moi","moi-meme","moi-même","moindres","moins","mon","moyennant","multiple","multiples","même","mêmes","n","na","naturel","naturelle","naturelles","ne","neanmoins","necessaire","necessairement","neuf","neuvième","ni","nombreuses","nombreux","non","nos","notamment","notre","nous","nous-mêmes","nouveau","nul","néanmoins","nôtre","nôtres","o","oh","ohé","ollé","olé","on","ont","onze","onzième","ore","ou","ouf","ouias","oust","ouste","outre","ouvert","ouverte","ouverts","o|","où","p","paf","pan","par","parce","parfois","parle","parlent","parler","parmi","parseme","partant","particulier","particulière","particulièrement","pas","passé","pendant","pense","permet","personne","peu","peut","peuvent","peux","pff","pfft","pfut","pif","pire","plein","plouf","plus","plusieurs","plutôt","possessif","possessifs","possible","possibles","pouah","pour","pourquoi","pourrais","pourrait","pouvait","prealable","precisement","premier","première","premièrement","pres","probable","probante","procedant","proche","près","psitt","pu","puis","puisque","pur","pure","q","qu","quand","quant","quant-à-soi","quanta","quarante","quatorze","quatre","quatre-vingt","quatrième","quatrièmement","que","quel","quelconque","quelle","quelles","quelqu'un","quelque","quelques","quels","qui","quiconque","quinze","quoi","quoique","r","rare","rarement","rares","relative","relativement","remarquable","rend","rendre","restant","reste","restent","restrictif","retour","revoici","revoilà","rien","s","sa","sacrebleu","sait","sans","sapristi","sauf","se","sein","seize","selon","semblable","semblaient","semble","semblent","sent","sept","septième","sera","seraient","serait","seront","ses","seul","seule","seulement","si","sien","sienne","siennes","siens","sinon","six","sixième","soi","soi-même","soit","soixante","son","sont","sous","souvent","specifique","specifiques","speculatif","stop","strictement","subtiles","suffisant","suffisante","suffit","suis","suit","suivant","suivante","suivantes","suivants","suivre","superpose","sur","surtout","t","ta","tac","tant","tardive","te","tel","telle","tellement","telles","tels","tenant","tend","tenir","tente","tes","tic","tien","tienne","tiennes","tiens","toc","toi","toi-même","ton","touchant","toujours","tous","tout","toute","toutefois","toutes","treize","trente","tres","trois","troisième","troisièmement","trop","très","tsoin","tsouin","tu","té","u","un","une","unes","uniformement","unique","uniques","uns","v","va","vais","vas","vers","via","vif","vifs","vingt","vivat","vive","vives","vlan","voici","voilà","vont","vos","votre","vous","vous-mêmes","vu","vé","vôtre","vôtres","w","x","y","z","zut","à","â","ça","ès","étaient","étais","était","étant","été","être","ô"]

def process_tokens(word_list:list):
    for index in range(len(word_list)):
        if word_list[index] == 'j' or word_list[index] =='J':
            word_list[index] = 'je'
        if word_list[index] == 'c' or word_list[index] == 'J':
            word_list[index] = 'ce'
        if word_list[index] == 't' or word_list[index]=='T':
            word_list[index] = 'tu'
        if word_list[index][0] == '\'':
            word_list[index] = word_list[index][1:]
        word_list[index] = word_list[index].replace('.', '')
        word_list[index] = word_list[index].replace('!', '')
        word_list[index] = word_list[index].replace('?', '')
        word_list[index] = word_list[index].replace(',', '')
        word_list[index] = word_list[index].replace('...', '')
    return word_list

def connect_token(result_parsr:list):
    pure_words_list = []
    pure_sentence = ''
    for token in result_parsr:
        if token[1] != 'PUNCT':
            pure_words_list.append(token[0])
            pure_sentence += token[0]+' '
    return pure_words_list, pure_sentence

def sentence_process(sentence_list:list):
    tokenizer_french = nltk.data.load('tokenizers/punkt/french.pickle')
    entity_count = {}
    sentence_pure_list = []
    for sentence in sentence_list:
        sentence_lower = sentence.lower()
        words = [str(word) for  word in tokenizer_french._tokenize_words( sentence_lower)]
        words = process_tokens(words)
        st = StanfordPOSTagger('tool/french-ud.tagger',
                               'tool/stanford-tagger-4.2.0/stanford-postagger-full-2020-11-17/stanford-postagger.jar')
        result_parser = st.tag(words)
        for token in result_parser:
            if token[1] == 'VERB':
                if entity_count.get(token[0]) != None:
                    entity_count[token[0]] += 1
                else:
                    entity_count[token[0]] = 1
        _, pure_sentence = connect_token(result_parser)
        sentence_pure_list.append(pure_sentence)

    return entity_count, sentence_pure_list

def get_tfidf_inblock(entity_name:list, sentence_list:list):
    entity_weight = {}
    tf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    result = tf_vectorizer.fit_transform(sentence_list)
    tfidf_block = result.toarray()
    feature_names = tf_vectorizer.get_feature_names()
    print(entity_name)
    for tfidf in  tfidf_block:
        temp_dict = dict(zip(feature_names, tfidf))
        for name in entity_name:
            try:
                if entity_weight.get(name) != None:
                    entity_weight[name] += temp_dict[name]
                else:
                    entity_weight[name] = temp_dict[name]
            except:
                continue

    return entity_weight


def get_entity_tfidf(length):
    length = str(length)
    if length not in ['128', '256', '512']:
        data_all = pd.read_csv('data_processed_'+length+'/data_all_'+length+'_pure.csv')
    else:
        data_all = pd.read_csv('data_processed_'+length+'/data_all_'+length+'.csv')

    entity_co_final = []
    for data in data_all.iterrows():
        entity_co= []
        entity_tfidf_final = {}
        sentence_list =  literal_eval(data[1]['sentence'])
        entity_count, sentence_pure_list = sentence_process(sentence_list)
        key_list = entity_count.keys()
        for key in list(key_list):
            if key in stop_words:
                entity_count.pop(key)
        print(entity_count)
        entity_tfidf_block = get_tfidf_inblock(list(entity_count.keys()), sentence_pure_list)
        with open('tfidf_dict/'+data[1]['source']+'.pkl', 'rb') as file:
            entity_tfidf_doc = pickle.load(file)
        for entity_name in list(entity_count.keys()):
            try:
                entity_tfidf_final[entity_name] = entity_tfidf_block[entity_name] + entity_count[entity_name]*entity_tfidf_doc[entity_name]
            except:
                continue
        # print(entity_tfidf_final)
        features = list(entity_tfidf_final.values())
        for index in range(len(features)):
            for index_1 in range(index+1, len(features),):
                entity_co.append(abs(features[index]- features[index_1]))
        if len(entity_co) < 5:
            entity_co = entity_co + [0]*(5-len(entity_co))
        entity_co = sorted(entity_co, reverse=True)[:5]
        # print(entity_co)
        # entity_co_final.append(entity_co)
        with open('data_processed_'+length+'/verb_tfidf_temp.txt', 'a+') as file:
            file.write(str(entity_co) + '\n')
    # data_all['feature_entity'] = entity_co_final
    # data_all.to_csv('data_all_10_feature.csv')

if __name__ == "__main__":
    length = sys.argv[1]
    get_entity_tfidf(length)
