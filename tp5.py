import numpy
import json
import multiclass_perceptron
path='./multiclass_corpora/'

full_path = path + 'fr.ud.train.json'
train_set = json.load(open(full_path))
soccer_set=json.load(open(path + 'foot.json'))
mc_set = json.load(open(path + 'minecraft.json'))

np_arr = numpy.array(train_set)
print(np_arr.shape)

# Donne la liste des labels
def mk_lbl_set(dataset):
    ret_dict = set()
    for (w, l) in dataset:
        for i in l:
            ret_dict.add(i)
        break
    return ret_dict

lbl_set = mk_lbl_set(train_set)

print('Labels :')
for i in lbl_set:
    print(i)
    
"""
Construit une représentation éparse a priori meilleure
[00:11]Rémi:
du coup, moi ce que j'ai fait c'est que mon dictionnaire est comme ça : 
clé = critère + valeur du critère 
valeur = 1
"""
def build_sparse2(sentence, pos):
    word = sentence[pos]
    ret['l3c'+word[-3:]] = 1
    ret['lc'+word[-1]] = 1
    ret['fc'+word[0]]  = 1
    ret['w'+word]      = 1
    ret['first_upper'+str(1 if word[0].isupper() else 0)] = 1
    ret['all_upper'+str(1 if word.isupper() else 0)] = 1 
    ret['w_-1'+ (sentence[pos-1] if pos > 0 else '', 0)] = 1
    ret['w_-2'+ (sentence[pos-2] if pos > 1 else '', 0)] = 1
    ret['w_+1'+ (sentence[pos+1] if pos < len(sentence)-1 else '', 0) ] = 1 
    ret['w_+2'+ (sentence[pos+2] if pos < len(sentence)-2 else '', 0)] = 1
    ret['f'] = 0
    return ret


# construit les compteurs :
#   chaque compteur est un dictionnaire associant
# une donnée (par exemple un mot) à un entier
def build_counters(data):
    word_counter = dict()
    word_counter[''] = 0 # Dictionnaire des mots
    last3char_counter = dict() # dictionnaire des suites de 3 caractères
    lastchar_counter = dict() # dictionnaire de caractères (pas forcément utile puisque on peut récupérer le code ascii)
    wc = 1
    l3c = 1
    lc = 1
    for sentence in data:
        for word in sentence:            
            if (not word[3:] in last3char_counter):
               last3char_counter[word[3:]] = l3c
               l3c += 1
            if word not in word_counter:
                word_counter[word] = wc
                wc += 1
            if word[-1] not in lastchar_counter:
                lastchar_counter[word[-1]] = lc
                lc += 1
    return (word_counter, last3char_counter, lastchar_counter)

# construit une matrice éparse
# la construction est probablement mauvaise
def build_sparse(counters, sentence, pos):
    word = sentence[pos]
    (word_c, l3c_c, lc_c, caps_c) = counters
    ret = dict()
    ret['l3c'] = l3c_c.get(word[-3:], 0)
    ret['lc'] = lc_c.get(word[-1] , 0)
    ret['fc'] = lc_c.get(word[0], 0)
    ret['w']  = word_c.get(word, 0)
    ret['first_upper'] = 1 if word[0].isupper() else 0
    ret['all_upper'] = 1 if word.isupper() else 0
    ret['w_-1'] = word_c.get(sentence[pos-1] if pos > 0 else '', 0)
    ret['w_-2'] = word_c.get(sentence[pos-2] if pos > 1 else '', 0)
    ret['w_+1'] = word_c.get(sentence[pos+1] if pos < len(sentence)-1 else '', 0)
    ret['w_+2'] = word_c.get(sentence[pos+2] if pos < len(sentence)-2 else '', 0)
    ret['f'] = 0
    return ret


perc = Perceptron(lbl_set)

