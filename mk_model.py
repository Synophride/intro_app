import numpy
import json
import data_importer as di
import math
import multiclass_perceptron as mp
import modele as m

def count_words(train_data):
    dico = dict()
    for i in train_data:
        sentence = i[0]
        for word in sentence:
            dico[word] = dico.get(word, 0) + 1
    return dico

# pê utiliser un set
def get_n_most_used_words(wc, N=10):
    n = 0
    liste = []
    for i in sorted(wc.items(), key = lambda x : x[1], reverse = True):
        liste.append(i[0])
        n+=1
    return liste

def build_freqs_dicts(data, most_used_words):
    d_g = dict()
    d_d = dict()
    for x in data:
        for sentence in x[0]:
            for pos in range(len(sentence)):
                if (sentence[pos] in most_used_words):
                    if(pos > 0):
                        key = sentence[pos-1] + '|' + sentence[pos] 
                        d_d[key] = d_d.get(key, 0) + 1
                    if(pos < len(sentence)-1):
                        key = sentence[pos] + '|' + sentence[pos+1] 
                        d_g[key] = d_g.get(key, 0) + 1
    return (d_g, d_d)

def build_distributional(pos, sentence, d):
    (d_g, d_d) = d
    word = sentence[pos]
    return_keyG = '0'
    return_keyD = '0'
    
    if pos > 0 :
        prev_word = sentence[pos - 1]
        key = sentence[pos-1] + '|' + word
        return_keyG = str(d_g.get(key, 1))
    if(pos < len(sentence) - 1):
        next_word = sentence[pos+1]
        key = word + '|' + sentence[pos+1]
        return_keyD = str(d_d.get(key, 1))

    return (return_keyG, return_keyD)
    


        
    
def build_sparse(sentence, pos, dcs):
    word = sentence[pos]
    ret = dict()

    ret['word' + word] = 1
    for i in range(1,3):
        ret['word_m' + str(i) + (sentence[pos - i] if (pos > i) else '')] = 1
        ret['word_p' + str(i) + (sentence[pos + i] if pos+i < len(sentence) else '')] = 1
    
    ## features de l'article

    #### 1. Word features
    ret['count_left'+ str(pos) ] = 1
    ret['count_right' + str(len(sentence) - pos)] = 1
    # binary suffix features
    # binary shape  features

    #### 2. Distributional features
    kg, kd = build_distributional(pos, sentence, dcs)

    # maydo : Adaptation des features suivantes au français
    #### 3. Suffix features
    ret['suffix_feat' + str(1 if word[-1] == 's' else 0)] = 1
    
    #### 4. Shape features
    ret['shape' +
        str(1 if any(char.isdigit() for char in word) else 0) +
        str(1 if '-' in word else 0) +
        str(1 if word.isupper() else 0) +
        str(1 if word[-2:] == 'ed' else 0) +
        str(1 if word[-3:] == 'ing' else 0)
    ] = 1

    return ret
    
#########
"""
Accomplit des tests, sur des données de test, et en prenant en paramètre un perceptron
params : 
  - test_set, les données de test
  - perceptron, le classifieur
"""
def test(test_set, perceptron, dcs):
    good = 0 # nombre de bonnes prédictions 
    total= 0 # nombre de prédictions totales
    for x in test_set: 
        sentence = x[0] # tableau de mots
        labels   = x[1] # tableau de labels
        for i in range(len(sentence)):
            representation= build_sparse(sentence, i, dcs)
            y = labels[i]
            ypred = perc.predict(representation)
            if(y == ypred):
                good +=1
            total += 1
    return (good, total)


# x = un tableau [[phrase]; [labels]]
def train(perceptron, data, dcs):
    for x in train_set :
        sentence= x[0] 
        labels  = x[1]
        for i in range(len(sentence)):
            representation = build_sparse(sentence, i, dcs)
            perc.train(representation, labels[i])


def init_dicts(train_set):
    wc  = count_words(train_set)
    muw = get_n_most_used_words(wc)
    dcs = build_freqs_dicts(train_set, muw)
    return dcs

class Modele_projet(m.Modele):
    def __init__(self):
        self.dcs = None
        self.lbls= None
        self.p = None
        pass
    
    def init_train(self, train_data):
        self.dcs = init_dicts(train_data)
        self.lbls= di.mk_lbl_set(train_data)
        self.p = mp.Perceptron(lbls)

    def train(self, train_data):
        train(train_data, self.p, self.dcs)

    def reset(self):
        self.dcs = None
        self.lbls= None
        self.p = None
        pass

    def predict(self, sentence, pos):
        features = build_sparse(sentence, pos, self.dcs)
        ypred = self.p.predict(features)
        return ypred

    # returns a tuple (good_predictions, nb_examples)
    def test(self, test_data):
        return test(test_data, self.p, dcs)
