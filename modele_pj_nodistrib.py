import modele as m

import numpy
import json
import helpers as pj
import math
import multiclass_perceptron as mp
    


def build_sparse(sentence, pos):
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


    # maydo : Adaptation des features suivantes au français
    #### 3. Suffix features
    ret['suffix_feat' + str(1 if word[-1] == 's' else 0)] = 1
    
    #### 4. Shape features
    ret['isdig' + str(1 if any(char.isdigit() for char in word) else 0)] = 1
    ret['hyphen' + str(1 if '-' in word else 0)]  = 1
    ret['uppr' + str(1 if word.isupper() else 0)] = 1
    ret['é' + str(1 if word[-1] == 'é' else 0)] = 1
    ret['er' + str( 1 if word[-2:] == 'er' else 0)] = 1
    ret['ant' + str(1 if word[-3:] == 'ant' else 0)] = 1
    return ret
    
#########

"""
Accomplit des tests, sur des données de test, et en prenant en paramètre un perceptron
params : 
  - test_set, les données de test
  - perceptron, le classifieur
"""
def test(test_set, perceptron):
    good = 0 # nombre de bonnes prédictions 
    total= 0 # nombre de prédictions totales
    for x in test_set: 
        sentence = x[0] # tableau de mots
        labels   = x[1] # tableau de labels
        for i in range(len(sentence)):
            representation= build_sparse(sentence, i)
            y = labels[i]
            ypred = perceptron.predict(representation)
            if(y == ypred):
                good +=1
            total += 1
    return (good, total)


# x = un tableau [[phrase]; [labels]]
def train( data, perc):
    for x in data :
        sentence= x[0] 
        labels  = x[1]
        for i in range(len(sentence)):
            representation = build_sparse(sentence, i)
            perc.train(representation, labels[i])


class Modele_nodistrib(m.Modele):
    def __init__(self):
        self.lbls = None
        self.p = None
        pass
    
    def init_train(self, train_data):
        self.lbls= pj.mk_lbl_set(train_data)
        self.p = mp.Perceptron(self.lbls)

    def train(self, train_data, epoch = 10):
        for i in range(epoch):
            train(train_data, self.p)

    def reset(self):
        self.lbls= None
        self.p = None
        pass

    def predict(self, sentence, pos):
        features = build_sparse(sentence, pos)
        ypred = self.p.predict(features)
        return ypred

    # returns a tuple (good_predictions, nb_examples)
    def test(self, test_data):
        return test(test_data, self.p)

    def get_str(self):
        return "Perceptron nodistrib"
