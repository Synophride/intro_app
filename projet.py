import json
import os
import sys
import numpy as np
import math
from itertools import product

fr_path = 'fr/'

fr_files = os.listdir(fr_path)

fr_json = dict()

#Calcule la taille de chaque data 
def corpus_size():
    print("Nombre de phrase des differents corpus:")
    for filename in fr_files :
        path = fr_path + filename
        j = json.load(open(path))
        arr = np.array(j)
        fr_json[filename] = arr
        if(arr.shape != ()):
            print(filename, ':', arr.shape[0])
        else:
            print(filename, ': 0')

#Calcule le pourcentage de mot donnés dans les datas test qui ne sont pas contenus dans leur train
def calcul_oov(train_file, test_file):
    vocab_train = set()
    for i in train_file:
        for j in i[0]:
            vocab_train.add(j)
    
    nb_out_vocab = 0
    for i in test_file:
        for j in i[0]:
            if j not in vocab_train:
                nb_out_vocab += 1
    return (nb_out_vocab / (len(vocab_train) + nb_out_vocab)) * 100

#Affichage de calcul_oov
def calcul_oov_print():
    data = ['fr.gsd.','fr.partut.','fr.sequoia.','fr.spoken.','fr.ftb.','fr.pud.'] 
    print("Pourcentage d'OOV des differents corpus:")  
    for i in data:
        train = fr_json[i+'train.json']
        test = fr_json[i+'test.json']
        name = i.split('.')
        print(name[1],':',calcul_oov(train,test),'%')

#Calcul de la KL divergence des 3-grammes characters des données train et test 
def kl_divergence_build(corpus):
    grams = []
    alphabet = []
    cpt = 0
    for i in corpus:
        sentence = ""
        for j in i[0]:
            sentence+=j+" "
        sentence = sentence[:-1]
        char = list(sentence)
        for k in range(len(char)):
            cpt += 1
            if char[k] not in alphabet:
                alphabet.append(char[k])
            if(k>=2):
                gram = char[k-2]+ char[k-1] + char[k]
                grams.append(gram)
    dictionary = {}
    for i in grams:
        if i not in dictionary:
            dictionary[i]=1
        else:
            dictionary[i]+=1
    return dictionary,grams,alphabet,cpt

def proba(trig,d,a,n):
    if trig not in d:
        return 1/(a**3+n-2)
        #return 1/(a+len(d)*(n-2))
    else:
        return (d[trig]+1)/(a**3+n-2)
        #return (d[trig]+1)/(a+len(d)*(n-2))
        

def calcul_kl_divergence(corpus):
    train = fr_json[corpus+'.train.json']
    test = fr_json[corpus+'.test.json']
    d_train, g_train, alpha_train, n_train = kl_divergence_build(train)
    d_test, g_test, alpha_test, n_test = kl_divergence_build(test)
    alpha = alpha_test + alpha_train
    grams = [product(alpha,repeat=3)]
    cpt = 0
    for gram in grams:
        train= proba(gram,d_train,len(alpha),n_train)
        test= proba(gram,d_test,len(alpha),n_test)
        #train= proba(gram,d_train,len(g_train),n_train)
        #test = proba(gram,d_test,len(g_test),n_test)
        cpt += test*math.log(test/train)
    print('KL_divergence de '+corpus+': '+ str(cpt))

    #N[tri_grams]/alphabet**n+lenth-n+1
    # n = 3

#Demandez la formule au prof
        



corpus_size()
print()
calcul_oov_print()
print()
calcul_kl_divergence('fr.sequoia')
