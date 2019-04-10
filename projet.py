import json
import os
import sys
import numpy as np
import json
import os

import math
from itertools import product

import modele

fr_path = './fr/'
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




################################################
################################################
############
############   CALCUL PERPLEXITE
############
################################################
################################################


#Calcule le pourcentage de mot donnés dans les datas test qui ne sont pas contenus dans leur train
def calcul_oov(train_file, test_file):
    vocab_train = set()
    for i in train_file:
        for j in i[0]:
            vocab_train.add(j)
    vocab_test = set()
    for i in test_file:
        for j in i[0]:
            vocab_test.add(j)

    nb_tr_vocab = len(vocab_train)
    nb_out_vocab = 0
    for i in vocab_test:
        if(i not in vocab_train): 
            nb_out_vocab +=1
    
    return (nb_out_vocab*100) / (nb_out_vocab + nb_tr_vocab)

#Affichage de calcul_oov
def oov_display():
    data = ['fr.gsd.','fr.partut.','fr.sequoia.','fr.spoken.','fr.ftb.','fr.pud.'] 
    print("Pourcentage d'OOV des differents corpus:")  
    for i in data:
        train = fr_json[i+'train.json']
        test = fr_json[i+'test.json']
        name = i.split('.')
        print(name[1],':',calcul_oov(train,test),'%')

#Créer les dictionnaires 3-grams, l'alphabet et le nombre de caratères du corpus
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

#Renvoie la probabilité d'un 3-gram
def proba(trig,d,a,n):
    if trig not in d:
        return 1/(a**3+(n-2))
        #return 1/(a*3+(n-2))
    else:
        return (d[trig]+1)/(a**3+(n-2))
        #return (d[trig]+1)/(a*3+(n-2))
        
#Organise les 2 fonctions au-dessus en calcule la KL-Divergence
def main_kl(corpus):
    train = fr_json[corpus+'train.json']
    test = fr_json[corpus+'test.json']
    d_train, g_train, alpha_train, n_train = kl_divergence_build(train)
    d_test, g_test, alpha_test, n_test = kl_divergence_build(test)
    alpha = alpha_test + alpha_train
    grams = g_train + g_test
    cpt = 0
    for gram in grams:
        train= proba(gram,d_train,len(alpha),n_train)
        test= proba(gram,d_test,len(alpha),n_test)
        cpt += test*math.log(test/train)
    return cpt

#Affichage de la KL divergence pour chaque corpus         
def kl_display():
    data = ['fr.gsd.','fr.partut.','fr.sequoia.','fr.spoken.','fr.ftb.','fr.pud.'] 
    print("KL Divergence des differents corpus:")  
    for i in data:
        name = i.split('.')
        print(name[1],':',main_kl(i))


#Je ne sais pas comment fonctionne ce truc, pas grand monde le sait d'ailleurs mais voilà
#corpus_size()
#print()
#calcul_oov_print()
#print()
#calcul_kl_divergence()


################################################
################################################
############
############   MATRICE CONFUSION
############
################################################
################################################

#Création et affichage de la matrice de confusion, matrice 2D composé de [label_voulu][label_prédit] pour chaque mot du corpus
def matrice_confusion(perceptron,corpus):
    dic = {}
    #Creation d'un dictionnaire de labels
    for i in perceptron.labels:
        dic[i] = {}
        for j in perceptron.labels:
            dic[i][j] = 0
    #Pour chaque mot du corpus, on itère [label_voulu][label_prédit]
    for i in range(len(corpus)):
        for j in range(len(i[0])):
            dic[corpus[i][1][j]][perceptron.predict(corpus[i][0][j])] += 1
    
    #Affichage (approximatif)
    for i in dic.keys():
        row = ""
        for j in i.keys():
            row+= str(dic[i][j])+" "
        print(row)


################################################
################################################
############
############   CHARGEMENT DES DONNEES 
############
################################################
################################################

path = './fr/'

def load_files():
    filenames = os.listdir(path)
    files = dict()
    for filename in filenames:
        if not filename.strip().endswith("json"):
            continue
        rang = filename.split('.')

        if files.get(rang[1], None) == None:
            files[rang[1]] = dict()    
        files[rang[1]][rang[2]] = json.load(open(path+filename))
    return files


# Donne la liste des labels du set
def mk_lbl_set(dataset):
    ret_dict = set()
    for j in range(len(dataset)):
        l = dataset[j][1]
        for i in l:
            ret_dict.add(i)
    return ret_dict

