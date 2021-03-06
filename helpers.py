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
    vocab_test = set()
    voc_out = set()
    nb_out_vocab = 0

    for i in train_file:
        for j in i[0]:
            vocab_train.add(j)

    for k in test_file:
        for l in k[0]:
            vocab_test.add(l)

    
    for i in vocab_test:
        if(i not in vocab_train and i not in voc_out):
            voc_out.add(i)
            nb_out_vocab +=1
    
    return (nb_out_vocab, len(vocab_train) + nb_out_vocab)


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

#Renvoie la probabilité d'un 3-gram (Formule donnée par le prof)
def proba(trig,d,a,n):
    if trig not in d:
        return 1/(a*3+(n-2))
    else:
        return (d[trig]+1)/(a*3+(n-2))
        
#Organise les 2 fonctions au-dessus en calcule la KL-Divergence
# Corpus : Ensemble des datasets
def main_kl(corpus, set_name):
    train = corpus[set_name]['train']
    test = corpus[set_name]['test']
    
    d_train, g_train, alpha_train, n_train = kl_divergence_build(train)
    d_test, g_test, alpha_test, n_test = kl_divergence_build(test)
    alpha = alpha_test + alpha_train
    grams = g_train + g_test
    cpt = 0
    for gram in grams:
        train= proba(gram,d_train,len(alpha),n_train)
        test = proba(gram,d_test,len(alpha),n_test)
        cpt += test*math.log(test/train)
    return cpt

#Affichage de la KL divergence pour chaque corpus         
def kl_display(dset = load_files()):
    data = ['gsd','partut','sequoia','spoken','ftb.','pud'] 
    print("KL Divergence des differents corpus:")  
    for i in data:
        print(i,':',main_kl(dset, i))

def get_kls(dset = load_files()):
    d = dict()
    data = ['gsd','partut','sequoia','spoken','ftb','pud']
    for nom in data:
        d[nom] = main_kl(dset, nom)
    return d
        
################################################
################################################
############
############   Fonction testant les résultats
############
################################################
################################################

#Création de la matrice de confusion, matrice 2D composé de [label_voulu][label_prédit] pour chaque mot du corpus
def matrice_confusion(modele, train_set, test_set):
    train_labels = mk_lbl_set(train_set)
    test_labels  = mk_lbl_set( test_set)
    dic = {}
    #Creation d'un dictionnaire de labels
    for i in test_labels:
        dic[i] = {}
        for j in train_labels:
            dic[i][j] = 0
            
    #Pour chaque mot du corpus, on incrémente [label_voulu][label_prédit]
    for i in test_set:
        sentence = i[0]
        labels = i[1]
        for j in range(len(sentence)):
            prediction = modele.predict(sentence, j)
            label = labels[j]
            dic[label][prediction] = dic[label][prediction] + 1
    return dic

def mk_ambigu_and_oov_set(train, test ):
    # mots ambigus
    amb_words = set()
    word_dicts = dict()
    
    for x in train:
        sentence = x[0]
        labels = x[1]
        for i in range(len(sentence)):
            word = sentence[i]
            lbl = labels[i]
            if word in word_dicts.keys():
                if(word_dicts[word] != lbl):
                    amb_words.add(word)
            else:
                word_dicts[word] = lbl

    vocab = set(word_dicts.keys())
    oov_words = set()
    for x in test:
        sentence = x[0]
        labels = x[1]
        for i in range(len(sentence)):
            word = sentence[i]
            lbl = labels[i]
            if(word not in vocab):
                oov_words.add(word)

    return (amb_words, oov_words)
    
