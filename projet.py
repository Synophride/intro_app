import json
import os
import sys
import numpy as np


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
def calcul_kl_divergence(train,test):
    train_char = []
    test_char = []
    train_gram = []
    test_gram = []
    for i in train: 
        for j in i[0]:
            char_list = list(j)
            word_list = []
            for k in char_list:
                word_list.append(k)
            train_char.append(word_list)
    for i in test: 
        for j in i[0]:
            char_list = list(j)
            word_list = []
            for k in char_list:
                word_list.append(k)
            test_char.append(word_list)
    for i in range(len(train_char)):
        for j in range(2,len(train_char[i])):
            gram = train_char[i][j-2]+train_char[i][j-1]+train_char[i][j]
            if gram not in train_gram:
                train_gram.append(gram)
    for i in range(len(test_char)):
        for j in range(2,len(test_char[i])):
            gram = test_char[i][j-2]+test_char[i][j-1]+test_char[i][j]
            if gram not in test_gram:
                test_gram.append(gram)




corpus_size()
print()
calcul_oov_print()
print()
calcul_kl_divergence(fr_json['fr.sequoia.train.json'],fr_json['fr.sequoia.test.json'])
