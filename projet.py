import json
import os
import sys
import numpy as np


data_path = './corpus/'
fr_path = data_path + 'fr/'
en_path = data_path + 'en/'

fr_files = os.listdir(fr_path)
en_diles = os.listdir(en_path)

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
def calcul_kl_divergence():
    pass

corpus_size()
print()
calcul_oov_print()

