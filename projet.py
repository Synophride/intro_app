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

for filename in fr_files :
    path = fr_path + filename
    j = json.load(open(path))
    arr = np.array(j)
    fr_json[filename] = arr
    print(filename, '\t', arr.shape)

def calcul_ooc(train_file, test_file):
    vocab_train = set()
    # 1 - construction du vocab
    for i in train_file:
        for j in i[0]:
            vocab_train.add(j)
    
    nb_out_vocab = 0
    for i in test_file:
        for j in i[0]:
            if j not in vocab_train:
                nb_out_vocab += 1
    return (nb_out_vocab / (len(vocab_train) + nb_out_vocab)) * 100

def calcul_kl_divergence():
    
    
gsd_beg = 'fr.gsd.'
gsd_train = fr_json[gsd_beg + 'train.json']
gsd_test = fr_json[gsd_beg + 'test.json']
print(calcul_ooc(gsd_train, gsd_test))

partut_beg = 'fr.partut.'
partut_train = fr_json[partut_beg + 'train.json']
partut_test = fr_json[partut_beg + 'test.json']
print('partut\t', calcul_ooc(partut_train, partut_test))

sequoia_beg = 'fr.sequoia.'
sequoia_train = fr_json[sequoia_beg + 'train.json']
sequoia_test = fr_json[sequoia_beg + 'test.json']
print('sequoia\t', calcul_ooc(sequoia_train, sequoia_test))

spoken_beg = 'fr.spoken.'
spoken_train = fr_json[spoken_beg + 'train.json']
spoken_test = fr_json[spoken_beg + 'test.json']
print('spoken\t', calcul_ooc(spoken_train, spoken_test))

ftb_beg = 'fr.ftb.'
ftb_train = fr_json[ftb_beg + 'train.json']
ftb_test = fr_json[ftb_beg + 'test.json']
print('ftb\t', calcul_ooc(ftb_train, ftb_test))

pud_beg = 'fr.pud.'
pud_train = fr_json[pud_beg + 'train.json']
pud_test = fr_json[pud_beg + 'test.json']
print('pud\t', calcul_ooc(pud_train, pud_test))
