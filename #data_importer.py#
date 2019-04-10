import numpy
import json
import os

path='./fr/'

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

