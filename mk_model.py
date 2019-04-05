import numpy
import json

def build_sparse(sentence, pos):
    word = sentence[pos]
    ret = dict()

    ret['word' + word] = 1
    for i in range(1,3):
        ret['word_m' + str(i) + (sentence[pos - i] if (pos > i) else '')] = 1
        ret['word_p' + str(i) + (sentence[pos + i] if pos+i < len(sentence) else '')] = 1

    ret['count_left'+ str(pos-1) ] = 1
    ret['count_right' + str(len(sentence) - pos)] = 1
    
