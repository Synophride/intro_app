import numpy
import json
import multiclass_perceptron as mp
path='./fr/'

full_path = path + 'fr.pud.train.json'
train_set = json.load(open(full_path))
test_set  = json.load(open(path + 'fr.foot.test.json'))

np_arr = numpy.array(train_set)

# Donne la liste des labels
def mk_lbl_set(dataset):
    ret_dict = set()
    for j in range(len(dataset)):
        l = dataset[j][1]
        for i in l:
            ret_dict.add(i)
    return ret_dict


lbl_set = mk_lbl_set(np_arr)

    
"""
Construit une représentation éparse a priori meilleure
[00:11]Rémi:
  du coup, moi ce que j'ai fait c'est que mon dictionnaire est comme ça : 
  clé = critère + valeur du critère 
  valeur = 1
"""
def build_sparse2(sentence, pos):
    ret = dict()
    word = sentence[pos]
    ret['l3c'+word[-3:]] = 1
    ret['lc'+word[-1]] = 1
    ret['fc'+word[0]]  = 1
    ret['w'+word]      = 1
    ret['first_upper'+str(1 if word[0].isupper() else 0)] = 1
    ret['all_upper'+str(1 if word.isupper() else 0)] = 1 
    ret['w_-1'+ (sentence[pos-1] if pos > 0 else '')] = 1
    ret['w_-2'+ (sentence[pos-2] if pos > 1 else '')] = 1
    ret['w_+1'+ (sentence[pos+1] if pos < len(sentence)-1 else '') ] = 1 
    ret['w_+2'+ (sentence[pos+2] if pos < len(sentence)-2 else '')] = 1
    return ret


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
            representation= build_sparse2(sentence, i)
            y = labels[i]
            ypred = perc.predict(representation)
            if(y == ypred):
                good +=1
            total += 1
    return (good, total)

##############
# exécution  #
##############

print('Labels :')
for i in lbl_set:
    print(i)
print('fin des labels')

perc = mp.Perceptron(lbl_set)

(g, t) = test(test_set, perc)
print('Avant entraînement : ', (g/t) * 100, '% justes')

# x = un tableau [[phrase]; [labels]]
for x in train_set :
    sentence= x[0] 
    labels  = x[1]
    for i in range(len(sentence)):
        representation = build_sparse2(sentence, i)
        perc.train(representation, labels[i])

(g, t) = test(test_set, perc)
print('Après entraînement : ', (g/t) * 100, '% justes')
