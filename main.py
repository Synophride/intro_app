import modeles_sklearn as ms
import helpers as pj

# modules implémentant divers modèles de features
import modele_tp5      as tp
import modele_pj_distrib as ml

# va entraîner les perceptrons avec train_data, le teste 
def train_and_mk_confusion_matrix(perc, train_data, test_data):

    perc.init_train(train_data)
    perc.train(train_data)

    mc = pj.matrice_confusion(perc, train_data, test_data)
    key_list = sorted(mc.keys())
    pass


def get_performances(p, train_set, test_set):
    p.init_train(train_set)
    p.train(train_set)
    print("résultat :", p.test(test_set))


data = pj.load_files()
train = data['gsd']['train']
test = data['gsd']['test']

p1 = ms.Modele_sklearn_Perceptron()
p2 = ms.Modele_sklearn_SVM()

get_performances(p1, train, test)
get_performances(p2, train, test)
