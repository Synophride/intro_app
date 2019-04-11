
import helpers as pj

# modules implémentant divers modèles de features
import modele_tp5      as tp
import modele_pj_distrib as ml

# va entraîner les perceptrons avec train_data, le teste 
def train_and_mk_confusion_matrix(train_data, test_data):

    m_tp = tp.Modele_tp5()
    m_tp.init_train(train_data)
    m_tp.train(train_data)
    
    m_pj = ml.Modele_projet()
    m_pj.init_train(train_data)
    m_pj.train(train_data)

    mc = pj.matrice_confusion(m_tp, train_data, test_data)
    key_list = sorted(mc.keys())
    pass


def get_performances(perceptron, train_set, test_set):
    dict_tp5 = dict()
    m_tp = tp.Modele_tp5()
    m_tp.init_train(train_set)
    m_tp.train(train_set)
    dict_tp5['perf'] =  m_tp.test(test_set)
    (dict_tp5['amb'], dict_tp5['oov']) = m_tp.test_ambigu_et_oov(test_set, train_set)

    dict_pj = dict()
    m_pj = ml.Modele_projet()
    m_pj.init_train(train_set)
    m_pj.train(train_set)
    dict_pj['perf'] =  m_pj.test(test_set)
    (dict_pj['amb'], dict_pj['oov']) = m_pj.test_ambigu_et_oov(test_set, train_set)

    return (dict_tp5, dict_pj)

data = pj.load_files()
train = data['gsd']['train']
test = data['gsd']['test']

#train_and_mk_confusion_matrix(train, test)
perfs = get_performances(train, test)
