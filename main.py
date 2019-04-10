
import projet as pj

# modules implémentant divers modèles de features
import tp5      as tp
import mk_model as ml


# va entraîner les perceptrons avec train_data, le teste 
def train_and_mk_confusion_matrix(train_data, test_data):
    m_tp = tp.Modele_tp5()
    m_tp.init_train(train_data)
    m_tp.train(train_data)
    
    #    m_pj = ml.Modele_projet()
    # m_pj.init_train(train_data)
    # m_pj.train(train_data)

    mc = pj.matrice_confusion(m_tp, train_data, test_data)
    key_list = sorted(mc.keys())

    pass

def get_performances(train_set, test_set):
    m_tp = tp.Modele_tp5()
    m_tp.init_train(train_set)
    m_tp.train(train_set)
    (good, total) = m_tp.test(test_set)
    ((amb_g, amb_t),(oov_g,oov_t)) = m_tp.test_ambigu_et_oov(test_set, train_set)
    print("Performances générales :\t", good, '/', total)
    print("Performances ambiguité :\t", amb_g, '/', amb_t)
    print("Performances OOV : \t", oov_g, '/', oov_t)
        
data = pj.load_files()
train = data['gsd']['train']
test = data['gsd']['test']

#train_and_mk_confusion_matrix(train, test)
get_performances(train, test)
