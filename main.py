
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

    str_ret = '\t' + '\t'.join(sorted(mc[key_list[0]].keys())) + '\n' 
    for k in key_list:
        str_ret += k +'\t' #affichage du label
        for j in sorted(mc[k].keys()):
            str_ret += str(mc[k][j])+'\t'  # affichage du score 
        str_ret += '\n'

    print(str_ret)

data = pj.load_files()
train = data['gsd']['train']
test = data['gsd']['test']

train_and_mk_confusion_matrix(train, test)
