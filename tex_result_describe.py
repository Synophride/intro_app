import helpers as pj
import modele_tp5 as tp
import modele_pj_distrib as ml
import modele_pj_nodistrib as nb
import modeles_sklearn as ms

path_first_result = './tex_descs/'

donnees = pj.load_files()

def str_perc(paire):
    (a, b) = paire
    if(b == 0):
        return 'xx'
    else:
        return str((100*a)/b)
    
def get_res(modele):
    dico = dict() # dictionnaire contenant les performance du perceptron du tp5
    for i in donnees.keys():
        if('train' not in donnees[i].keys()):
            continue
        dico[i] = dico.get(i, dict())
        train_data = donnees[i]['train']
        
        print("Entraînement pour le dataset ", i)
        modele.init_train(train_data)
        modele.train(train_data)
        
        print("Fin de l'entraînement. Début des tests")
        
        for j in donnees.keys():
            if 'test' not in donnees[j].keys():
                continue
            test_data = donnees[j]['test']
            
            dico[i][j] = dico[i].get(j, dict())
            
            d_test_p1 = dict()
            d_test_p1['perf'] = modele.test(test_data)
            (d_test_p1['amb'], d_test_p1['oov']) = modele.test_ambigu_et_oov(test_data, train_data)
            dico[i][j] = d_test_p1            
    return dico

def parse_res(dico, path_out, caption = 'taux de réussite'):
    d1 = dico
    endl = '\n'
    nb_train_data = len(d1)
    names_train = sorted(d1.keys())
    test_names = sorted(d1[names_train[0]].keys())
    nb_of_test_data = len(test_names)
    len_table = 1 + 3*nb_train_data
    
    str_first_dict = r'\begin{figure}[H] \begin{adjustbox}{width=\textwidth} \begin{centering} \begin{tabular}{ | l || *{ ' \
                     + str(nb_train_data) \
                     + r'}{c|c|c||} } \hline ' + endl

    for i in names_train:
        str_first_dict += r'& \multicolumn{3}{|c|}{ ' + i + r' } '

    str_first_dict += r' \\ \hline ' + endl
    
    for i in range(len(names_train)):
        str_first_dict += r'& OOV & AMB & GEN '

    str_first_dict += r'  \\ \hline \hline ' + endl

    for i in range(len(test_names)):
        name_test_set = test_names[i]
        str_first_dict += name_test_set + ' '

        for j in range(len(names_train)):
            name_train_set = names_train[j]
            current_dict  = d1[name_train_set][name_test_set]

            n = 5
            str_oov = str_perc(current_dict['oov'] )[:n]
            str_amn = str_perc(current_dict['amb'] )[:n]
            str_gen = str_perc(current_dict['perf'])[:n]

            str_first_dict += r' & ' + str_oov + ' & ' + str_amn + ' & ' + str_gen + endl
        str_first_dict += r' \\ \hline ' + endl

    str_first_dict += r' \end{tabular} \end{centering} \end{adjustbox} \caption{ ' + caption + r'} \end{figure} '
    
    fdesc = open(path_out, 'w')
    fdesc.write(str_first_dict)
    fdesc.close()


def write_small_result_array(modele_list, train_data, test_data, filename):
    ## 1. Calcul
    dct = dict()
    for p in modele_list:
        p.init_train(train_data)
        p.train(train_data)
        (g, t) = p.test(test_data)
        dct[p.get_str()] = str((g*100)/t)[:5]

    ## 2. Calcul de la str résultante
    str_to_write = r' \begin{figure}[H] ·\centering \begin{tabular}{ |l|| *{' + str(len(dct)) \
                   + r'}{c|} }\hline' + '\n'
    str_up = 'Classifieur  '
    str_d  = 'Pourcentage de réussite  '
    
    for perc_name in dct.keys():
        str_up +=  ' & ' + perc_name
        str_d  +=  ' & ' + dct[perc_name]

    str_to_write += (str_up + r' \\ \hline ' + str_d + r' \\ \hline ' \
                     + r' \end{tabular} \caption{Taux de réussite sur les données pour divers clasifieurs} \end{figure}' )

    fn = open(filename, 'w')
    fn.write(str_to_write)

    
modeles = [tp.Modele_tp5() ,
           nb.Modele_nodistrib(),
           ml.Modele_projet(),
           ms.Modele_sklearn_Perceptron(),
           ms.Modele_sklearn_SVM()]

train_set = donnees['ftb']['train']
test_set  = donnees['ftb']['test']

filename = 'tex_descs/fast_results.tex'

write_small_result_array(modeles, train_set, test_set, filename)

for i in modeles:
    print(i.get_str().replace(' ', '_'))
    parse_res(get_res(i),
              path_first_result + (i.get_str().replace(' ', '_')) + '.tex',
              caption= "Taux de réussite détaillés pour le perceptron " + i.get_str()
    )
