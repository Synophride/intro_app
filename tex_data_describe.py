import matplotlib.pyplot as plt
import helpers as pj


datasets = pj.load_files()
path_output = './tex_descs/'
N = 10

explanation_of_models = {
    'foot' : 'Ensemble de tweets parlant de football',
    'partut' : 'Ensemble varié de textes, incluant de extraits de conférences comme des extraits de Wikipedia ou des textes légaux',
    'sequoia': 'Ensemble de texte, à la base disponible avec des annotations «profondes», ie plus complètes. Néanmoins, ces annotations ont été normalisées par rapport aux autres données',
    'spoken': 'Données sur le français oral',
    'pud' : 'Phrases qui proviennent de sites informatifs, id est de journaux ou de Wikipedia',
    'ftb' : 'Phrases provenant du journal «Le Monde»',
    'gsd' : 'Phrases variées',
    'natdis' : 'Ensemble de tweets'
    }


def wr(string):
    print(string)
    
def count_labels():
    ret = dict()
    return ret

def get_lbl_count(data):
    dico = dict()
    for i in data:
        sentence = i[1]
        for word in sentence:
            dico[word] = dico.get(word, 0) + 1
    return dico

def get_most_used_words(data):
    dico = dict()
    for i in data:
        sentence = i[0]
        for word in sentence:
            dico[word] = dico.get(word, 0) + 1
    liste = []
    n = 0
    for word in sorted(dico.items(), key = lambda x : x[1], reverse = True):
        liste.append( word) 
        n += 1
        if(n == N):
            break
    return liste

def parse_most_used_words(mow, setname = '' ):
    retour  = r'\begin{tabular}{|l || *{' + str(N+1) + r' }{|c} |} \hline' + '\n'
    str_arr = r'Mot & Apparitions  \\ \hline' + '\n'
    for (mot, nb) in mow:
        str_arr += r'\begin{verb} ' +  mot + r" \end{verb} &" + str(nb) + r'\\ \hline' + '\n'
    retour += str_arr + '\n'
    retour += r'\end{tabular}' + '\n'
    retour += r'\caption{ Mots les plus utilisés dans le set ' + setname + r' } \label{Fig:muw}'
    return retour


def mk_perplexity_tabular(corpus_name):
    train_set = datasets[corpus_name]['train']
    test_set  = datasets[corpus_name]['test']
    perc_oov = pj.calcul_oov(train_set, test_set)
    pass #todo 
    


def mk_label_histo(dico_lbl, path, name):
    labels = []
    nb = []
    for label in dico_lbl.keys():
        value = dico_lbl[label]
        labels.append(label)
        nb.append(value)
    plt.pie(nb, labels = labels, pctdistance = 5)
    # plt.xlabel(labels)
    plt.savefig(path, bbox_inches='tight')
    plt.clf()
    #     r'\begin{figure}[h] ' + '\n' \  
    #                + r'\end{figure} '
    str_ret  =   r'\includegraphics[width=.7\linewidth]{' + name + r'}' + '\n' \
               + r'\caption{distribution des labels}' 
    #               + r'\label{Fig:' + name + r'}' + '\n' 
    return str_ret



def parse_descriptors(dico_all):
    string_retour = ''

    for set_name in dico_all.keys():
        written_str = r"\subsection{" + set_name + ' } \n' \
                      + r' \begin{itemize} ' + '\n' + r' \item[Présentation :] ' \
                      + explanation_of_models.get(set_name, '')+ '\n\n'
        oov_str = '' 
        str_divergence = ''

        dico_name = dico_all[set_name]

        print('keys', dico_name.keys())
        if 'kl' in dico_name.keys() and 'oov' in dico_name.keys():
            str_divergence = r' \item[KL-Divergence :]' \
                             + dico_name['kl'] + '\n'

            oov_str = r' \item[Pourcentage de mots hors vocabulaire : ]' \
                      + dico_name["oov"] + '\n'
            
        written_str += (oov_str  + str_divergence + r' \end{itemize} ') 
        for set_type in sorted(dico_all[set_name].keys()):
            if(set_type == 'dev' or set_type == 'oov' or set_type == 'kl'): # on oublie les Dev
                continue
            
            dico = dico_name[set_type]
            
            path = path_output + set_name + set_type
            
            label_str = mk_label_histo(dico['labels'], path+'_img.png', set_name + set_type + '_img.png')
            
            str_muw = parse_most_used_words(dico['used_w'],
                                            setname = set_name + '(' + set_type +')'  )

            nb_examples = dico['length']
        
            written_str =  written_str \
                           + r' \paragraph{Données de ' + set_type + r' \\ }  ' + '\n' \
                           + r' Nombre de phrases : ' + str(nb_examples) + r'\\ ' + '\n' \
                           + r'\begin{figure}[H] \begin{minipage}{0.48\textwidth} \centering ' \
                           + str_muw + r'\end{minipage} ' + '\n' \
                           + r'\begin{minipage}{0.48\textwidth} \centering'+ '\n' \
                           + label_str + '\n' \
                           + r'\end{minipage}' + '\n' \
                           + r'\end{figure}'
            
        string_retour = string_retour + '\n\n\n' + written_str 
    x = open(path_output + 'description_donnees.tex', 'w')
    x.write(string_retour)
    x.close()
        
def write_descriptors():
    all_dicts = dict()
    kl_divergences = pj.get_kls(dset = datasets)
    
    for dataset_key in datasets.keys():
        if (dataset_key not in all_dicts.keys()):
            all_dicts[dataset_key] = dict()

        dataset_type = datasets[dataset_key]
        
        if 'train' in dataset_type.keys() and 'test' in dataset_type.keys():
            (out_vocab, nb_vocab_train) = pj.calcul_oov(dataset_type['train'], dataset_type['test'])
            all_dicts[dataset_key]['oov']= str((out_vocab*100) / nb_vocab_train)[:5]
            all_dicts[dataset_key]['kl'] = str(kl_divergences[dataset_key])[:5]
            
        for dataset in dataset_type.keys():
            if(dataset == 'oov' or dataset == 'kl'): # on oublie les Dev
                continue
            dico = dict()
            data = dataset_type[dataset]
            dico['used_w'] = get_most_used_words(data)
            dico['labels'] = get_lbl_count(data)
            dico['length'] = len(data)
            all_dicts[dataset_key][dataset] = dico
            
    return all_dicts

parse_descriptors(write_descriptors())

