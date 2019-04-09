import matplotlib.pyplot as plt
import data_importer as di

datasets = di.load_files()
path_output = './tex_descs/'
N = 10
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

def parse_most_used_words(mow):
    retour = r'\begin{tabular}{|l || *{' + str(N+1) + r' }{|c} |} \hline'
    retour += '\n'
    str_up = r'Mot & '
    str_down = r"apparitions & "
    for (mot, nb) in mow:
        str_up += r'\begin{verb} ' + mot + r' \end{verb} & '
        str_down += str(nb) + ' & '

    retour = retour +  str_up + r'\\ \hline' + '\n'
    retour = retour + str_down+ r'\\ \hline' + '\n'
    retour += r'\end{tabular} ' + '\n'
    return retour

def mk_label_histo(dico_lbl, path):
    labels = []
    nb = []
    for label in dico_lbl.keys():
        labels.append(label)
        nb.append(dico_lbl[label])
    histo = plt.hist(nb, label = labels)    
    # todo : string latex qui inclut l'image
    return 'Image du label fdp'
        
def parse_descriptors(dico_all):
    string_retour = ''
    for set_name in dico_all.keys():
        written_str = r"\subsubsection{" + set_name + '} \n'
        # todo : Ajouter description brève
        
        dico_name = dico_all[set_name]
        for set_type in dico_all[set_name].keys():
            dico = dico_name[set_type]
            
            path = path_output + set_name + set_type
            
            label_str = mk_label_histo(dico['labels'], path+'_img.png')
            
            str_muw = parse_most_used_words(dico['used_w'])

            nb_examples = dico['length']
        
            written_str =  written_str \
                           + r'\paragraph{' + set_type + '}' + '\n' \
                           + r'\subparagraph{Taille du dataset}' + str(nb_examples) + '\n' \
                           + r'\subparagraph{Mots les plus utilisés\\}' + '\n' + str_muw + '\n' \
                           + r'\subparagraph{Distribution des labels}' + '\n'+ label_str + '\n'
        string_retour = string_retour + '\n\n\n' + written_str 
    x = open(path_output + 'fin.tex', 'w')
    x.write(string_retour)
    x.close()
        
def write_descriptors():
    all_dicts = dict()
    for dataset_key in datasets.keys():
        if (dataset_key not in all_dicts.keys()):
            all_dicts[dataset_key] = dict()
            
        dataset_type = datasets[dataset_key]
        
        for dataset in dataset_type.keys():
            dico = dict()
            data = dataset_type[dataset]
            dico['used_w'] = get_most_used_words(data)
            dico['labels'] = get_lbl_count(data)
            dico['length'] = len(data)
            all_dicts[dataset_key][dataset] = dico
            
    return all_dicts

parse_descriptors(write_descriptors())
            
