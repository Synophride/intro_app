import multiclass_perceptron as mp
import helpers as pj

class Modele:

    ## Constructeur qui ne fait rien.
    def __init__(self):
        pass

    # Initialisation du modèle à partir des données d'entraînement
    # (utiles pour mk_model)
    def init_train(self, train_data):
        pass

    # Entrainement du modèle.
    def train(self, train_data, epoch = None):
        pass

    # Prédiction d'un label. Renvoie le label prédit.
    def predict(self, sentence, pos):
        pass

    # Renvoie un tuple (bonne prédiction, nombre d'exemples)
    def test(self, test_data):
        pass

    # donne une str indiquant quel est le type du classifieur
    def get_str(self):
        pass
    
    # teste la réussite sur les mots ambigus (pouvant avoir deux labels différents)
    # et les mots inconnus dans l'ensemble de test
    def test_ambigu_et_oov(self, test_data, train_data):
        (a_words, oov_words) = pj.mk_ambigu_and_oov_set(train_data, test_data)
        total_amb = 0
        good_amb = 0

        total_oov= 0
        good_oov = 0
        
        for x in test_data:
            sentence = x[0]
            labels = x[1]
            
            for i in range(len(sentence)):
                word = sentence[i]
                lbl = labels[i]
                
                if(word in a_words):
                    ypred = self.predict(sentence, i)
                    if(ypred == lbl):
                        good_amb += 1
                    total_amb +=1
                    
                if(word in oov_words):
                    ypred = self.predict(sentence, i)
                    if(ypred == lbl):
                        good_oov += 1
                    total_oov +=1
            
        return ((good_amb, total_amb), (good_oov, total_oov)) 
