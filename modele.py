import multiclass_perceptron as mp

class Modele:
    ## Non-initialisation du modèle. Ne fait rien
    def __init__(self):
        pass

    # Initialisation du modèle à partir des données d'entraînement
    # (utiles pour mk_model)
    def init_train(self, train_data):
        pass

    # Entrainement du modèle
    def train(self, train_data):
        pass

    # Remise à zéro du modèle (il faudra l'initialiser
    # de nouveau avec init_train)
    # peut-être inutile
    def reset(self):
        pass

    # Prédiction d'un label. Renvoie le label prédit
    def predict(self, sentence, pos):
        pass

    # Renvoie un tuple (bonne prédiction, nombre d'exemples)
    def test(self, test_data):
        pass
