import modele as m
import helpers as hp
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
"""
     'is_first': index == 0,
     'is_last': index == len(sentence) - 1,
"""
def mk_features(sentence, index):
    return {
        'word': sentence[index],
        'is_capitalized': sentence[index][0].isupper(),
        'is_all_caps': sentence[index].isupper(),
        'prefix-3': sentence[index][:3],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def trs_data(data):
    features_list = []
    label_list    = []
    for x in data:
        sentence = x[0]
        lbls = x[1]
        for i in range(len(sentence)):
            features_list.append(mk_features(sentence, i))
            label_list.append(lbls[i])
    return (features_list, label_list)


class Modele_skl(m.Modele):
    def __init__(self):
        pass
    
    def init_train(self, train_data):
        pass
        
    def train(self, train_data, epoch=-1):
        (feats, labels) = trs_data(train_data)
        self.clf.fit(feats, labels)
            
    def predict(self, sentence, pos):
        features = mk_features(sentence, pos)
        prediction = self.clf.predict(features)
        return prediction
        
    def test(self, test_data):
        total, good = (0,0)
        for x in test_data:
            sentence = x[0]
            labels   = x[1]
            for i in range(len(sentence)):
                ypred = self.predict(sentence, i)
                y = labels[i]
                if ypred == y:
                    good += 1
                total +=1
        return (good, total)

class Modele_sklearn_Perceptron(Modele_skl):   
    def init_train(self, train_data):
        self.labels = hp.mk_lbl_set(train_data)
        self.clf = Pipeline( [('vectorizer', DictVectorizer()),
                              ('classifier', Perceptron()) ])
    def get_str(self):
        return 'SKLearn Perceptron'
    
class Modele_sklearn_SVM(Modele_skl):
    def init_train(self, train_data):
        self.labels = hp.mk_lbl_set(train_data)
        self.clf = Pipeline( [('vectorizer', DictVectorizer()),
                              ('classifier', LinearSVC()) ])
    def get_str(self):
        return 'SKLearn SVM'

    
