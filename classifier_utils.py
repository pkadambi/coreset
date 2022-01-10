import numpy as np
import ml_insights as mli
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# random_state = 123455

""" Train baseline classifiers """
#see how to make these into object oriented
def train_classifier(xtrain, ytrain, classifier):
    
    if 'svm' in classifier:
        clf = SVC(gamma = 'auto', probability=True, max_iter = 10000, class_weight='balanced')
        
    elif 'forest' in classifier:
        clf = RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', n_jobs=4)
    
    elif 'logistic' in classifier:
        clf = LogisticRegression(solver='lbfgs', class_weight='balanced_sample')
    elif 'mlp' in classifier:
#         clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu',
#                             hidden_layer_sizes=(10, 10, 5), random_state=1, 
#                             max_iter = 10000) 
        clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu',
                            hidden_layer_sizes=(16, 32, 32, 16), random_state=1, 
                            max_iter = 10000) 
    clf.fit(xtrain, ytrain)
    return clf

""" Classifier calibration functions"""

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

def get_platt_calibrator(yhat_uncalibrated, ycalib):
    lrcalib = LogisticRegression(C=999999999999, solver='lbfgs')
    lrcalib.fit(yhat_uncalibrated, ycalib)
    return lrcalib

def platt_calibrate(trained_calibrator, yhat):
    return trained_calibrator.predict_proba(yhat)[:, 1]

def get_isotonic_calibrator(yhat_uncalibrated, ycalib):
    iso_calib = IsotonicRegression(out_of_bounds = 'clip')
    iso_calib.fit(yhat_uncalibrated, ycalib)
    return iso_calib

def isotonic_calibrate(trained_calibrator, yhat):
    return trained_calibrator.predict(yhat)

# TODO
class CalibratedClassifier:
    def __init__():
        pass


""" Metrics """

def accuracy(yhat, y):
    return np.mean(yhat==y)

def mse(yhat, y):
    return np.mean((yhat - y) ** 2)

def xentropy(p, q, eps=1e-6):
    return -np.mean(np.sum(p * np.log(q + eps), axis=0))

def evaluate_classifier(clf, xdata, ydata, score_fn):
    return score_fn(clf.predict(xdata), ydata)

def get_uncalibrated_probas(clf, xdata, classifier):
    if classifier.lower()=='svm':
        pass
    elif classifier.lower()=='forest':
        pass
    else:
        print('Classifier must be either SVM or Forest')
        return Exception