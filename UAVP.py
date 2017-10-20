"""
The Unaccounted Variables Perspective.
"""


import numpy as np
from matplotlib import pyplot as plt

from sklearn.decomposition import pca
from sklearn.datasets import load_boston, load_diabetes
from sklearn.cross_validation import train_test_split

from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor

def UAVP(model_arch, data):
    X,y = load_boston().data, load_boston().target

    # Split data into required subsamples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.5)
    X_ass, y_ass = np.concatenate((X_train, X_val), axis= 0), np.concatenate((y_train, y_val), axis= 0)

    model_arch = svm.SVR()
    trained_model = model_arch.fit(X_train, y_train)

    print 'Conventionally trained Model (Training Set): ', trained_model.score(X_test, y_test)

    def obviate_backlash(model_arch, trained_model, X_ass, y_ass, X_test, f_unacc_t):
        f_unacc = np.array([trained_model.predict(X_ass)]).T
        X_com = np.concatenate((X_ass, f_unacc), axis= 1)
        model_com = model_arch.fit(X_com, y_ass)
        X_test_com = np.concatenate((X_test, f_unacc_t), axis= 1)
        return model_com, X_test_com

    model_ass = model_arch.fit(X_ass, y_ass)
    print 'Conventionally trained Model (Assessment Set):', model_ass.score(X_test, y_test)

    f_unacc_t = np.array([trained_model.predict(X_test)]).T
    model_com, X_test_com = obviate_backlash(model_arch, trained_model, X_ass, y_ass, X_test, f_unacc_t)

    print 'Model trained with UAVP:', model_com.score(X_test_com, y_test)
    print '\n'


def run():
    for model_arch in [svm.SVR(), DecisionTreeRegressor(), LogisticRegression(), MLPRegressor()]:

        print 'Results for:', str(model_arch), '\n'

        print 'Results on Boston Dataset: \n', UAVP(model_arch, load_boston)
        print 'Results on Diabetes Dataset: \n', UAVP(model_arch, load_diabetes)


run()
