#!/usr/bin/python3

import os
import sys
import code
import warnings
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from time import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.metrics import f1_score

from sklearn.exceptions import ConvergenceWarning
#local project packages
from dataset_utils import get_normalized_data, filename_se, filename_ho, SEED

#adapted from snippet found here from https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def get_cached_params_se():
    #these parameters were generated using the get_params_se() function
    return {'dtr_params': {'ccp_alpha': 0.005, 'class_weight': 'balanced', 'criterion': 'gini', 'random_state': 903539276}, 'ada_params': {'algorithm': 'SAMME.R', 'base_estimator': DecisionTreeClassifier(ccp_alpha=0.025, class_weight='balanced', random_state=903539276), 'learning_rate': 0.8, 'n_estimators': 69, 'random_state': 903539276}, 'nnt_params_adm': {'activation': 'logistic', 'hidden_layer_sizes': (30,), 'max_iter': 200, 'random_state': 903539276, 'solver': 'adam'}, 'nnt_params_lbfgs': {'activation': 'relu', 'hidden_layer_sizes': (34,), 'max_iter': 200, 'random_state': 903539276, 'solver': 'lbfgs'}, 'nnt_params_sgd': {'activation': 'relu', 'hidden_layer_sizes': (25,), 'learning_rate': 'constant', 'max_iter': 200, 'random_state': 903539276, 'solver': 'sgd'}, 'knn_params': {'algorithm': 'ball_tree', 'leaf_size': 30, 'n_jobs': 14, 'n_neighbors': 6, 'p': 1, 'weights': 'distance'}, 'svm_params_rbf': {'C': 10, 'class_weight': None, 'gamma': 0.001, 'kernel': 'rbf', 'random_state': 903539276}, 'svm_params_sigmoid': {'C': 2000, 'class_weight': 'balanced', 'gamma': 2e-05, 'kernel': 'sigmoid', 'random_state': 903539276}, 'svm_params_poly': {'C': 2e-05, 'class_weight': 'balanced', 'degree': 1, 'gamma': 2000, 'kernel': 'poly', 'random_state': 903539276}, 'svm_params_linear': {'C': 1.0, 'class_weight': 'balanced', 'kernel': 'linear', 'random_state': 903539276}}

def get_cached_params_ho():
    #these parameters were generated using the get_params_ho() function
    pass

def get_params_se():
    params = {}
    data = get_normalized_data(filename_se, 0.5)
    train_feat = data[0]
    train_class = data[1]

    ### DecisionTreeClassifier
    param_grid = {
        'ccp_alpha': [i/1000 for i in range(0, 26, 1)],
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy'],
        'random_state': [SEED]}

    classifier = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=2, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    dtr_params = classifier.best_params_
    params['dtr_params'] = dtr_params
    print("DecisionTreeClassifier best parameters:  " + str(dtr_params))
    evaluate_params(DecisionTreeClassifier(**dtr_params), *data, True)

    ### AdaBoostClassifier with highly pruned Decision Tree learner
    ada_base_params = dict(dtr_params) #make a copy to avoid modifying original
    ada_base_params['ccp_alpha'] = 0.025

    param_grid = {
        'base_estimator': [DecisionTreeClassifier(**ada_base_params)],
        'n_estimators': [i for i in range(60, 71)],
        'learning_rate': [i/10 for i in range(2, 10, 1)],
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': [SEED]}

    classifier = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    ada_params = classifier.best_params_
    params['ada_params'] = ada_params
    print("AdaBoostClassifier best parameters:  " + str(ada_params))
    evaluate_params(AdaBoostClassifier(**ada_params), *data, True)

    ### MLPClassifier
    ## solver: adam
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(25, 36)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'max_iter': [200],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    nnt_params_adm = classifier.best_params_
    params['nnt_params_adm'] = nnt_params_adm
    print("MLPClassifier solver=adam best parameters:  " + str(nnt_params_adm))
    evaluate_params(MLPClassifier(**nnt_params_adm), *data, True)

    ## solver: lbfgs
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(28, 39)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs'],
        'max_iter': [200],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    nnt_params_lbfgs = classifier.best_params_
    params['nnt_params_lbfgs'] = nnt_params_lbfgs
    print("MLPClassifier solver=lbfgs best parameters:  " + str(nnt_params_lbfgs))
    evaluate_params(MLPClassifier(**nnt_params_lbfgs), *data, True)

    ## solver: sgd
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(15, 30)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    nnt_params_sgd = classifier.best_params_
    params['nnt_params_sgd'] = nnt_params_sgd
    print("MLPClassifier solver=sgd best parameters:  " + str(nnt_params_sgd))
    evaluate_params(MLPClassifier(**nnt_params_sgd), *data, True)

    ### KNeighborsClassifier
    param_grid = {
        'n_neighbors': [i for i in range(1, 13)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30], #leaf size does not seem to affect accuracy.  Larger trains slightly faster
        'p': [1, 2],
        'n_jobs': [14]}

    classifier = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    knn_params = classifier.best_params_
    params['knn_params'] = knn_params
    print("KNeighborsClassifier best parameters:  " + str(knn_params))
    evaluate_params(KNeighborsClassifier(**knn_params), *data, True)

    ### SVC
    ## kernel: rbf
    param_grid = {
        'C': [10**i for i in range(-5, 6)],
        'kernel': ['rbf'],
        'gamma': [10**i for i in range(-5, 6)],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_rbf = classifier.best_params_
    params['svm_params_rbf'] = svm_params_rbf
    print("SVC() kernel=rbf best parameters:  " + str(svm_params_rbf))
    evaluate_params(SVC(**svm_params_rbf), *data, True)

    ## kernel: sigmoid
    param_grid = {
        'C': [2*10**i for i in range(-5, 6)],
        'kernel': ['sigmoid'],
        'gamma': [2*10**i for i in range(-5, 6)],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_sigmoid = classifier.best_params_
    params['svm_params_sigmoid'] = svm_params_sigmoid
    print("SVC() kernel=sigmoid best parameters:  " + str(svm_params_sigmoid))
    evaluate_params(SVC(**svm_params_sigmoid), *data, True)

    ## kernel: poly
    param_grid = {
        'C': [2*10**i for i in range(-5, 6)],
        'kernel': ['poly'],
        'degree': [1, 2, 3],
        'gamma': [2*10**i for i in range(-5, 6)],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_poly = classifier.best_params_
    params['svm_params_poly'] = svm_params_poly
    print("SVC() kernel=poly best parameters:  " + str(svm_params_poly))
    evaluate_params(SVC(**svm_params_poly), *data, True)

    ## kernel: linear
    param_grid = {
        'C': [i/10 for i in range(10, 30, 1)],
        'kernel': ['linear'],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_linear = classifier.best_params_
    params['svm_params_linear'] = svm_params_linear
    print("SVC() kernel=linear best parameters:  " + str(svm_params_linear))
    evaluate_params(SVC(**svm_params_linear), *data, True)

    return params

def get_params_ho():
    params = {}
    data = get_normalized_data(filename_ho, 0.5)
    train_feat = data[0]
    train_class = data[1]

    ### DecisionTreeClassifier
    param_grid = {
        'ccp_alpha': [i/1000 for i in range(0, 26, 1)],
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy'],
        'random_state': [SEED]}

    classifier = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=2, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    dtr_params = classifier.best_params_
    params['dtr_params'] = dtr_params
    print("DecisionTreeClassifier best parameters:  " + str(dtr_params))
    evaluate_params(DecisionTreeClassifier(**dtr_params), *data, True)

    ### AdaBoostClassifier with highly pruned Decision Tree learner
    ada_base_params = dict(dtr_params) #make a copy to avoid modifying original
    ada_base_params['ccp_alpha'] = 0.025

    param_grid = {
        'base_estimator': [DecisionTreeClassifier(**ada_base_params)],
        'n_estimators': [i for i in range(65, 105, 5)],
        'learning_rate': [i/10 for i in range(2, 8, 1)],
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': [SEED]}

    classifier = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    ada_params = classifier.best_params_
    params['ada_params'] = ada_params
    print("AdaBoostClassifier best parameters:  " + str(ada_params))
    evaluate_params(AdaBoostClassifier(**ada_params), *data, True)

    ### MLPClassifier
    ## solver: adam
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(30, 45)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'max_iter': [200],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    nnt_params_adm = classifier.best_params_
    params['nnt_params_adm'] = nnt_params_adm
    print("MLPClassifier solver=adam best parameters:  " + str(nnt_params_adm))
    evaluate_params(MLPClassifier(**nnt_params_adm), *data, True)

    ## solver: lbfgs
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(35, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs'],
        'max_iter': [200],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    nnt_params_lbfgs = classifier.best_params_
    params['nnt_params_lbfgs'] = nnt_params_lbfgs
    print("MLPClassifier solver=lbfgs best parameters:  " + str(nnt_params_lbfgs))
    evaluate_params(MLPClassifier(**nnt_params_lbfgs), *data, True)

    ## solver: sgd
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(20, 35)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['sgd'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    nnt_params_sgd = classifier.best_params_
    params['nnt_params_sgd'] = nnt_params_sgd
    print("MLPClassifier solver=sgd best parameters:  " + str(nnt_params_sgd))
    evaluate_params(MLPClassifier(**nnt_params_sgd), *data, True)

    ### KNeighborsClassifier
    param_grid = {
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30], #leaf size does not seem to affect accuracy.  Larger trains slightly faster
        'p': [1, 2],
        'n_jobs': [14]}

    classifier = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    knn_params = classifier.best_params_
    params['knn_params'] = knn_params
    print("KNeighborsClassifier best parameters:  " + str(knn_params))
    evaluate_params(KNeighborsClassifier(**knn_params), *data, True)

    ### SVC
    ## kernel: rbf
    param_grid = {
        # 'C': [i/10 for i in range(5, 81, 5)],
        'C': [10**i for i in range(-5, 6)],
        'kernel': ['rbf'],
        # 'gamma': [1/i for i in range(2, 17)],#['scale'],
        'gamma': [10**i for i in range(-5, 6)],#['scale'],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_rbf = classifier.best_params_
    params['svm_params_rbf'] = svm_params_rbf
    print("SVC() kernel=rbf best parameters:  " + str(svm_params_rbf))
    evaluate_params(SVC(**svm_params_rbf), *data, True)

    ## kernel: sigmoid
    param_grid = {
        # 'C': [i/10 for i in range(30, 41, 1)],
        'C': [10**i for i in range(-5, 6)],
        'kernel': ['sigmoid'],
        # 'gamma': [1/i for i in range(68, 76, 1)],#['scale'],
        'gamma': [10**i for i in range(-5, 6)],#['scale'],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_sigmoid = classifier.best_params_
    params['svm_params_sigmoid'] = svm_params_sigmoid
    print("SVC() kernel=sigmoid best parameters:  " + str(svm_params_sigmoid))
    evaluate_params(SVC(**svm_params_sigmoid), *data, True)

    ## kernel: poly
    param_grid = {
        'C': [10**i for i in range(-3, 4)],#[i/10 for i in range(5, 81, 5)],
        'kernel': ['poly'],
        'degree': [1, 2, 3],
        'gamma': [10**i for i in range(-3, 4)],#[1/i for i in range(5, 100, 5)],#['scale'],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_poly = classifier.best_params_
    params['svm_params_poly'] = svm_params_poly
    print("SVC() kernel=poly best parameters:  " + str(svm_params_poly))
    evaluate_params(SVC(**svm_params_poly), *data, True)

    ## kernel: linear
    param_grid = {
        'C': [10**i for i in range(-2, 3)],#[i/10 for i in range(1, 20, 1)],
        'kernel': ['linear'],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    classifier = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=192)
    classifier.fit(train_feat, train_class)
    svm_params_linear = classifier.best_params_
    params['svm_params_linear'] = svm_params_linear
    print("SVC() kernel=linear best parameters:  " + str(svm_params_linear))
    evaluate_params(SVC(**svm_params_linear), *data, True)

    return params

def evaluate_params(classifier, train_feat, train_class, test_feat, test_class, verbose=False):
    train_time = int(time() * 1000) #milliseconds
    classifier.fit(train_feat, train_class)
    train_time = int(time() * 1000) - train_time

    infer_time = int(time() * 1000) #milliseconds
    train_pred = classifier.predict(train_feat)
    infer_time = int(time() * 1000) - infer_time

    test_pred = classifier.predict(test_feat)

    train_score = f1_score(train_class, train_pred, average=None)
    train_score_mean = train_score.mean()
    test_score = f1_score(test_class, test_pred, average=None)
    test_score_mean = test_score.mean()

    if verbose:
        print("train score:  " + str(train_score))
        print("train score mean:  " + str(train_score_mean))
        print("test set score:  " + str(test_score))
        print("test set score mean:  " + str(test_score_mean))
        print()

    return train_score, train_score_mean, test_score, test_score_mean, train_time, infer_time
