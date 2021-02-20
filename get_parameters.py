#!/usr/bin/python3

import os
import numpy as np
import pandas as pd
import sklearn as sk

from time import time
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import code #note to self, remove after debugging complete

SEED = 903539276 #GTid random seed for repeatability

def get_dt_params(train_feat, train_class, *args):
    param_grid = {
        'ccp_alpha': float_range(0.0, 0.025, 0.001),
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy'],
        'random_state': [SEED]}

    classifier = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=14)
    classifier.fit(train_feat, train_class)
    return classifier.best_params_

def get_adaboost_params(train_feat, train_class, *args):
    dt_params = get_dt_params(train_feat, train_class)
    dt_params['ccp_alpha'] = 0.025

    param_grid = {
        'base_estimator': [DecisionTreeClassifier(**dt_params)],
        'n_estimators': [i for i in range(55, 66)],
        'learning_rate': float_range(0.2, 0.7, 0.1),
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': [SEED]}

    classifier = GridSearchCV(AdaBoostClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=14)
    classifier.fit(train_feat, train_class)
    return classifier.best_params_

def get_neuralnet_params(train_feat, train_class, *args):
    param_grid = {
        'hidden_layer_sizes': [(i,) for i in range(25, 36)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['adam'],
        'learning_rate': ['constant', 'invscaling'],
        'max_iter': [1000],
        'random_state': [SEED]}

    classifier = GridSearchCV(MLPClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=14)
    classifier.fit(train_feat, train_class)
    return classifier.best_params_

def get_knn_params(train_feat, train_class, *args):
    param_grid = {
        'n_neighbors': [i for i in range(4, 13)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [1, 30, 300], #leaf size does not seem to affect accuracy.  Larger trains slightly faster
        'p': [1, 2],
        'n_jobs': [8]}

    classifier = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_macro', n_jobs=14)
    classifier.fit(train_feat, train_class)
    return classifier.best_params_

def get_svm_params(train_feat, train_class, *args):
    param_grid = {
        'C': float_range(1.0, 1.6, 0.1),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [i for i in range(1, 6)],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced', None],
        'random_state': [SEED]}

    # param_grid = {
    #     'C': [3.43],#[i/100 for i in range(310, 360, 1)],
    #     'kernel': ['sigmoid'],
    #     'gamma': [1/72],#[1/i for i in range(64, 73, 1)],
    #     'coef0': [0.0],
    #     'class_weight': [None],
    #     'random_state': [SEED]}


    classifier = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=14)
    classifier.fit(train_feat, train_class)
    return classifier.best_params_

def float_range(start, stop, step):
    result = []
    while start < stop:
        result.append(start)
        start += step
    return result

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
