#!/usr/bin/python3
import code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from dataset_utils import get_normalized_data
from get_parameters import get_dt_params
from get_parameters import get_adaboost_params
from get_parameters import get_neuralnet_params
from get_parameters import get_knn_params
from get_parameters import get_svm_params
from get_parameters import evaluate_params
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

from time import time


def main():
    get_dataset1_best_params()
    # get_dataset2_best_params()
    # plot_accuracy_vs_training_ratio_dataset1()
    # plot_accuracy_vs_training_ratio_dataset2()

    # plot_dt_accuracy_vs_ccp()
    # plot_ada_accuracy_vs_ccp()
    # plot_nnt_accuracy_vs_iter()
    # plot_knn_accuracy_vs_k()
    # plot_SVM_accuracy_vs_kernel()

def plot_SVM_accuracy_vs_kernel():
    svm_params = {
        'C': 1.5,
        'class_weight': 'balanced',
        'degree': 1,
        'gamma': 'scale',
        'random_state': 903539276}

    ratios = [i/100 for i in range(5, 100, 5)]
    ratios.insert(0, 0.03)
    ratios.insert(0, 0.01)
    ratios.append(0.99)

    files = ["datasets/Semeion/semeion.csv" ,"datasets/Letter-Recognition/letter-recognition.data"]
    fig, ax = plt.subplots(len(files), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})
    for i in range(len(files)):
        n_samples = []
        lin_is = [] #linear kernel in-sample
        pol_is = [] #etc...
        rbf_is = []
        sig_is = []
        lin_os = [] #linear kernel out-sample
        pol_os = [] #etc...
        rbf_os = []
        sig_os = []

        for ratio in ratios:
            data = get_normalized_data(files[i], ratio)
            n_samples.append(data[0].shape[0])

            res = evaluate_params(svm.SVC(**svm_params, kernel='linear'), *data)
            lin_is.append(res[1])
            lin_os.append(res[3])

            res = evaluate_params(svm.SVC(**svm_params, kernel='poly'), *data)
            pol_is.append(res[1])
            pol_os.append(res[3])

            res = evaluate_params(svm.SVC(**svm_params, kernel='rbf'), *data)
            rbf_is.append(res[1])
            rbf_os.append(res[3])

            res = evaluate_params(svm.SVC(C=3.43, kernel='sigmoid', gamma=1/72, class_weight=None, random_state=903539276), *data)
            sig_is.append(res[1])
            sig_os.append(res[3])

        ### F1 Score plot
        df_temp = pd.DataFrame([ratios, lin_os, pol_os, rbf_os, sig_os, lin_is, pol_is, rbf_is, sig_is])
        df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
        df_temp.rename(columns={1:'Linear Out-Of-Sample', 2:'Poly Out-Of-Sample', 3:'RBF Out-Of_Sample', 4:'Sigmoid Out-Of-Sample', 5:'Linear In-Sample', 6:'Poly In-Sample', 7:'RBF In-Sample', 8:'Sigmoid In-Sample'}, inplace=True)

        df_temp.plot(ax=ax[i], color=['black', 'blue', 'red', 'orange', 'black', 'blue', 'red', 'orange'], style=['-', '-', '-', '-', ':', ':', ':', ':'])
        ax[i].grid(axis='both')
        ax[i].legend(loc='lower right', shadow=True, fontsize='small', facecolor='#d0d0d0')

    ax[0].set(ylabel='Mean F1 Score', title='Semeion Dataset')
    ax[1].set(ylabel='Mean F1 Score', title='Letter Recognition Dataset')

    plt.xlabel('Train/Test Ratio')
    fig.suptitle('SVM Kernel Performance VS Train/Test Ratio')
    fig.set_size_inches(12, 8)
    plt.savefig("SVM_performance.png")

def plot_knn_accuracy_vs_k():
    data = list()
    data.append(get_normalized_data("datasets/Semeion/semeion.csv", 0.5))
    data.append(get_normalized_data("datasets/Letter-Recognition/letter-recognition.data", 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})

    n_neighbors = [i for i in range(1, 31)]
    for i in range(len(data)):
        knn_is = [] #decision tree in-sample
        knn_os = [] #decision tree out-sample

        for n in n_neighbors:
            knn_params = {
                'algorithm': 'ball_tree',
                'leaf_size': 30,
                'n_jobs': 8,
                'n_neighbors': n,
                'p': 1,
                'weights': 'distance'}

            res = evaluate_params(KNeighborsClassifier(**knn_params), *data[i])
            knn_is.append(res[1])
            knn_os.append(res[3])

        ### F1 Score plot
        df_temp = pd.DataFrame([n_neighbors, knn_os, knn_is])
        df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
        df_temp.rename(columns={1:'KNN Out-Of-Sample', 2:'KNN In-Sample'}, inplace=True)

        df_temp.plot(ax=ax[i], color=['black', 'black'], style=['-', ':'])
        ax[i].grid(axis='both')
        ax[i].legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')

    ax[0].set(ylabel='Mean F1 Score', title='Semeion Dataset')
    ax[1].set(ylabel='Mean F1 Score', title='Letter Recognition Dataset')

    plt.xlabel('K Neighbors')
    fig.suptitle('KNN Performance VS K')
    fig.set_size_inches(12, 8)
    plt.savefig("KNN_performance.png")

def plot_nnt_accuracy_vs_iter():
    nnt_params = {
        'activation': 'logistic',
        'hidden_layer_sizes': (30,),
        'learning_rate': 'constant',
        'max_iter': 10,
        'random_state': 903539276,
        'warm_start': True} #warm_start combined with max_iter lets us train 10 iterations, check performance, and continue training where we left off


    data = list()
    data.append(get_normalized_data("datasets/Semeion/semeion.csv", 0.5))
    data.append(get_normalized_data("datasets/Letter-Recognition/letter-recognition.data", 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})

    iterations = [i for i in range(10, 1001, 10)]
    for i in range(len(data)):
        nnt_os = [[], [], []] #neural net w/ adam solver out-sample
        nnts = list()
        nnt_params['solver'] = 'adam'
        nnts.append(MLPClassifier(**nnt_params))
        nnt_params['solver'] = 'sgd'
        nnts.append(MLPClassifier(**nnt_params))
        nnt_params['solver'] = 'lbfgs'
        nnts.append(MLPClassifier(**nnt_params))

        for _ in range (100):

            for j in range(len(nnts)):
                nnts[j].fit(data[i][0], data[i][1])
                test_pred = nnts[j].predict(data[i][2])
                test_score = f1_score(data[i][3], test_pred, average=None)
                test_score_mean = test_score.mean()
                nnt_os[j].append(test_score_mean)

        ### F1 Score plot
        df_temp = pd.DataFrame([iterations, nnt_os[0], nnt_os[1], nnt_os[2]])
        df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
        df_temp.rename(columns={1:'adam', 2:'sgd', 3:'lbfgs'}, inplace=True)

        df_temp.plot(ax=ax[i], color=['black', 'blue', 'red'], style=['-', '-', '-'])
        ax[i].grid(axis='both')
        ax[i].legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')

    ax[0].set(ylabel='Mean F1 Score', title='Semeion Dataset')
    ax[1].set(ylabel='Mean F1 Score', title='Letter Recognition Dataset')

    plt.xlabel('Training Iterations')
    fig.suptitle('Neural Net Solver Performance VS Training Iterations')
    fig.set_size_inches(12, 8)
    plt.savefig("NN_performance.png")

def plot_ada_accuracy_vs_ccp():
    data = list()
    data.append(get_normalized_data("datasets/Semeion/semeion.csv", 0.5))
    data.append(get_normalized_data("datasets/Letter-Recognition/letter-recognition.data", 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})

    ccps = [i/10000 for i in range(0, 501, 1)]
    for i in range(len(data)):
        ada_is = [] #decision tree in-sample
        ada_os = [] #decision tree out-sample

        for ccp in ccps:
            dt_params = {
                'ccp_alpha': ccp,
                'class_weight': 'balanced',
                'criterion': 'entropy',
                'random_state': 903539276}

            adaboost_params = {
                'algorithm': 'SAMME.R',
                'base_estimator': DecisionTreeClassifier(**dt_params),
                'learning_rate': 0.6,
                'n_estimators': 59,
                'random_state': 903539276}

            res = evaluate_params(AdaBoostClassifier(**adaboost_params), *data[i])
            ada_is.append(res[1])
            ada_os.append(res[3])

        ### F1 Score plot
        df_temp = pd.DataFrame([ccps, ada_os, ada_is])
        df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
        df_temp.rename(columns={1:'AdaBoost Out-Of-Sample', 2:'AdaBoost In-Sample'}, inplace=True)

        df_temp.plot(ax=ax[i], color=['black', 'black'], style=['-', ':'])
        ax[i].grid(axis='both')
        ax[i].legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')

    ax[0].set(ylabel='Mean F1 Score', title='Semeion Dataset')
    ax[1].set(ylabel='Mean F1 Score', title='Letter Recognition Dataset')

    plt.xlabel('CCP Parameter')
    fig.suptitle('AdaBoost Performance VS DT Cost Complexity Pruning Parameter')
    fig.set_size_inches(12, 8)
    plt.savefig("ADA_performance.png")

def plot_dt_accuracy_vs_ccp():
    data = list()
    data.append(get_normalized_data("datasets/Semeion/semeion.csv", 0.5))
    data.append(get_normalized_data("datasets/Letter-Recognition/letter-recognition.data", 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, gridspec_kw={'hspace': 0.12})

    ccps = [i/10000 for i in range(0, 501, 1)]
    for i in range(len(data)):
        dtr_is = [] #decision tree in-sample
        dtr_os = [] #decision tree out-sample

        for ccp in ccps:
            dt_params = {
                'ccp_alpha': ccp,
                'class_weight': 'balanced',
                'criterion': 'entropy',
                'random_state': 903539276}

            res = evaluate_params(DecisionTreeClassifier(**dt_params), *data[i])
            dtr_is.append(res[1])
            dtr_os.append(res[3])



        ### F1 Score plot
        df_temp = pd.DataFrame([ccps, dtr_os, dtr_is])
        df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
        df_temp.rename(columns={1:'DT Out-Of-Sample', 2:'DT In-Sample'}, inplace=True)

        df_temp.plot(ax=ax[i], color=['black', 'black'], style=['-', ':'])
        ax[i].grid(axis='both')
        ax[i].legend(loc='lower left', shadow=True, fontsize='small', facecolor='#d0d0d0')

    ax[0].set(ylabel='Mean F1 Score', title='Semeion Dataset')
    ax[1].set(ylabel='Mean F1 Score', title='Letter Recognition Dataset')

    plt.xlabel('CCP Parameter')
    fig.suptitle('Decision Tree Performance VS Cost Complexity Pruning Parameter')
    fig.set_size_inches(12, 8)
    plt.savefig("DT_performance.png")


def plot_accuracy_vs_training_ratio_dataset1():
    pass

def plot_accuracy_vs_training_ratio_dataset2():
    ### dataset1 with 0.5 train/test ratio
    dt_params = {
        'ccp_alpha': 0.0,
        'class_weight': 'balanced',
        'criterion': 'entropy',
        'random_state': 903539276}

    adaboost_params = {
        'algorithm': 'SAMME.R',
        'base_estimator': DecisionTreeClassifier(ccp_alpha=0.02, class_weight='balanced', criterion='entropy', random_state=903539276),
        'learning_rate': 0.6,
        'n_estimators': 59,
        'random_state': 903539276}

    neuralnet_params = {
        'activation': 'logistic',
        'hidden_layer_sizes': (30,),
        'learning_rate': 'constant',
        'max_iter': 1000,
        'random_state': 903539276,
        'solver': 'adam'}

    knn_params = {
        'algorithm': 'ball_tree',
        'leaf_size': 1,
        'n_jobs': 8,
        'n_neighbors': 6,
        'p': 1,
        'weights': 'distance'}

    svm_params = {
        'C': 1.5,
        'class_weight': 'balanced',
        'degree': 1,
        'gamma': 'scale',
        'kernel': 'rbf',
        'random_state': 903539276}

    #In-Sample performance was 1.0 for all except Decision Tree.  It started close to 1.0 and and declined past 0.8 as the train ratio increased
    ratios = [i/100 for i in range(5, 100, 5)]
    ratios.insert(0, 0.01)
    ratios.append(0.99)
    # ratios = [0.01, 0.2]
    n_samples = []
    dtr_is = [] #decision tree in-sample
    ada_is = [] #etc...
    nnt_is = []
    knn_is = []
    svm_is = []
    dtr_os = [] #decision tree out-sample
    ada_os = [] #etc...
    nnt_os = []
    knn_os = []
    svm_os = []
    dtr_tt = [] #decision tree train time
    ada_tt = [] #etc...
    nnt_tt = []
    knn_tt = []
    svm_tt = []
    dtr_it = [] #decision tree infer time
    ada_it = [] #etc...
    nnt_it = []
    knn_it = []
    svm_it = []

    for ratio in ratios:
        data = get_normalized_data("datasets/Letter-Recognition/letter-recognition.data", ratio)
        n_samples.append(data[0].shape[0])

        res = evaluate_params(DecisionTreeClassifier(**dt_params), *data)
        dtr_is.append(res[1])
        dtr_os.append(res[3])
        dtr_tt.append(res[4])
        dtr_it.append(res[5])

        res = evaluate_params(AdaBoostClassifier(**adaboost_params), *data)
        ada_is.append(res[1])
        ada_os.append(res[3])
        ada_tt.append(res[4])
        ada_it.append(res[5])

        res = evaluate_params(MLPClassifier(**neuralnet_params), *data)
        nnt_is.append(res[1])
        nnt_os.append(res[3])
        nnt_tt.append(res[4])
        nnt_it.append(res[5])

        res = evaluate_params(KNeighborsClassifier(**knn_params), *data)
        knn_is.append(res[1])
        knn_os.append(res[3])
        knn_tt.append(res[4])
        knn_it.append(res[5])

        res = evaluate_params(svm.SVC(**svm_params), *data)
        svm_is.append(res[1])
        svm_os.append(res[3])
        svm_tt.append(res[4])
        svm_it.append(res[5])

    ### F1 Score plot
    df_temp = pd.DataFrame([ratios, dtr_os, ada_os, nnt_os, knn_os, svm_os, dtr_is, ada_is, nnt_is, knn_is, svm_is])
    df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
    df_temp.rename(columns={1:'DT Out-Of-Sample', 2:'Adaboost Out-Of-Sample', 3:'Neural Net Out-Of_Sample', 4:'KNN Out-Of-Sample', 5:'SVM Out-Of-Sample', 6:'DT In-Sample', 7:'Adaboost In-Sample', 8:'Neural Net In-Sample',  9:'KNN In-Sample', 10:'SVM In-Sample'}, inplace=True)

    fig, ax = plt.subplots()
    df_temp.plot(ax=ax, color=['black', 'blue', 'red', 'orange', 'green', 'black', 'blue', 'red', 'orange', 'green'], style=['-', '-', '-', '-', '-', ':', ':', ':', ':', ':'])
    plt.xlabel('Train/Test Ratio')
    plt.ylabel('Mean F1 Score')
    plt.title('"Letter Recognition Data Set" Performance VS Train/Test Ratio')
    plt.legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    plt.grid(axis='both')
    fig.set_size_inches(12, 8)
    plt.savefig("LR_performance.png")

    ### Train/Infer Time plot
    df_temp = pd.DataFrame([n_samples, dtr_tt, ada_tt, nnt_tt, knn_tt, svm_tt, dtr_it, ada_it, nnt_it, knn_it, svm_it])
    df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
    df_temp.rename(columns={1:'DT Train-Time', 2:'Adaboost Train-Time', 3:'Neural Net Train-Time', 4:'KNN Train-Time', 5:'SVM Train-Time', 6:'DT Infer-Time', 7:'Adaboost Infer-Time', 8:'Neural Net Infer-Time', 9:'KNN Infer-Time', 10:'SVM Infer-Time'}, inplace=True)

    fig, ax = plt.subplots()
    df_temp.plot(ax=ax, color=['black', 'blue', 'red', 'orange', 'green', 'black', 'blue', 'red', 'orange', 'green'], style=['-', '-', '-', '-', '-', ':', ':', ':', ':', ':'])
    plt.xlabel('Sample Size')
    plt.ylabel('Time in milliseconds')
    plt.title('"Letter Recognition Data Set" Training & Inference Time VS Sample Size')
    plt.legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    plt.grid(axis='both')
    fig.set_size_inches(12, 8)
    plt.savefig("LR_train_infer_time.png")

    ### Infer-Only Time plot
    df_temp = pd.DataFrame([n_samples, dtr_it, ada_it, nnt_it, knn_it, svm_it])
    df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
    df_temp.rename(columns={1:'DT Infer-Time', 2:'Adaboost Infer-Time', 3:'Neural Net Infer-Time', 4:'KNN Infer-Time', 5:'SVM Infer-Time'}, inplace=True)

    fig, ax = plt.subplots()
    df_temp.plot(ax=ax, color=['black', 'blue', 'red', 'orange', 'green'], style=[':', ':', ':', ':', ':'])
    plt.xlabel('Sample Size')
    plt.ylabel('Time in milliseconds')
    plt.title('"Letter Recognition Data Set" Inference Time VS Sample Size')
    plt.legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    plt.grid(axis='both')
    fig.set_size_inches(12, 8)
    plt.savefig("LR_infer_time.png")

def get_dataset1_best_params():
    data = get_normalized_data("datasets/Semeion/semeion.csv", 0.5)

    ### DecisionTreeClassifier
    dt_params = get_dt_params(*data)
    print("DecisionTreeClassifier best parameters:  " + str(dt_params))
    evaluate_params(DecisionTreeClassifier(**dt_params), *data, True)

    ### AdaBoostClassifier with more pruned Decision Tree learner
    adaboost_params = get_adaboost_params(*data)
    print("AdaBoostClassifier best parameters:  " + str(adaboost_params))
    evaluate_params(AdaBoostClassifier(**adaboost_params), *data, True)

    ### MLPClassifier
    neuralnet_params = get_neuralnet_params(*data)
    print("MLPClassifier best parameters:  " + str(neuralnet_params))
    evaluate_params(MLPClassifier(**neuralnet_params), *data, True)

    ### KNeighborsClassifier
    knn_params = get_knn_params(*data)
    print("KNeighborsClassifier best parameters:  " + str(knn_params))
    evaluate_params(KNeighborsClassifier(**knn_params), *data, True)

    ### KNeighborsClassifier
    svm_params = get_svm_params(*data)
    print("svm.SVC() best parameters:  " + str(svm_params))
    evaluate_params(svm.SVC(**svm_params), *data, True)

def get_dataset2_best_params():
    data = get_normalized_data("datasets/Letter-Recognition/letter-recognition.data", 0.5)

    ### DecisionTreeClassifier
    dt_params = get_dt_params(*data)
    print("DecisionTreeClassifier best parameters:  " + str(dt_params))
    evaluate_params(DecisionTreeClassifier(**dt_params), *data, True)

    ### AdaBoostClassifier with more pruned Decision Tree learner
    adaboost_params = get_adaboost_params(*data, dt_params)
    print("AdaBoostClassifier best parameters:  " + str(adaboost_params))
    evaluate_params(AdaBoostClassifier(**adaboost_params), *data, True)

    ### MLPClassifier
    neuralnet_params = get_neuralnet_params(*data)
    print("MLPClassifier best parameters:  " + str(neuralnet_params))
    evaluate_params(MLPClassifier(**neuralnet_params), *data, True)

    ### KNeighborsClassifier
    knn_params = get_knn_params(*data)
    print("KNeighborsClassifier best parameters:  " + str(knn_params))
    evaluate_params(KNeighborsClassifier(**knn_params), *data, True)

    ### KNeighborsClassifier
    svm_params = get_svm_params(*data)
    print("svm.SVC() best parameters:  " + str(svm_params))
    evaluate_params(svm.SVC(**svm_params), *data, True)


if "__main__" == __name__:
    main()
