#!/usr/bin/python3
import code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from multiprocessing import Process

#local project packages
from dataset_utils import get_normalized_data, filename_se, filename_ho, SEED
from parameter_utils import get_params_se, get_params_ho, get_cached_params_se, get_cached_params_ho, evaluate_params

def main():
    # plot_accuracy_ho() includes a time-based test.  It should be run by itself
    # to reduce the effect of CPU load on the results.  The system should be
    # loaded as little as possible.
    # plot_accuracy_ho()

    # Process(target=get_params_se()).start()
    # Process(target=get_params_ho()).start()
    Process(target=plot_accuracy_se).start()
    Process(target=plot_dtr_accuracy).start()
    Process(target=plot_ada_accuracy).start()
    Process(target=plot_nnt_accuracy).start()
    Process(target=plot_knn_accuracy).start()
    Process(target=plot_svm_accuracy).start()

def plot_svm_accuracy():
    print("plot_svm_accuracy")
    params = [get_cached_params_se(), get_cached_params_ho()]

    ratios = [i/100 for i in range(5, 100, 5)]
    ratios.insert(0, 0.03)
    ratios.insert(0, 0.01)
    ratios.append(0.99)

    files = [filename_se ,filename_ho]
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

            res = evaluate_params(SVC(**params[i]['svm_params_linear']), *data)
            lin_is.append(res[1])
            lin_os.append(res[3])

            res = evaluate_params(SVC(**params[i]['svm_params_poly']), *data)
            pol_is.append(res[1])
            pol_os.append(res[3])

            res = evaluate_params(SVC(**params[i]['svm_params_rbf']), *data)
            rbf_is.append(res[1])
            rbf_os.append(res[3])

            res = evaluate_params(SVC(**params[i]['svm_params_sigmoid']), *data)
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
    ax[1].set(ylabel='Mean F1 Score', title='Holland Dataset')

    plt.xlabel('Train/Test Ratio')
    fig.suptitle('SVM Kernel Performance VS Train/Test Ratio')
    fig.set_size_inches(12, 8)
    plt.savefig("SVM_performance.png")

def plot_knn_accuracy():
    print("plot_knn_accuracy")
    params = [get_cached_params_se(), get_cached_params_ho()]

    data = list()
    data.append(get_normalized_data(filename_se, 0.5))
    data.append(get_normalized_data(filename_ho, 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})

    n_neighbors = [i for i in range(1, 31)]
    for i in range(len(data)):
        knn_is = [] #decision tree in-sample
        knn_os = [] #decision tree out-sample

        for n in n_neighbors:
            knn_params = dict(params[i]['knn_params'])
            knn_params['n_neighbors'] = n

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
    ax[1].set(ylabel='Mean F1 Score', title='Holland Dataset')

    plt.xlabel('K Neighbors')
    fig.suptitle('KNN Performance VS K')
    fig.set_size_inches(12, 8)
    plt.savefig("KNN_performance.png")

def plot_nnt_accuracy():
    print("plot_nnt_accuracy")
    params = [get_cached_params_se(), get_cached_params_ho()]

    data = list()
    data.append(get_normalized_data(filename_se, 0.5))
    data.append(get_normalized_data(filename_ho, 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})

    iterations = [i for i in range(10, 1001, 10)]
    for i in range(len(data)):
        nnt_os = [[], [], []]
        nnts = list()

        #warm_start combined with max_iter lets us train 10 iterations, check performance, and continue training where we left off
        nnt_params = dict(params[i]['nnt_params_adam'])
        nnt_params['max_iter'] = 10
        nnt_params['warm_start'] = True
        nnts.append(MLPClassifier(**nnt_params))

        nnt_params = dict(params[i]['nnt_params_sgd'])
        nnt_params['max_iter'] = 10
        nnt_params['warm_start'] = True
        nnts.append(MLPClassifier(**nnt_params))

        nnt_params = dict(params[i]['nnt_params_lbfgs'])
        nnt_params['max_iter'] = 10
        nnt_params['warm_start'] = True
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
    ax[1].set(ylabel='Mean F1 Score', title='Holland Dataset')

    plt.xlabel('Training Iterations')
    fig.suptitle('Neural Net Solver Performance VS Training Iterations')
    fig.set_size_inches(12, 8)
    plt.savefig("NN_performance.png")

def plot_ada_accuracy():
    print("plot_ada_accuracy")
    params = [get_cached_params_se(), get_cached_params_ho()]

    data = list()
    data.append(get_normalized_data(filename_se, 0.5))
    data.append(get_normalized_data(filename_ho, 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, sharey=True, gridspec_kw={'hspace': 0.12})

    ccps = [i/10000 for i in range(0, 301, 10)]
    for i in range(len(data)):
        ada_is = [] #decision tree in-sample
        ada_os = [] #decision tree out-sample

        for ccp in ccps:
            dtr_params = dict(params[i]['dtr_params'])
            dtr_params['ccp_alpha'] = ccp

            ada_params = dict(params[i]['ada_params'])
            ada_params['base_estimator'] = DecisionTreeClassifier(**dtr_params)

            res = evaluate_params(AdaBoostClassifier(**ada_params), *data[i])
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
    ax[1].set(ylabel='Mean F1 Score', title='Holland Dataset')

    plt.xlabel('CCP Parameter')
    fig.suptitle('AdaBoost Performance VS DT Cost Complexity Pruning Parameter')
    fig.set_size_inches(12, 8)
    plt.savefig("ADA_performance.png")

def plot_dtr_accuracy():
    print("plot_dtr_accuracy")
    params = [get_cached_params_se(), get_cached_params_ho()]

    data = list()
    data.append(get_normalized_data(filename_se, 0.5))
    data.append(get_normalized_data(filename_ho, 0.5))

    fig, ax = plt.subplots(len(data), sharex=True, gridspec_kw={'hspace': 0.12})

    ccps = [i/10000 for i in range(0, 301, 1)]
    for i in range(len(data)):
        dtr_is = [] #decision tree in-sample
        dtr_os = [] #decision tree out-sample

        for ccp in ccps:
            dtr_params = dict(params[i]['dtr_params'])
            dtr_params['ccp_alpha'] = ccp

            res = evaluate_params(DecisionTreeClassifier(**dtr_params), *data[i])
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
    ax[1].set(ylabel='Mean F1 Score', title='Holland Dataset')

    plt.xlabel('CCP Parameter')
    fig.suptitle('Decision Tree Performance VS Cost Complexity Pruning Parameter')
    fig.set_size_inches(12, 8)
    plt.savefig("DT_performance.png")


def plot_accuracy_se():
    print("plot_accuracy_se")
    params = get_cached_params_se()

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

    for ratio in ratios:
        data = get_normalized_data(filename_se, ratio)
        n_samples.append(data[0].shape[0])

        res = evaluate_params(DecisionTreeClassifier(**params['dtr_params']), *data)
        dtr_is.append(res[1])
        dtr_os.append(res[3])

        res = evaluate_params(AdaBoostClassifier(**params['ada_params']), *data)
        ada_is.append(res[1])
        ada_os.append(res[3])

        res = evaluate_params(MLPClassifier(**params['nnt_params_lbfgs']), *data)
        nnt_is.append(res[1])
        nnt_os.append(res[3])

        res = evaluate_params(KNeighborsClassifier(**params['knn_params']), *data)
        knn_is.append(res[1])
        knn_os.append(res[3])

        res = evaluate_params(SVC(**params['svm_params_rbf']), *data)
        svm_is.append(res[1])
        svm_os.append(res[3])

    ### F1 Score plot
    df_temp = pd.DataFrame([ratios, dtr_os, ada_os, nnt_os, knn_os, svm_os, dtr_is, ada_is, nnt_is, knn_is, svm_is])
    df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
    df_temp.rename(columns={1:'DT Out-Of-Sample', 2:'Adaboost Out-Of-Sample', 3:'Neural Net Out-Of_Sample', 4:'KNN Out-Of-Sample', 5:'SVM Out-Of-Sample', 6:'DT In-Sample', 7:'Adaboost In-Sample', 8:'Neural Net In-Sample',  9:'KNN In-Sample', 10:'SVM In-Sample'}, inplace=True)

    fig, ax = plt.subplots()
    df_temp.plot(ax=ax, color=['black', 'blue', 'red', 'orange', 'green', 'black', 'blue', 'red', 'orange', 'green'], style=['-', '-', '-', '-', '-', ':', ':', ':', ':', ':'])
    plt.xlabel('Train/Test Ratio')
    plt.ylabel('Mean F1 Score')
    plt.title('Model Performance VS Train/Test Ratio Over Semeion Dataset')
    plt.legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    plt.grid(axis='both')
    fig.set_size_inches(12, 8)
    plt.savefig("SE_performance.png")

def plot_accuracy_ho():
    print("plot_accuracy_ho")
    params = get_cached_params_ho()

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
        data = get_normalized_data(filename_ho, ratio)
        n_samples.append(data[0].shape[0])

        res = evaluate_params(DecisionTreeClassifier(**params['dtr_params']), *data)
        dtr_is.append(res[1])
        dtr_os.append(res[3])
        dtr_tt.append(res[4])
        dtr_it.append(res[5])

        res = evaluate_params(AdaBoostClassifier(**params['ada_params']), *data)
        ada_is.append(res[1])
        ada_os.append(res[3])
        ada_tt.append(res[4])
        ada_it.append(res[5])

        res = evaluate_params(MLPClassifier(**params['nnt_params_lbfgs']), *data)
        nnt_is.append(res[1])
        nnt_os.append(res[3])
        nnt_tt.append(res[4])
        nnt_it.append(res[5])

        res = evaluate_params(KNeighborsClassifier(**params['knn_params']), *data)
        knn_is.append(res[1])
        knn_os.append(res[3])
        knn_tt.append(res[4])
        knn_it.append(res[5])

        res = evaluate_params(SVC(**params['svm_params_rbf']), *data)
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
    plt.title('Model Performance VS Train/Test Ratio Over Holland Dataset')
    plt.legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    plt.grid(axis='both')
    fig.set_size_inches(12, 8)
    plt.savefig("HO_performance.png")

    ### Train/Infer Time plot
    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0.12})

    ## Train time
    df_temp = pd.DataFrame([n_samples, dtr_tt, ada_tt, nnt_tt, knn_tt, svm_tt])
    df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
    df_temp.rename(columns={1:'DT Train-Time', 2:'Adaboost Train-Time', 3:'Neural Net Train-Time', 4:'KNN Train-Time', 5:'SVM Train-Time'}, inplace=True)

    df_temp.plot(ax=ax[0], color=['black', 'blue', 'red', 'orange', 'green'])
    ax[0].grid(axis='both')
    ax[0].legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    ax[0].set(ylabel='Time (milliseconds)', title='Train Time')

    ### Infer time
    df_temp = pd.DataFrame([n_samples, dtr_it, ada_it, nnt_it, knn_it, svm_it])
    df_temp = df_temp.transpose().set_index(0)#transpose and set column 0 (ratios) as index
    df_temp.rename(columns={1:'DT Infer-Time', 2:'Adaboost Infer-Time', 3:'Neural Net Infer-Time', 4:'KNN Infer-Time', 5:'SVM Infer-Time'}, inplace=True)

    df_temp.plot(ax=ax[1], color=['black', 'blue', 'red', 'orange', 'green'])
    ax[1].grid(axis='both')
    ax[1].legend(loc='best', shadow=True, fontsize='small', facecolor='#d0d0d0')
    ax[1].set(ylabel='Time (milliseconds)', title='Infer Time')

    ## Common
    fig.suptitle('Training & Inference Time VS Sample Size')
    plt.xlabel('Sample Size')
    fig.set_size_inches(12, 8)
    plt.savefig("Time_performance.png")

if "__main__" == __name__:
    main()
