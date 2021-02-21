#!/usr/bin/python3

import os
import code
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

SEED = 903539276 #GTid random seed for repeatability

filename_se = "datasets/Semeion/semeion.csv" #semeion
filename_ho = "datasets/Letter-Recognition/letter-recognition.data" #holland

data_se = None
data_ho = None

def get_raw_data_se(train_ratio):
    #get the raw Semeion data split into train/test feature vectore/classification
    global data_se

    #Only load the data if it has not been loaded previously
    if type(data_se) == type(None):
        #read data and shuffle
        data_se = pd.read_csv(filename_se, index_col=None, header=None, na_values=['nan'])
        data_se = data_se.sample(frac=1.0).reset_index(drop=True)

        #In the original dataset the label for each instance is indicated by a 1 in columns 256-265, to represent the values 0-9 respectively.  This code fragment reformats this such that the label is represented by 0-9 in column 256.
        for i in range(10):
            data_se.iloc[:,256+i] *= i
        data_se.iloc[:,256] = data_se.iloc[:,256:].sum(axis=1)
        data_se = data_se.iloc[:,:257]#slice off excess

    #split into train and test set
    i_split = int(data_se.shape[0] * train_ratio)
    train_feat = data_se.iloc[:i_split, :-1].to_numpy(dtype=np.float64)
    test_feat = data_se.iloc[i_split:, :-1].to_numpy(dtype=np.float64)
    train_class = data_se.iloc[:i_split, -1].to_numpy(dtype=np.int64)
    test_class = data_se.iloc[i_split:, -1].to_numpy(dtype=np.int64)

    return train_feat, train_class, test_feat, test_class

def get_raw_data_ho(train_ratio):
    #get the raw Holland data split into train/test feature vectore/classification
    global data_ho

    #Only load the data if it has not been loaded previously
    if type(data_ho) == type(None):
        #read data and shuffle
        data_ho = pd.read_csv(filename_ho, na_values=['nan'])
        data_ho = data_ho.sample(frac=1.0).reset_index(drop=True)

        #refactor str-types  to int-types
        replacements = {chr(ord('A') + i):i for i in range(26)}#replace A with 0, B with 1, etc...
        data_ho.replace(replacements, inplace=True)


    #split into train and test set
    i_split = int(data_ho.shape[0] * train_ratio)
    train_feat = data_ho.iloc[:i_split, 1:].to_numpy(dtype=np.float64)
    test_feat = data_ho.iloc[i_split:, 1:].to_numpy(dtype=np.float64)
    train_class = data_ho.iloc[:i_split, 0].to_numpy(dtype=np.int64)
    test_class = data_ho.iloc[i_split:, 0].to_numpy(dtype=np.int64)

    return train_feat, train_class, test_feat, test_class

def get_normalized_data(file_name, train_ratio):
    #get normalized data split into train/test feature vectore/classification
    np.random.seed(SEED)

    if file_name == filename_se:
        train_feat, train_class, test_feat, test_class = get_raw_data_se(train_ratio)

    elif file_name == filename_ho:
        train_feat, train_class, test_feat, test_class = get_raw_data_ho(train_ratio)

    else:
        return None

    #get normalization scaling
    scaler = preprocessing.StandardScaler().fit(train_feat)

    #normalize training and testing feature vectors
    train_feat = scaler.transform(train_feat)
    test_feat = scaler.transform(test_feat)

    #deal with missing values
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_feat)
    train_feat = imp.transform(train_feat)
    test_feat = imp.transform(test_feat)

    return train_feat, train_class, test_feat, test_class
