#!/usr/bin/python3

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

import code #note to self, remove after debugging complete

SEED = 903539276 #GTid random seed for repeatability

filename1 = "datasets/Semeion/semeion.csv"
filename2 = "datasets/Letter-Recognition/letter-recognition.data"

data1 = None
data2 = None

def get_raw_data1(train_ratio):
    global data1

    #Only load the data if it has not been loaded previously
    if type(data1) == type(None):
        #read data and shuffle
        data1 = pd.read_csv(filename1, index_col=None, header=None, na_values=['nan'])
        data1 = data1.sample(frac=1.0).reset_index(drop=True)

        #In the original dataset the label for each instance is indicated by a 1 in columns 256-265, to represent the values 0-9 respectively.  This code fragment reformats this such that the label is represented by 0-9 in column 256.
        for i in range(10):
            data1.iloc[:,256+i] *= i
        data1.iloc[:,256] = data1.iloc[:,256:].sum(axis=1)
        data1 = data1.iloc[:,:257]#slice off excess


        data=data1
        code.interact(local=locals())
        a=data.iloc[:, :256].sum(axis=1)
        a.min#47
        a.max()#151
        a.mean()#84.12806026365348
        a.std()#15.714792260649231

        collections.Counter(a)#Counter({73: 46, 86: 42, 88: 41, 84: 41, 81: 41, 80: 41, 74: 40, 79: 39, 77: 39, 76: 39, 91: 38, 82: 37, 75: 37, 90: 35, 83: 34, 78: 34, 85: 34, 67: 34, 69: 33, 97: 33, 87: 33, 71: 32, 92: 31, 89: 31, 72: 31, 70: 30, 95: 30, 66: 28, 99: 27, 65: 26, 94: 24, 93: 24, 100: 24, 63: 23, 68: 22, 62: 22, 101: 22, 64: 21, 96: 21, 104: 21, 98: 20, 102: 20, 103: 18, 59: 16, 108: 16, 105: 14, 60: 14, 109: 13, 110: 11, 116: 11, 106: 11, 112: 11, 58: 10, 107: 10, 61: 10, 114: 9, 115: 9, 56: 8, 117: 8, 111: 7, 118: 6, 113: 5, 53: 5, 52: 5, 122: 4, 55: 4, 57: 4, 54: 3, 132: 3, 49: 3, 127: 3, 129: 2, 120: 2, 51: 2, 47: 2, 124: 2, 139: 1, 121: 1, 128: 1, 123: 1, 50: 1, 141: 1, 126: 1, 125: 1, 134: 1, 119: 1, 151: 1})
        # {47: 2, 49: 3, 50: 1, 51: 2, 52: 5, 53: 5, 54: 3, 55: 4, 56: 8, 57: 4, 58: 10, 59: 16, 60: 14, 61: 10, 62: 22, 63: 23, 64: 21, 65: 26, 66: 28, 67: 34, 68: 22, 69: 33, 70: 30, 71: 32, 72: 31, 73: 46, 74: 40, 75: 37, 76: 39, 77: 39, 78: 34, 79: 39, 80: 41, 81: 41, 82: 37, 83: 34, 84: 41, 85: 34, 86: 42, 87: 33, 88: 41, 89: 31, 90: 35, 91: 38, 92: 31, 93: 24, 94: 24, 95: 30, 96: 21, 97: 33, 98: 20, 99: 27, 100: 24, 101: 22, 102: 20, 103: 18, 104: 21, 105: 14, 106: 11, 107: 10, 108: 16, 109: 13, 110: 11, 111: 7, 112: 11, 113: 5, 114: 9, 115: 9, 116: 11, 117: 8, 118: 6, 119: 1, 120: 2, 121: 1, 122: 4, 123: 1, 124: 2, 125: 1, 126: 1, 127: 3, 128: 1, 129: 2, 132: 3, 134: 1, 139: 1, 141: 1, 151: 1}



    #split into train and test set
    i_split = int(data1.shape[0] * train_ratio)
    train_feat = data1.iloc[:i_split, :-1].to_numpy(dtype=np.float64)
    test_feat = data1.iloc[i_split:, :-1].to_numpy(dtype=np.float64)
    train_class = data1.iloc[:i_split, -1].to_numpy(dtype=np.int64)
    test_class = data1.iloc[i_split:, -1].to_numpy(dtype=np.int64)

    return train_feat, train_class, test_feat, test_class

def get_raw_data2(train_ratio):
    global data2

    #Only load the data if it has not been loaded previously
    if type(data2) == type(None):
        #read data and shuffle
        data2 = pd.read_csv(filename2, na_values=['nan'])
        data2 = data2.sample(frac=1.0).reset_index(drop=True)

        #refactor str-types  to int-types
        replacements = {chr(ord('A') + i):i for i in range(26)}#replace A with 0, B with 1, etc...
        data2.replace(replacements, inplace=True)


    #split into train and test set
    i_split = int(data2.shape[0] * train_ratio)
    train_feat = data2.iloc[:i_split, 1:].to_numpy(dtype=np.float64)
    test_feat = data2.iloc[i_split:, 1:].to_numpy(dtype=np.float64)
    train_class = data2.iloc[:i_split, 0].to_numpy(dtype=np.int64)
    test_class = data2.iloc[i_split:, 0].to_numpy(dtype=np.int64)

    return train_feat, train_class, test_feat, test_class

def get_normalized_data(file_name, train_ratio):
    np.random.seed(SEED)

    if file_name == filename1:
        train_feat, train_class, test_feat, test_class = get_raw_data1(train_ratio)

    elif file_name == filename2:
        train_feat, train_class, test_feat, test_class = get_raw_data2(train_ratio)

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
