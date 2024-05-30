import numpy as np
import pandas as pd
import pathlib
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import os
import random

"""
Many of the codes here are taken from the Github repository of the CARD (Han et al., 2022)
https://github.com/XzwHan/CARD/blob/main/regression/data_loader.py
"""


DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.joinpath("dataset/")

def _get_index_train_test_path(data_directory_path, split_num, train=True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return os.path.join(data_directory_path, "index_train_" + str(split_num) + ".txt")
    else:
        return os.path.join(data_directory_path, "index_test_" + str(split_num) + ".txt")

def onehot_encode_cat_feature(X, cat_var_idx_list):
    """
    Apply one-hot encoding to the categorical variable(s) in the feature set,
        specified by the index list.
    """
    # select numerical features
    X_num = np.delete(arr=X, obj=cat_var_idx_list, axis=1)
    # select categorical features
    X_cat = X[:, cat_var_idx_list]
    X_onehot_cat = []
    for col in range(X_cat.shape[1]):
        X_onehot_cat.append(pd.get_dummies(X_cat[:, col], drop_first=True))
    X_onehot_cat = np.concatenate(X_onehot_cat, axis=1).astype(np.float32)
    dim_cat = X_onehot_cat.shape[1]  # number of categorical feature(s)
    X = np.concatenate([X_num, X_onehot_cat], axis=1)
    return X, dim_cat


def preprocess_uci_feature_set(X, data_path, one_hot_encoding=False):
    """
    Obtain preprocessed UCI feature set X (one-hot encoding applied for categorical variable)
        and dimension of one-hot encoded categorical variables.
    """
    dim_cat = 0
    if one_hot_encoding:
        if data_path == 'bostonHousing/data':
            X, dim_cat = onehot_encode_cat_feature(X, [3])
        elif data_path == 'energy/data':
            X, dim_cat = onehot_encode_cat_feature(X, [4, 6, 7])
        elif data_path == 'naval-propulsion-plant/data':
            X, dim_cat = onehot_encode_cat_feature(X, [0, 1, 8, 11])
        else:
            pass
    return X, dim_cat

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class UCI_Dataset(object):
    def __init__(self, data_path, split, normalize_x=True, normalize_y=True):
        # global variables for reading data files
        _DATA_DIRECTORY_PATH = os.path.join(DATA_PATH, data_path)
        _DATA_FILE = os.path.join(_DATA_DIRECTORY_PATH, "data.txt")
        _INDEX_FEATURES_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_features.txt")
        _INDEX_TARGET_FILE = os.path.join(_DATA_DIRECTORY_PATH, "index_target.txt")
        _N_SPLITS_FILE = os.path.join(_DATA_DIRECTORY_PATH, "n_splits.txt")

        # set random seed 1 
        set_random_seed(1)

        # load the data
        data = np.loadtxt(_DATA_FILE)
        # load feature and target indices
        index_features = np.loadtxt(_INDEX_FEATURES_FILE)
        index_target = np.loadtxt(_INDEX_TARGET_FILE)
        # load feature and target as X and y
        X = data[:, [int(i) for i in index_features.tolist()]].astype(np.float32)
        y = data[:, int(index_target.tolist())].astype(np.float32)
        # preprocess feature set X

        X, dim_cat = preprocess_uci_feature_set(X=X, data_path=data_path)
        self.dim_cat = dim_cat

        # load the indices of the train and test sets
        index_train = np.loadtxt(_get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=True))
        index_test = np.loadtxt(_get_index_train_test_path(_DATA_DIRECTORY_PATH, split, train=False))

        # read in data files with indices
        x_train = X[[int(i) for i in index_train.tolist()]]
        y_train = y[[int(i) for i in index_train.tolist()]].reshape(-1, 1)
        x_test = X[[int(i) for i in index_test.tolist()]]
        y_test = y[[int(i) for i in index_test.tolist()]].reshape(-1, 1)

        
        self.x_train = x_train 
        self.y_train = y_train 
        self.x_test = x_test 
        self.y_test = y_test 
        

        self.train_n_samples = x_train.shape[0]
        self.train_dim_x = self.x_train.shape[1]  # dimension of training data input
        self.train_dim_y = self.y_train.shape[1]  # dimension of training regression output

        self.test_n_samples = x_test.shape[0]
        self.test_dim_x = self.x_test.shape[1]  # dimension of testing data input
        self.test_dim_y = self.y_test.shape[1]  # dimension of testing regression output

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.scaler_x, self.scaler_y = None, None

        if self.normalize_x:
            self.normalize_train_test_x()
        if self.normalize_y:
            self.normalize_train_test_y()

    def normalize_train_test_x(self):
        """
        When self.dim_cat > 0, we have one-hot encoded number of categorical variables,
            on which we don't conduct standardization. They are arranged as the last
            columns of the feature set.
        """
        self.scaler_x = StandardScaler(with_mean=True, with_std=True)
        if self.dim_cat == 0:
            self.x_train = self.scaler_x.fit_transform(self.x_train).astype(np.float32)
            self.x_test = self.scaler_x.transform(self.x_test).astype(np.float32)
        else:  
            x_train_num, x_train_cat = self.x_train[:, :-self.dim_cat], self.x_train[:, -self.dim_cat:]
            x_test_num, x_test_cat = self.x_test[:, :-self.dim_cat], self.x_test[:, -self.dim_cat:]
            x_train_num = self.scaler_x.fit_transform(x_train_num).astype(np.float32)
            x_test_num = self.scaler_x.transform(x_test_num).astype(np.float32)
            self.x_train = np.concatenate([x_train_num, x_train_cat], axis=1)
            self.x_test = np.concatenate([x_test_num, x_test_cat], axis=1)

    def normalize_train_test_y(self):
        self.scaler_y = StandardScaler(with_mean=True, with_std=True)
        self.y_train = self.scaler_y.fit_transform(self.y_train).astype(np.float32)
        self.y_test = self.scaler_y.transform(self.y_test).astype(np.float32)

    def return_dataset(self, split="train"):
        if split == "train":
            return self.x_train, self.y_train
        else:
            return self.x_test, self.y_test

    def summary_dataset(self, split="train"):
        if split == "train":
            return {'n_samples': self.train_n_samples, 'dim_x': self.train_dim_x, 'dim_y': self.train_dim_y}
        else:
            return {'n_samples': self.test_n_samples, 'dim_x': self.test_dim_x, 'dim_y': self.test_dim_y}
        

def generate_uci(num_seeds: int, data_path: str):
    UCI = UCI_Dataset(data_path, split=0)
    X_ex, y_ex=UCI.return_dataset(split = 'train')
    Xp_ex, yp_ex=UCI.return_dataset(split = 'test')

    vX=np.zeros((num_seeds, X_ex.shape[0], X_ex.shape[1]))
    vy=np.zeros((num_seeds, y_ex.shape[0], y_ex.shape[1]))
    vXp = np.zeros((num_seeds, Xp_ex.shape[0], Xp_ex.shape[1]))
    vyp = np.zeros((num_seeds, yp_ex.shape[0], yp_ex.shape[1]))
    vUCI = []
    
    for i in range(num_seeds):
        UCI = UCI_Dataset(data_path, split=i)
        X, y=UCI.return_dataset(split = 'train')
        Xp, yp=UCI.return_dataset(split = 'test')
        vX[i] = X
        vy[i] = y
        vXp[i] = Xp
        vyp[i] = yp
        vUCI.append(UCI)

    return vX, vy, vXp, vyp, vUCI
