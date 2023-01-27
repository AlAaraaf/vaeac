import sys
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd


mcar_prob = 0.5
random_seed = int(sys.argv[1])


def yeast_loader(path):
    # read and preprocess yeast dataset
    data = [[y for y in x.split(' ') if y][1:]
            for x in open(join(path, 'yeast.data')).read().split('\n') if x]
    target_id = {x: i for i, x in enumerate(set([x[-1] for x in data]))}
    data = [x[:-1] + [target_id[x[-1]]] for x in data]
    data = [[float(y) for y in x] for x in data]
    return np.array(data)


def white_loader(path):
    # read and preprocess white-wine dataset
    data = pd.read_csv(join(path, 'winequality-white.csv'), sep=';')
    return np.array(data)

def boston_loader(path):
    # read boston housing dataset
    data = pd.read_csv(join(path, 'boston.csv'))
    return np.array(data)

def acs_loader(path):
    # read ACS dataset
    # randomly select 10000 samples from the dataset for training and testing
    raw_data = pd.read_csv('../../MissingData_DL/data/house.csv')
    raw_data = raw_data.values.astype(np.float32)
    total_index = np.random.permutation(raw_data.shape[0])
    sample_index = total_index[:10000]
    data = raw_data[sample_index,:]
    return np.array(data)

def credit_loader(path):
    # read credit dataset
    data = pd.read_csv(join(path, 'credit.csv'))
    return np.array(data)


def mushroom_loader(path):
    # read and preprocess mushroom dataset
    data = pd.read_csv(join(path, 'agaricus-lepiota.data'),
                       header=None, na_values='?')
    target = np.array(data[0] == 'e')
    data.drop(0, axis=1, inplace=True)
    data.drop(16, axis=1, inplace=True)
    categorical_sizes = []
    mtx = []
    for column_name in data.columns:
        column = np.array(pd.get_dummies(data[column_name])).astype('float')
        categorical_sizes.append(column.shape[1])
        column = column.dot(np.arange(column.shape[1]).reshape(-1, 1))
        column[np.array(data[column_name].isnull()), :] = np.nan
        mtx.append(column)
    data = np.hstack(mtx + [target.reshape(-1, 1)])
    categorical_sizes += [2]
    return data


def corrupt_data_mcar(data):
    # return a copy of data with missing values with density mcar_prob
    mask = np.random.choice(2, size=data.shape, p=[mcar_prob, 1 - mcar_prob])
    nw_data = data.copy()
    nw_data[(1 - mask).astype('bool')] = np.nan
    return nw_data


def save_data(filename, data):
    np.savetxt(filename, data, delimiter='\t')


loader_dict = {
    'yeast':yeast_loader,
    'white':white_loader,
    'mushroom':mushroom_loader,
    'boston':boston_loader,
    'acs':acs_loader,
    'credit':credit_loader
}

name = sys.argv[2]
loader = loader_dict[name]
np.random.seed(random_seed)
data = loader('original_data')
train_data = corrupt_data_mcar(data)

makedirs('train_test_split', exist_ok=True)
save_data(join('train_test_split', '{}_train.tsv'.format(name)),
            train_data)
save_data(join('train_test_split', '{}_groundtruth.tsv'.format(name)),
            data)