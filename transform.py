""" Data transformations """

from functools import partial
from sklearn import decomposition
from sklearn import preprocessing
import sklearn.metrics.pairwise as metrics
import pca

def perform_pca(train_data, test_data, components_cnt):
    ''' Performs PCA transform on given training and test  data '''
    pca_trans = decomposition.PCA(n_components=components_cnt)
    pca_trans.fit(train_data)
    train_data_transformed = pca_trans.transform(train_data)
    test_data_transformed = pca_trans.transform(test_data)

    return train_data_transformed, test_data_transformed


def perform_kernel_pca(train_data, test_data, components_cnt, kernel_params):
    ''' Performs kernel PCA transform on given training and test  data '''
    kernel, gamma, degree, coef0 = kernel_params
    if kernel == 'chi2':
        k = partial(metrics.chi2_kernel, gamma=gamma)
        m, t = pca.prepare(train_data, components_cnt)
        train_data_transformed = pca.transform(train_data, m, t, k)
        test_data_transformed = pca.transform(test_data, m, t, k)
    else:
        pca_trans = decomposition.KernelPCA(n_components=components_cnt, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        pca_trans.fit(train_data)
        train_data_transformed = pca_trans.transform(train_data)
        test_data_transformed = pca_trans.transform(test_data)

    return train_data_transformed, test_data_transformed

def normalize_data(train_data, test_data):
    ''' Scale features to [0,1] range '''
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_normalized = min_max_scaler.fit_transform(train_data)
    test_data_normalized = min_max_scaler.transform(test_data)

    # Trim test data to avoid out of range values
    test_data_trimmed = [[min(max(x, 0.0), 1.0) for x in row] for row in test_data_normalized]

    return train_data_normalized, test_data_trimmed
