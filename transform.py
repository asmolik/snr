""" Data transformations """

from sklearn import decomposition
from sklearn import preprocessing
import pca

def perform_pca(train_data, test_data, components_cnt):
    ''' Performs PCA transform on given training and test  data '''
    # pca_trans = decomposition.PCA(n_components=components_cnt)
    # pca_trans.fit(train_data)
    # train_data_trasformed = pca_trans.transform(train_data)
    # test_data_trasformed = pca_trans.transform(test_data)

    mu, trmx = pca.prepare(train_data, components_cnt)
    kernel = pca.linear_kernel
    train_data_trasformed = pca.transform(train_data, mu, trmx, kernel)
    test_data_trasformed = pca.transform(test_data, mu, trmx, kernel)

    return train_data_trasformed, test_data_trasformed

def normalize_data(train_data, test_data):
    ''' Scale features to [0,1] range '''
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_normalized = min_max_scaler.fit_transform(train_data)
    test_data_normalized = min_max_scaler.transform(test_data)

    # Trim test data to avoid out of range values
    test_data_trimmed = [[min(max(x, 0.0), 1.0) for x in row] for row in test_data_normalized]

    return train_data_normalized, test_data_trimmed
