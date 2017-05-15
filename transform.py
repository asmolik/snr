""" Data transformations """

from sklearn import decomposition
from sklearn import preprocessing

def perform_pca(train_data, test_data, components_cnt):
    ''' Performs PCA transform on given training and test  data '''
    pca = decomposition.PCA(n_components=components_cnt)
    pca.fit(train_data)
    train_data_trasformed = pca.transform(train_data)
    test_data_trasformed = pca.transform(test_data)

    return train_data_trasformed, test_data_trasformed

def normalize_data(train_data, test_data):
    ''' Scale features to [0,1] range '''
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data_normalized = min_max_scaler.fit_transform(train_data)
    test_data_normalized = min_max_scaler.transform(test_data)

    # Trim test data to avoid out of range values
    test_data_trimmed = [[min(max(x, 0.0), 1.0) for x in row] for row in test_data_normalized]

    return train_data_normalized, test_data_trimmed
