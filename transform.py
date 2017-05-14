""" Data transformations """

from sklearn import decomposition

def perform_pca(train_data, test_data, components_cnt):
    ''' Performs PCA transform on given training and test  data '''
    pca = decomposition.PCA(n_components=components_cnt)
    pca.fit(train_data)
    train_data_trasformed = pca.transform(train_data)
    test_data_trasformed = pca.transform(test_data)

    return train_data_trasformed, test_data_trasformed

def normalize_data(train_data, test_data, norm_func):
    ''' Normalize data to remove negative values '''
    norm = norm_func(train_data)
    train_data_normalized = [norm(row) for row in train_data]
    test_data_normalized = [norm(row) for row in test_data]
    return train_data_normalized, test_data_normalized

def get_minmax_norm(mat):
    ''' Creates minmax norm function for given data '''
    min_x = min([min(row) for row in mat])
    max_x = max([max(row) for row in mat])

    def norm_func(row):
        ''' Minmax norm '''
        res = [min(max((x - min_x)/(max_x - min_x), 0.0), 1.0) for x in row]
        return res
    return norm_func

def get_offset_norm(mat):
    ''' Creates norm function that adds min value offset to each component of row '''
    # Select min value from mat, but not less than 0
    offset = - min(min([min(row) for row in mat]), 0.0)

    def norm_func(row):
        ''' Offset norm '''
        return [max(x + offset, 0.0) for x in row]
    return norm_func
