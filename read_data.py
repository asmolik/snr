"""Read data from The German Traffic Sign Recognition Benchmark (GTSRB)"""

import os
import const

def load_data(feature, classes):
    """ Returns training and test data for given classes and features set """
    # Read test data
    test_ids, test_labels = read_test_image_labels(
        const.DATA_CSV_PATH, classes=classes)
    test_data = read_test_image_features(
        const.TEST_FEATURE_PATHS[feature],
        labels=dict(zip(test_ids, test_labels)),
        classes=classes)

    # Read training data
    _, train_labels = read_training_image_labels(
        const.DATA_TRAINING_IMG_PATH,
        classes=classes)
    train_data = read_training_image_features(
        const.TRAINING_FEATURES_PATHS[feature],
        classes=classes)

    return train_data, train_labels, test_data, test_labels

def read_training_image_labels(path, classes=range(0, 43)):
    """Reads images' ids and labels of specified classes from a directory.
    Returns list of ids, list of labels."""
    images = []
    labels = []
    # iterate over classes (dirs)
    for dirname in os.listdir(path):
        class_id = int(dirname)
        if class_id not in classes:
            continue
        # iterate over images (files)
        for filename in os.listdir(path + dirname):
            if filename.endswith('.csv'):
                continue
            img_id = filename.partition('.')[0]
            images.append(img_id)
            labels.append(class_id)
    return images, labels

def read_test_image_labels(path, classes=range(0, 43)):
    """Reads file GT-final_test.csv"""
    images = []
    labels = []
    with open(path) as file:
        next(file)
        for line in file:
            tmp = line.split(';')
            if int(tmp[7]) not in classes:
                continue
            images.append(tmp[0].partition('.')[0])
            labels.append(int(tmp[7]))
    return images, labels

def read_training_image_features(path, classes=range(0, 43)):
    """Reads features of specified classes from a directory.
    Returns a matrix where one row is a feature vector."""
    features = []
    # iterate over classes (dirs)
    for dirname in os.listdir(path):
        if int(dirname) not in classes:
            continue
        # iterate over images (files)
        for filename in os.listdir(path + dirname):
            with open(path + '/' + dirname + '/' + filename) as file:
                features.append([float(line) for line in file])
    return features

def read_test_image_features(path, labels, classes=range(0, 43)):
    """Reads HOG features of specified classes from a directory.
    labels - dictionary img_id, class label
    Returns a matrix where one row is a feature vector."""
    features = []
    # iterate over images (files)
    for filename in os.listdir(path):
        # if it isn't in the dictionary don't read it
        if filename.partition('.')[0] not in labels:
            continue
        # if it isn't of specified class don't read it
        if labels[filename.partition('.')[0]] not in classes:
            continue
        with open(path + '/' + filename) as file:
            features.append([float(line) for line in file])
    return features
