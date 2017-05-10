"""Read data from The German Traffic Sign Recognition Benchmark (GTSRB)"""

import os


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


def read_training_image_hues(path, classes=range(0, 43)):
    """Reads hues of images of specified classes from a directory.
    Returns a matrix where one row is a hue vector of an image."""
    hues = []
    # iterate over classes (dirs)
    for dirname in os.listdir(path):
        if int(dirname) not in classes:
            continue
        # iterate over images (files)
        for filename in os.listdir(path + dirname):
            with open(path + '/' + dirname + '/' + filename) as file:
                hues.append([float(line) for line in file])
    return hues


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


def read_test_image_hues(path, labels, classes=range(0, 43)):
    """Reads hues of images of specified classes from a directory.
    labels - dictionary img_id, class label
    Returns a matrix where one row is a hue vector of an image."""
    hues = []
    # iterate over images (files)
    for filename in os.listdir(path):
        # if it isn't in the dictionary don't read it
        if filename.partition('.')[0] not in labels:
            continue
        # if it isn't of specified class don't read it
        if labels[filename.partition('.')[0]] not in classes:
            continue
        with open(path + '/' + filename) as file:
            hues.append([float(line) for line in file])
    return hues


def read_training_image_hog(path,
                            columns=range(0, 256),
                            classes=range(0, 43)):
    """Reads HOG features of specified classes from a directory.
    columns - features to read
    Returns a matrix where one row is a feature vector."""
    features = []
    # iterate over classes (dirs)
    for dirname in os.listdir(path):
        if int(dirname) not in classes:
            continue
        # iterate over images (files)
        for filename in os.listdir(path + dirname):
            with open(path + '/' + dirname + '/' + filename) as file:
                features.append([float(line) for i, line in enumerate(file) if i in columns])
    return features


def read_test_image_hog(path,
                        labels,
                        columns=range(0, 256),
                        classes=range(0, 43)):
    """Reads HOG features of specified classes from a directory.
    columns - features to read
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
            features.append([float(line) for i, line in enumerate(file) if i in columns])
    return features
