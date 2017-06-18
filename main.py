""" Main module """

import time
import read_data as d
import transform
import const
import log
import chi2
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(feature, classes, components_cnt):
    log.line()
    log.dataset_configuration(feature, classes, components_cnt)
    log.message('Loading data...')

    start = time.time()
    train_data, train_labels, test_data, test_labels = d.read_test_and_train_data(feature, classes)
    log.time(time.time() - start)

    log.dataset_summary(len(train_data), len(test_data))
    return train_data, train_labels, test_data, test_labels

def transform_data(data, components_cnt):
    train_data, train_labels, test_data, test_labels = data

    log.line()
    start = time.time()
    log.message('pca...')
    train_data, test_data = transform.perform_pca(train_data, test_data, components_cnt)
    log.time(time.time() - start)
    start = time.time()
    log.message('normalize...')
    train_data, test_data = transform.normalize_data(train_data, test_data)
    log.time(time.time() - start)

    log.dataset_summary(len(train_data), len(test_data))
    return train_data, train_labels, test_data, test_labels


def transform_data_kernel(data, components_cnt, kernel_params):
    train_data, train_labels, test_data, test_labels = data

    log.line()
    start = time.time()
    log.message('pca...')
    if kernel_params[0] == 'chi2':
        train_data, test_data = transform.normalize_data(train_data, test_data)
    train_data, test_data = transform.perform_kernel_pca(train_data, test_data, components_cnt, kernel_params)
    log.time(time.time() - start)
    start = time.time()
    log.message('normalize...')
    train_data, test_data = transform.normalize_data(train_data, test_data)
    log.time(time.time() - start)

    log.dataset_summary(len(train_data), len(test_data))
    return train_data, train_labels, test_data, test_labels


def evaluate_kernel(svc, name, data):
    train_data, train_labels, test_data, test_labels = data
    log.message('Training {} kernel ...'.format(name))

    start = time.time()
    svc.fit(train_data, train_labels)
    log.time(time.time() - start)

    log.message('Evaluating ...')
    predicted_labels = svc.predict(test_data)
    log.message(classification_report(test_labels, predicted_labels))
    total_accuracy = np.count_nonzero(np.array(test_labels) == np.array(predicted_labels)) / len(test_labels)
    log.accuracy(total_accuracy)

def evaluate_chi2_kernel(data, gamma, C):
    train_data, train_labels, test_data, test_labels = data
    log.message('Training chi^2 kernel ...')

    start = time.time()
    svc = svm.SVC(kernel='precomputed', C=C)
    train_kernel = chi2_kernel(train_data, gamma=gamma)
    svc.fit(train_kernel, train_labels)
    log.time(time.time() - start)

    log.message('Evaluating ...')
    test_kernel = chi2_kernel(test_data, train_data, gamma=gamma)
    predicted_labels = svc.predict(test_kernel)
    log.message(classification_report(test_labels, predicted_labels))
    total_accuracy = np.count_nonzero(np.array(test_labels) == np.array(predicted_labels)) / len(test_labels)
    log.accuracy(total_accuracy)

# http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py

def parameters_search(score, params, data, pair, n_jobs=8):

    train_data, train_labels, test_data, test_labels = data

    log.line()
    print("# Tuning hyper-parameters for %s\n" % score)
    clf = GridSearchCV(
        svm.SVC(C=1),
        params,
        cv=5,
        scoring=score,
        n_jobs=n_jobs
    )
    start = time.time()
    clf.fit(train_data, train_labels)
    log.time(time.time() - start)

    print("Best parameters set found on development set:\n%s\n" % clf.best_params_)
    print("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    pred_labels = clf.predict(test_data)
    report = classification_report(test_labels, pred_labels)
    print(report)

    with open(const.RESULTS_FILE_NAME, "a") as file:
        file.write(str(pair) + "\n")
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            file.write("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))
        file.write("\n%s\n\n" % clf.best_params_)
        file.write(report + "\n\n")

    return clf

####################################### MAIN SCRIPT #######################################

CLASSES = list(range(0, 43)) # Use (0, 43) for full dataset
COMPONENTS_CNT = 256
FEATURE = const.HOG02

if __name__ == '__main__':

    data = load_data(FEATURE, CLASSES, COMPONENTS_CNT)

    pca_kernel_params = {
        'poly2': (('poly', 1, 2, 1), 0.1),
        'poly3': (('poly', 1, 3, 1), 0.1),
        'rbf': (('rbf', 0.01, None, None), 10),
        'chi2': (('chi2', 0.01, None, None), 100)
    }
    kernel_ch = 'rbf'

    params = pca_kernel_params[kernel_ch][0]
    data = transform_data_kernel(data, COMPONENTS_CNT, params)

    svc = svm.SVC(pca_kernel_params[kernel_ch][1])

    evaluate_kernel(svc, 'svc', data)






    # Best parameters selected:
    kernels = [
        (svm.SVC(kernel='linear', C=0.1), 'linear'),
        (svm.SVC(kernel='poly', degree=2, C=0.1, gamma=0.1, coef0=0.0), 'poly2'),
        (svm.SVC(kernel='poly', degree=3, C=0.1, gamma=0.1, coef0=0.0), 'poly3'),
        (svm.SVC(kernel='rbf', C=10.0, gamma=0.01), 'rbf')
    ]

    data = transform_data(data, COMPONENTS_CNT)

    for k in kernels:
        evaluate_kernel(k[0], k[1], data)
    gamma, C = (0.01, 100.0)
    evaluate_chi2_kernel(data, gamma, C)

    kernel_params = {
        'linear': {
            'kernel': ['linear'],
            'C': [0.01, 0.1, 1, 10, 100]
        },
        'poly2': {
            'kernel': ['poly'],
            'degree': [2],
            'coef0': [0, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C': [0.1, 1, 10, 100]
        },
        'poly3': {
            'kernel': ['poly'],
            'degree': [3],
            'coef0': [0, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C': [0.1, 1, 10, 100]
        },
        'rbf': {
            'kernel': ['rbf'],
            'gamma': [0.001, 0.01, 0.1, 1, 10],
            'C': [0.1, 1, 10, 100, 1000]
        },
        'chi2': {
            'svm__kernel': ['precomputed'],
            'chi2__gamma': [1e-2, 1e-1, 1, 10, 100],
            'svm__C': [0.1, 1, 10, 100]}
    }

    SCORE = 'accuracy' # Najlepiej wybrac jeden w miare ogolny score, zeby zaoszczedzic obliczen
    # Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
    # 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
    # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',
    # 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
    # 'precision_weighted', 'r2', 'recall', 'recall_macro',
    # 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

    N_JOBS = const.N_THREADS # Set to number of processor threads for the best performance

    selected_kernel = 'poly3'
    search_procedure = chi2.parameters_search if selected_kernel == 'chi2' else parameters_search

    # for pair in const.PAIRS:
    #     data = load_data(FEATURE, pair, COMPONENTS_CNT)
    #     search_procedure(SCORE, kernel_params[selected_kernel], data, pair, N_JOBS)
