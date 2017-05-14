""" Main module """

import time
from functools import partial
import read_data as d
import transform
import const
import log
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

CLASSES = range(1, 3) # Use (0, 43) for full dataset
COMPONENTS_CNT = 256
FEATURE = const.HOG01

if __name__ == '__main__':

    # Load and prepare data

    log.dataset_configuration(FEATURE, len(CLASSES), COMPONENTS_CNT)
    log.message('Loading data...')
    start = time.time()

    train_data, train_labels, test_data, test_labels = d.load_data(FEATURE, CLASSES)
    train_data, test_data = transform.perform_pca(train_data, test_data, COMPONENTS_CNT)
    train_data, test_data = transform.normalize_data(
        train_data,
        test_data,
        transform.get_offset_norm
    )

    log.time(time.time() - start)
    log.dataset_summary(len(train_data), len(test_data))

    # Check kernels

    def evaluate_kernel(svc, name):
        log.message('Training {} kernel ...'.format(name))
        start = time.time()

        svc.fit(train_data, train_labels)

        log.time(time.time() - start)

        log.message('Evaluating ...')
        predicted_labels = svc.predict(test_data)
        log.message(classification_report(test_labels, predicted_labels))

    kernels = [
        (svm.SVC(kernel='linear'), 'linear'),
        (svm.SVC(kernel='poly', degree=2), 'polynomial - squared'),
        (svm.SVC(kernel='poly', degree=3), 'polynomial - cubed'),
        (svm.SVC(kernel='rbf'), 'gaussian'),
        (svm.SVC(kernel=chi2_kernel), 'chi^2')
    ]

    for k in kernels:
        evaluate_kernel(k[0], k[1])

    # Optimize kernel parameters

    # http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py

    def parameters_search(score, params, n_jobs=8):

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

        print(
            "\nDetailed classification report:\n"
            "The model is trained on the full development set.\n"
            "The scores are computed on the full evaluation set.\n"
        )
        pred_labels = clf.predict(test_data)
        print(classification_report(test_labels, pred_labels))

    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    ]

    scores = ['precision_macro', 'recall_macro']
    # Valid options are ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1',
    # 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss',
    # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error',
    # 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
    # 'precision_weighted', 'r2', 'recall', 'recall_macro',
    # 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']

    for score in scores:
        parameters_search(score, tuned_parameters)
