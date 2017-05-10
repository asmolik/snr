import time
from functools import partial
import read_data as d
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

main_path = 'C:/Users/Aleksander/Documents/studia/snr/'

data_training_img_path = main_path + 'GTSRB/Final_Training/Images/'
data_training_hue_path = main_path + 'GTSRB/Final_Training/HueHist/'
data_training_HOG01_path = main_path + 'GTSRB/Final_Training/HOG/HOG_01/'
data_csv_path = main_path + 'GTSRB/Final_Test/GT-final_test.csv'
data_test_img_path = main_path + 'GTSRB/Final_Test/Images/'
data_test_hue_path = main_path + 'GTSRB/Final_Test/HueHist/'
data_test_HOG_path = main_path + 'GTSRB/Final_Test/HOG/HOG_01/'

CLASSES = [1, 2]

# Read test data

test_ids, test_labels = d.read_test_image_labels(data_csv_path, classes=CLASSES)
# test_hues = d.read_test_image_hues(data_test_hue_path,
#     dict(zip(test_ids, test_labels)),
#     classes=CLASSES)
test_HOG01 = d.read_test_image_hog(data_test_HOG_path,
                                   labels=dict(zip(test_ids, test_labels)),
                                   classes=CLASSES)

# Read training data

img_ids, img_labels = d.read_training_image_labels(data_training_img_path, classes=CLASSES)
start = time.time()
# IMG_HUES = d.read_training_image_hues(data_training_hue_path, classes=CLASSES)
img_HOG01 = d.read_training_image_hog(data_training_HOG01_path, classes=CLASSES)
end = time.time()
print(end - start)



start = time.time()
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(img_HOG01, img_labels)
y = linear_svc.predict(test_HOG01)
print(classification_report(test_labels, y))
end = time.time()
print(end - start)


#to nie dziala
'''start = time.time()
chi2 = chi2_kernel(img_HOG01, gamma=1.0)
chi2_svc = svm.SVC(kernel=chi2_kernel)
chi2_svc.fit(chi2, img_labels)
y = chi2_svc.predict(test_HOG01)
print(classification_report(test_labels, y))
end = time.time()
print(end - start)'''




# http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_digits.html#sphx-glr-auto-examples-model-selection-grid-search-digits-py
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

A_train = img_HOG01
y_train = img_labels
A_test = test_HOG01
y_test = test_labels

scores = ['precision', 'recall']

# to trwa 2*1500s
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    start = time.time()
    clf.fit(A_train, y_train)
    end = time.time()
    print(end - start)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_test, y_pred = y_test, clf.predict(A_test)
    print(classification_report(y_test, y_pred))
    print()
