""" Paths to data and other constants """

MAIN_DATA_PATH = 'C:/Users/Aleksander/Documents/studia/snr/'

HUE = 'HUE'
HOG01 = 'HOG01'
HOG02 = 'HOG02'
HOG03 = 'HOG03'
HAAR = 'HAAR'

DATA_CSV_PATH = MAIN_DATA_PATH + 'GTSRB/Final_Test/GT-final_test.csv'
DATA_TRAINING_IMG_PATH = MAIN_DATA_PATH + 'GTSRB/Final_Training/Images/'
DATA_TEST_IMG_PATH = MAIN_DATA_PATH + 'GTSRB/Final_Test/Images/'

RESULTS_FILE_NAME = "results/results.txt"

TRAINING_FEATURES_PATHS = {
    HUE: MAIN_DATA_PATH + 'GTSRB/Final_Training/HueHist/',
    HOG01: MAIN_DATA_PATH + 'GTSRB/Final_Training/HOG/HOG_01/',
    HOG02: MAIN_DATA_PATH + 'GTSRB/Final_Training/HOG/HOG_02/',
    HOG03: MAIN_DATA_PATH + 'GTSRB/Final_Training/HOG/HOG_03/',
    HAAR: MAIN_DATA_PATH + 'GTSRB/Final_Training/Haar/'
}

TEST_FEATURE_PATHS = {
    HUE: MAIN_DATA_PATH + 'GTSRB/Final_Test/HueHist/',
    HOG01: MAIN_DATA_PATH + 'GTSRB/Final_Test/HOG/HOG_01/',
    HOG02: MAIN_DATA_PATH + 'GTSRB/Final_Test/HOG/HOG_02/',
    HOG03: MAIN_DATA_PATH + 'GTSRB/Final_Test/HOG/HOG_03/',
    HAAR: MAIN_DATA_PATH + 'GTSRB/Final_Test/Haar/'
}

N_THREADS = 8

PAIRS = [(29, 13), (7, 15), (19, 1), (34, 8), (35, 27), (26, 9), (14, 18), (10, 33), (5, 37), (22, 16), (36, 23),
         (12, 28), (17, 40), (0, 30), (41, 11), (39, 21), (6, 24), (20, 31), (3, 32), (38, 2), (25, 42)]
