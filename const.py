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
