import platform

REMOVE_UNWANTED_MAIN_TRAIN_CLS = True
REMOVE_UNWANTED_TEST_CLASS = False
DEVICE = 'cuda'

AMOUNT_OF_COLS_FOR_LOG_IMAGES = 4
#AMOUNT_OF_ROWS_FOR_LOG_IMAGES = 8

DIRT_IN_DATASET = 0.01  # amount of dirt in new dataset
OUTLIER_QUANTILE = 0.01

if platform.system() == 'Windows':
    MAIN_DIR = 'F:\\master_data\\'
else:
    MAIN_DIR = '~/master_data/'
