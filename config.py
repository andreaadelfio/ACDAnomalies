'''
Configuration file for ACNBkg.scripts
'''
import os

USER = os.environ.get('USERNAME')
DIR = os.path.dirname(os.path.realpath(__file__))

DATA_BAT_REL_FOLDER_PATH = 'data/BATgrbs'
DATA_BAT_FOLDER_PATH = os.path.join(os.getcwd(), DIR, DATA_BAT_REL_FOLDER_PATH)

DATA_LATACD_REL_FOLDER_PATH = 'data/LATACD/output runs'
DATA_LATACD_FOLDER_PATH = os.path.join(os.getcwd(), DIR, DATA_LATACD_REL_FOLDER_PATH)

LOGGING_FILE_NAME = 'acnbkg.log'
LOGGING_FOLDER_NAME = 'logs'
LOGGING_FILE_REL_PATH = os.path.join(DIR, LOGGING_FOLDER_NAME, LOGGING_FILE_NAME)
LOGGING_FILE_PATH = os.path.join(os.getcwd(), LOGGING_FILE_REL_PATH)

SC_FOLDER_NAME = 'data/spacecraft'
SC_LAT_FILENAME = 'lat_spacecraft_merged.fits'
SC_LAT_FILE_REL_PATH = os.path.join(DIR, SC_FOLDER_NAME, SC_LAT_FILENAME)
SC_LAT_FILE_PATH = os.path.join(os.getcwd(), SC_LAT_FILE_REL_PATH)

SC_GBM_FILENAME = 'gbm_spacecraft_merged.fits'
SC_GBM_FILE_REL_PATH = os.path.join(DIR, SC_FOLDER_NAME, SC_GBM_FILENAME)
SC_GBM_FILE_PATH = os.path.join(os.getcwd(), SC_GBM_FILE_REL_PATH)

SC_FOLDER_NAME_FROM_GBM = 'data/spacecraft/from_gbm'
SC_FILENAMES_FROM_GBM = os.listdir(SC_FOLDER_NAME_FROM_GBM)
SC_FILE_REL_PATHS_FROM_GBM = []
SC_FILE_PATHS_FROM_GBM = []
for SC_FILENAMES_FROM_GBM in SC_FILENAMES_FROM_GBM:
    SC_FILE_REL_PATH_FROM_GBM = os.path.join(DIR, SC_FOLDER_NAME_FROM_GBM, SC_FILENAMES_FROM_GBM)
    SC_FILE_REL_PATHS_FROM_GBM.append(SC_FILE_REL_PATH_FROM_GBM)
    SC_FILE_PATHS_FROM_GBM.append(os.path.join(os.getcwd(), SC_FILE_REL_PATH_FROM_GBM))

def regenerate_gbm():
    SC_FOLDER_NAME_FROM_GBM = 'data/spacecraft/from_gbm'
    SC_FILENAMES_FROM_GBM = os.listdir(SC_FOLDER_NAME_FROM_GBM)
    SC_FILE_REL_PATHS_FROM_GBM = []
    SC_FILE_PATHS_FROM_GBM = []
    for SC_FILENAMES_FROM_GBM in SC_FILENAMES_FROM_GBM:
        SC_FILE_REL_PATH_FROM_GBM = os.path.join(DIR, SC_FOLDER_NAME_FROM_GBM, SC_FILENAMES_FROM_GBM)
        SC_FILE_REL_PATHS_FROM_GBM.append(SC_FILE_REL_PATH_FROM_GBM)
        SC_FILE_PATHS_FROM_GBM.append(os.path.join(os.getcwd(), SC_FILE_REL_PATH_FROM_GBM))
    return SC_FILE_PATHS_FROM_GBM