from easydict import EasyDict
from data.data_read import read_h5,read_npz,read_water_quality,read_grid_inflow,read_grid_outflow
import numpy as np

CFG_DATASET = EasyDict()
# ======== dataset: PEMS08 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS08"
DATASET_ARGS.NUM_NODES = 170
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_npz
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature
DATASET_ARGS.NULL_VAL = 0
DATASET_ARGS.MASK_VAL = np.nan

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
DATASET_ARGS.DTW_FILE_PATH = "datasets/raw_data/{0}/{0}_dtw.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.SHMX_FILE_PATH = "datasets/raw_data/{0}/{0}_sh_mx.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.PATTERNKESY_FILE_PATH = "datasets/raw_data/{0}/pattern_keys_kshape_{0}.npy".format(DATASET_ARGS.NAME)

CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: PEMS07 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS07"
DATASET_ARGS.NUM_NODES = 883
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_npz
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature
DATASET_ARGS.NULL_VAL = 0
DATASET_ARGS.MASK_VAL = np.nan

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
DATASET_ARGS.DTW_FILE_PATH = "datasets/raw_data/{0}/{0}_dtw.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.SHMX_FILE_PATH = "datasets/raw_data/{0}/{0}_sh_mx.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.PATTERNKESY_FILE_PATH = "datasets/raw_data/{0}/pattern_keys_kshape_{0}.npy".format(DATASET_ARGS.NAME)

CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS

# ======== dataset: PEMS04 =============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "PEMS04"
DATASET_ARGS.NUM_NODES = 307
DATASET_ARGS.STEPS_PER_DAY = 288
DATASET_ARGS.READ_DATA_FUNC = read_npz
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature
DATASET_ARGS.NULL_VAL = 0
DATASET_ARGS.MASK_VAL = np.nan

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.npz".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
DATASET_ARGS.DTW_FILE_PATH = "datasets/raw_data/{0}/{0}_dtw.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.SHMX_FILE_PATH = "datasets/raw_data/{0}/{0}_sh_mx.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.PATTERNKESY_FILE_PATH = "datasets/raw_data/{0}/pattern_keys_kshape_{0}.npy".format(DATASET_ARGS.NAME)

CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS


# ======== dataset: T-Drive=============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "T-Drive"
DATASET_ARGS.NUM_NODES = 1024
DATASET_ARGS.STEPS_PER_DAY = 24
DATASET_ARGS.READ_DATA_FUNC = read_grid_outflow
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature
DATASET_ARGS.NULL_VAL = 0
DATASET_ARGS.MASK_VAL = 10

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.grid".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
DATASET_ARGS.DTW_FILE_PATH = "datasets/raw_data/{0}/{0}_dtw.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.SHMX_FILE_PATH = "datasets/raw_data/{0}/{0}_sh_mx.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.PATTERNKESY_FILE_PATH = "datasets/raw_data/{0}/pattern_keys_kshape_{0}.npy".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS



# ======== dataset: CHIBike=============== #
DATASET_ARGS = EasyDict()
DATASET_ARGS.NAME = "CHIBike"
DATASET_ARGS.NUM_NODES = 270
DATASET_ARGS.STEPS_PER_DAY = 48
DATASET_ARGS.READ_DATA_FUNC = read_grid_outflow
DATASET_ARGS.TOD = True                  # if add time_of_day feature
DATASET_ARGS.DOW = True                  # if add day_of_week feature
DATASET_ARGS.DOM = False                  # if add day_of_month feature
DATASET_ARGS.DOY = False                  # if add day_of_year feature
DATASET_ARGS.NULL_VAL = 0
DATASET_ARGS.MASK_VAL = 5

DATASET_ARGS.OUTPUT_DIR = "datasets/" + DATASET_ARGS.NAME
DATASET_ARGS.DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.grid".format(DATASET_ARGS.NAME)
DATASET_ARGS.GRAPH_FILE_PATH = "datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_ARGS.NAME)
DATASET_ARGS.DTW_FILE_PATH = "datasets/raw_data/{0}/{0}_dtw.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.SHMX_FILE_PATH = "datasets/raw_data/{0}/{0}_sh_mx.npy".format(DATASET_ARGS.NAME)
DATASET_ARGS.PATTERNKESY_FILE_PATH = "datasets/raw_data/{0}/pattern_keys_kshape_{0}.npy".format(DATASET_ARGS.NAME)
CFG_DATASET[DATASET_ARGS.NAME] = DATASET_ARGS





