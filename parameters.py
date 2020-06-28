import numpy as np
import os

###________DIRECTORY STRUCTURE________###
"""
Create a directory with:
Tinst.py
parameters.py

prod/
throughput/
*outputs/* (create by Tinst.py)
tests/
____data/ 
________reduc_20170530_060_atmsim.fits
"""

###________INPUT WITH A NEW PROD________###

PROD = "CTIODataJune2017_reduced_RG715_v2_prod6.5/data_30may17_A2=0.1"
PROD_NUM = "6.5"


###________DIRECTORY PROD________###

mypath = os.path.dirname(__file__)
PROD_DIRECTORY = os.path.join(mypath, "prod/")
PROD_TXT = os.path.join(PROD_DIRECTORY, "data_txt/"+PROD)
PROD_NAME = os.path.join(PROD_DIRECTORY, "data/"+PROD)

"""
prod names: 
version_4_order2=0.05
version_6.3 : corrected simulations
version_6.4 : adr add
version_6.4_order2=0.05
version_6.4_order2=0.1
version_6.5_order2 : simu with order2
version_6.6

CC Lyon:
CTIODataJune2017_reduced_RG715_v2_prod6.5/data_30may17_A2=0.1 : version 6.5 

"""
###________DIRECTORY THROUGHPUT________###

THROUGHPUT_DIR = os.path.join(mypath, "throughput/")
rep_tel_name = os.path.join(THROUGHPUT_DIR, "ctio_throughput_1.txt") #for ctio_telescope
rep_disp_ref = os.path.join(THROUGHPUT_DIR, "Thorlab.txt")
____rep_disp_name = os.path.join(THROUGHPUT_DIR, "disperseur")
____file_tdisp_order2 = os.path.join(THROUGHPUT_DIR, "disperseur_order2")

OUTPUTS = os.path.join(mypath, "outputs/")
OUTPUTS_DIR = os.path.join(OUTPUTS, PROD)
OUTPUTS_SIMU = os.path.join(OUTPUTS_DIR, "simu/")
OUTPUTS_REDUC = os.path.join(OUTPUTS_DIR, "reduc/")

OUTPUTS_TARGET = os.path.join(OUTPUTS_DIR, "target/")
OUTPUTS_FILTER = os.path.join(OUTPUTS_DIR, "filter/")

OUTPUTS_ATM_SIM = os.path.join(OUTPUTS_SIMU, "atm/")
OUTPUTS_ATM_REDUC = os.path.join(OUTPUTS_REDUC, "atm/")

OUTPUTS_BOUGUER_SIM = os.path.join(OUTPUTS_SIMU, "bouguer/")
OUTPUTS_BOUGUER_REDUC = os.path.join(OUTPUTS_REDUC, "bouguer/")

OUTPUTS_THROUGHPUT_SIM = os.path.join(OUTPUTS_SIMU, "throughput/")
OUTPUTS_THROUGHPUT_REDUC = os.path.join(OUTPUTS_REDUC, "throughput/")


###________DISPERSER________###
DISPERSER = ['Ron400', 'Thor300', 'HoloPhP', 'HoloPhAg', 'HoloAmAg']

###________LAMBDA SCALE________###
LAMBDA_MIN = 360
LAMBDA_MAX = 1030
BINWIDTHS = 5
BIN = np.arange(LAMBDA_MIN, LAMBDA_MAX + BINWIDTHS, BINWIDTHS)
NEW_LAMBDA= 0.5 * (BIN[1:] + BIN[:-1])


###________CHECK OUTLIERS________###
MAG_MAX = 1.5

###________FILTER SPEC________###
SAVGOL_LENGTH = 9
SAVGOL_ORDER = 3
SMOOTH_LENGTH = 18
SMOOTH_WINDOW = 'gaussian'
SMOOTH_SIGMA = 6

###________FILTER DETECT LINES________###
SAVGOL_LENGTH_DL = 5
SAVGOL_ORDER_DL = 3
START_FILTER = 0.05
SAVGOL_LENGTH_GLOB = 37
SAVGOL_ORDER_GLOB = 3
STD_LENGTH = 5
TRIGGER = 4
HALF_LENGTH_MAX = 40
AROUND_LINES = 4
START_WATER = 922
END_WATER = 972
RIGHT = 980
LEFT = 378

###________DEBUG_____PLOT____SAVE________###
DEBUG = False


###________ARGUMENTS________###