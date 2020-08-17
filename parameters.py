import numpy as np
import os

###________DIRECTORY STRUCTURE________###
"""
Create a directory with:
runTinst.py
TransmissionInstrumentale.py
SpectrumRangeAirmass.py
SpectrumAirmassFixed.py
parameters.py

sps/
prod/
throughput/
*outputs/* (create by runTinst.py)
tests/
____data/ 
________reduc_20170530_060_atmsim.fits
"""
###________ARGUMENTS (change by the algorithm, do not change)________###
SIM = True
DISP = 'Thor300'

###________INPUT WITH A NEW PROD________###

PROD_NUM = "6.10"
PROD = "CTIODataJune2017_reduced_RG715_v2_prod"+PROD_NUM+"/data_30may17_A2=1"

###________CTIO_THROUGHPUT________###
THROUGHPUT_SIM = "ctio_throughput_baseThor300_prod6.9.txt" #used on simulations
THROUGHPUT_REDUC = "ctio_throughput_baseThor300_prod6.9.txt" #for comparison or throughput disperser extraction

"""
List of CTIO_THROUGHPUT:
ctio_throughput_1.txt (Sylvie) 
ctio_throughput_basethor300_prod6.7.txt
ctio_throughput_basethor300_prod6.9.txt
"""

###________DISPERSER________###
DISPERSER = ['Ron400', 'Thor300', 'HoloPhP', 'HoloPhAg', 'HoloAmAg']
DISPERSER_REF = 'Thor300'

###_________DISPERSER FILES_________###
DISPERSER_ORDER2_SIM = DISP+"_order2_sim.txt"
DISPERSER_ORDER2 = DISP+"_order2.txt"
DISPERSER_REF_BANC = DISPERSER_REF+"_banc.txt"
DISPERSER_REF_SIM = DISP+"_sim.txt"
DISPERSER_EXTRACTION = DISP+"_basectio"+DISPERSER_REF+", version_"+PROD_NUM+".txt"
DISPERSER_BANC = DISP+"_banc.txt"

###________LAMBDA SCALE________###
LAMBDA_MIN = 370
LAMBDA_MAX = 980
BINWIDTHS = 5
BIN = np.arange(LAMBDA_MIN, LAMBDA_MAX + BINWIDTHS, BINWIDTHS)
NEW_LAMBDA= 0.5 * (BIN[1:] + BIN[:-1])

###________CHECK OUTLIERS________###
MULT_MAX = 3

###________PLOTS________###
plot_atmo = True
plot_Throughput = True
plot_residuals = True
plot_fitspectrum = False

plot_specs = False
plot_target = False

###________SAVE________###
save_atmo = False
save_Throughput = False
save_residuals = False

save_target = False

###________ARGUMENTS________###
target = "HD111980"
REMOVE_SPECTRA = [58]
COMPARISON_MD_SPECTRA = []
COMPARISON_BANC = True


###_____________DO NOT CHANGE_____________###


###________DIRECTORY PROD________###

mypath = os.path.dirname(__file__)
PROD_DIRECTORY = os.path.join(mypath, "prod/")
PROD_TXT = os.path.join(PROD_DIRECTORY, "data_txt/"+PROD)
PROD_NAME = os.path.join(PROD_DIRECTORY, "data/"+PROD)

###________DIRECTORY THROUGHPUT________###

THROUGHPUT_DIR = os.path.join(mypath, "throughput/")

OUTPUTS = os.path.join(mypath, "outputs/")
OUTPUTS_DIR = os.path.join(OUTPUTS, PROD)
OUTPUTS_SIMU = os.path.join(OUTPUTS_DIR, "simu/")
OUTPUTS_REDUC = os.path.join(OUTPUTS_DIR, "reduc/")

OUTPUTS_TARGET = os.path.join(OUTPUTS_DIR, "target/")
OUTPUTS_FILTER = os.path.join(OUTPUTS_DIR, "filter/")

OUTPUTS_ATM_SIM = os.path.join(OUTPUTS_SIMU, "atm/")
OUTPUTS_ATM_REDUC = os.path.join(OUTPUTS_REDUC, "atm/")

OUTPUTS_FITSPECTRUM_SIM = os.path.join(OUTPUTS_SIMU, "fitspectrum/")
OUTPUTS_FITSPECTRUM_REDUC = os.path.join(OUTPUTS_REDUC, "fitspectrum/")

OUTPUTS_THROUGHPUT_SIM = os.path.join(OUTPUTS_SIMU, "throughput/")
OUTPUTS_THROUGHPUT_REDUC = os.path.join(OUTPUTS_REDUC, "throughput/")



