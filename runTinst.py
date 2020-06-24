from Tools import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(dest="input", metavar='path', default=["tests/data/reduc_20170530_134_spectrum.fits"],
                    help="Input fits file name. It can be a list separated by spaces, or it can use * as wildcard.",
                    nargs='*')
parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                    help="Enter debug mode (more verbose and plots).", default=False)
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                    help="Enter verbose (print more stuff).", default=False)
parser.add_argument("-o", "--output_directory", dest="output_directory", default="outputs/",
                    help="Write results in given output directory (default: ./outputs/).")
parser.add_argument("-l", "--logbook", dest="logbook", default="ctiofulllogbook_jun2017_v5.csv",
                    help="CSV logbook file. (default: ctiofulllogbook_jun2017_v5.csv).")
parser.add_argument("-c", "--config", dest="config", default="config/ctio.ini",
                    help="INI config file. (default: config.ctio.ini).")
args = parser.parse_args()

parameters.VERBOSE = args.verbose
if args.debug:
    parameters.DEBUG = True
    parameters.VERBOSE = True

prod_txt = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_TXT)
prod_name = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME)
extract_throughput(prod_txt, True, 'Thor300', glob.glob(prod_txt + "/sim*spectrum.txt"),
                   glob.glob(prod_txt + "/reduc*spectrum.txt"), plot_specs=False, plot_bouguer=False, plot_atmo=False,
                   order2=True, mega_fit=False, save_Throughput=False, plot_Throughput=True)
# prod_analyse(prod_name, prod_txt, data = 'sim')