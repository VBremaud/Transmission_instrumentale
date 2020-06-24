from Tools import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                    help="Enter debug mode (more verbose and plots).", default=False)
parser.add_argument("-disp", "--disperser", type=str, dest="disperser",
                    help="Find the throughput of the disperser (default=None).")
parser.add_argument("-s", "--sim", dest="sim", action="store_true",default=False,
                    help="Find the throughput with simulations, else with data (default=False).")
parser.add_argument("-p", "--prod", type=str, dest="prod", default="all",
                    help="Find the throughput of all disperser with simulations (sim), data (reduc) or both (default='all').")

args = parser.parse_args()

parameters.VERBOSE = args.verbose
disperser = args.disperser

if args.debug:
    parameters.DEBUG = True
    parameters.VERBOSE = True

prod_txt = parameters.PROD_TXT
prod_name = parameters.PROD_NAME

if disperser is not None:
    extract_throughput(prod_txt, args.sim, disperser, glob.glob(prod_txt + "/sim*spectrum.txt"),
                   glob.glob(prod_txt + "/reduc*spectrum.txt"), plot_specs=False, plot_bouguer=False, plot_atmo=False,
                   order2=True, mega_fit=False, save_Throughput=False, plot_Throughput=True)

else:
    prod_analyse(prod_name, prod_txt, data = args.prod)