from Tools import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                    help="Enter debug mode (more verbose and plots).", default=False)
parser.add_argument("-D", "--disperser", type=str, dest="disperser",
                    help="Find the throughput of the disperser (default=None).")
parser.add_argument("-s", "--sim", dest="sim", action="store_true",default=False,
                    help="Find the throughput with simulations, else with data (default=False).")
parser.add_argument("-p", "--prod", type=str, dest="prod", default="all",
                    help="Find the throughput of all disperser with simulations (sim), data (reduc) or both (default='all').")

args = parser.parse_args()

parameters.SIM = args.sim
parameters.DISP = args.disperser

if args.disperser is not None:
    extract_throughput()

else:
    prod_analyse(data = args.prod)