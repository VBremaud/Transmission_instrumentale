from Tinst import TransmissionInstrumentale, plot_spec_target
import os #gestion de fichiers
import glob
import parameters

prod_txt = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_TXT)
prod_name = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME)

Throughput = TransmissionInstrumentale(prod_name=prod_txt, sim=False, disperseur='Thor300', target="HD111980", order2=False, plot_filt=False, save_filter=False, prod=glob.glob(prod_txt + "/sim*spectrum.txt"))

Throughput.spec_calspec()
plot_spec_target(Throughput, False)