import numpy as np
import os
import scipy as sp
from scipy.interpolate import interp1d
import parameters
import matplotlib.pyplot as plt

fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'Thor300_order2_bis.txt'), 'w')

A2 = 0.1
lambdas = np.arange(300,1100,1)
for i in range(len(lambdas)):
    fichier.write(str(lambdas[i]) + '\t' + str(A2) + '\n')

fichier.close()

ron400_banc = np.load('throughput/20190130-ronchi-145-x11-y51-thru.npy')

lambda_ron = [ron400_banc[i][0] for i in range(len(ron400_banc))]
intensite_ron = [ron400_banc[i][1] for i in range(len(ron400_banc))]
fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'ron400_banc.txt'), 'w')

Ron400 = sp.interpolate.interp1d(lambda_ron, intensite_ron, kind="linear",
                                                     bounds_error=False,
                                                     fill_value="extrapolate")
lambdas = np.arange(300,1100,1)
for i in range(len(lambdas)):
    fichier.write(str(lambdas[i]) + '\t' + str(Ron400(lambdas[i])) + '\n')

plt.plot(lambdas, Ron400(lambdas))
plt.show()
fichier.close()
