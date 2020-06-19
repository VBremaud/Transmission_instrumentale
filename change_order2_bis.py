import numpy as np
import os
import parameters

fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'Thor300_order2_bis.txt'), 'w')

A2 = 0.1
lambdas = np.arange(300,1100,1)
for i in range(len(lambdas)):
    fichier.write(str(lambdas[i]) + '\t' + str(A2) + '\n')

fichier.close()