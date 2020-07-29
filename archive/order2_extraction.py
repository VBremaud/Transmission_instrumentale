from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import Tinst as T

simu_A2_0 = T.SpectrumAirmassFixed(file_name='prod/version_6.5/sim_oder2=0_20170530_134.txt')
simu_A2_05 = T.SpectrumAirmassFixed(file_name='prod/version_6.5/sim_oder2=0.4_20170530_134.txt')
simu_A2_1 = T.SpectrumAirmassFixed(file_name='prod/version_6.5/sim_oder2=0.1_20170530_134.txt')
simu_A2_5 = T.SpectrumAirmassFixed(file_name='prod/version_6.5/sim_oder2=0.5_20170530_134.txt')

hdu = fits.open('../Spectractor/tests/data/sim_oder2=0.5_20170530_134.fits')
header = hdu[0].header
data_sim = hdu[0].data
Amplitude_truth = np.array(header['AMPLIS_T'][1:-1].split(' '))
Lambdas_truth = np.array(header['LBDAS_T'][1:-1].split(' '))

amplitude_truth_5 = np.zeros(len(Amplitude_truth))
lambdas_truth_5 = np.zeros(len(amplitude_truth_5))
print(Amplitude_truth)
print(Lambdas_truth)
print(len(Amplitude_truth),len(Lambdas_truth))

last=-1
for i in range(len(amplitude_truth_5)):

    j=last+1
    while Lambdas_truth[j] == '':
        j+=1

    lambdas_truth_5[i] = float(Lambdas_truth[j])
    amplitude_truth_5[i] = float(Amplitude_truth[i])
    last = j

hdu = fits.open('../Spectractor/tests/data/sim_oder2=0.1_20170530_134.fits')
header = hdu[0].header
data_sim = hdu[0].data

Amplitude_truth = np.array(header['AMPLIS_T'][1:-1].split(' '))
Lambdas_truth = np.array(header['LBDAS_T'][1:-1].split(' '))

amplitude_truth_1 = np.zeros(len(Amplitude_truth))
lambdas_truth_1 = np.zeros(len(amplitude_truth_1))
print(Amplitude_truth)
last=-1
for i in range(len(amplitude_truth_1)):

    j=last+1
    while Lambdas_truth[j] == '':
        j+=1

    lambdas_truth_1[i] = float(Lambdas_truth[j])
    amplitude_truth_1[i] = float(Amplitude_truth[i])
    last = j

hdu = fits.open('../Spectractor/tests/data/sim_oder2=0.4_20170530_134.fits')
header = hdu[0].header
data_sim = hdu[0].data

Amplitude_truth = np.array(header['AMPLIS_T'][1:-1].split(' '))
Lambdas_truth = np.array(header['LBDAS_T'][1:-1].split(' '))

amplitude_truth_05 = np.zeros(len(Amplitude_truth))
lambdas_truth_05 = np.zeros(len(amplitude_truth_05))

last=-1
for i in range(len(amplitude_truth_05)):

    j=last+1
    while Lambdas_truth[j] == '':
        j+=1

    lambdas_truth_05[i] = float(Lambdas_truth[j])
    amplitude_truth_05[i] = float(Amplitude_truth[i])
    last = j

hdu = fits.open('../Spectractor/tests/data/sim_oder2=0_20170530_134.fits')
header = hdu[0].header
data_sim = hdu[0].data

Amplitude_truth = np.array(header['AMPLIS_T'][1:-1].split(' '))
Lambdas_truth = np.array(header['LBDAS_T'][1:-1].split(' '))

amplitude_truth_0 = np.zeros(len(Amplitude_truth))
lambdas_truth_0 = np.zeros(len(amplitude_truth_0))
print(Amplitude_truth)
last=-1
for i in range(len(amplitude_truth_0)):

    j=last+1
    while Lambdas_truth[j] == '':
        j+=1

    lambdas_truth_0[i] = float(Lambdas_truth[j])
    amplitude_truth_0[i] = float(Amplitude_truth[i])
    last = j

print(lambdas_truth_0)
print(lambdas_truth_05)
print(simu_A2_0.lambdas)
print(len(lambdas_truth_0),len(simu_A2_0.lambdas),len(simu_A2_05.lambdas),len(simu_A2_1.lambdas),len(simu_A2_5.lambdas))
plt.figure(figsize=[10, 10])
plt.plot(lambdas_truth_0, amplitude_truth_0 - amplitude_truth_0, c='black', label='truth_0')
#plt.plot(lambdas_truth_05, amplitude_truth_05 - amplitude_truth_0, c='darkblue', label='truth_4')
#plt.plot(lambdas_truth_1, amplitude_truth_1 - amplitude_truth_0, c='darkgreen', label='truth_1')
#plt.plot(lambdas_truth_5, amplitude_truth_5 - amplitude_truth_0, c='darkred', label='truth_5')
plt.plot(simu_A2_0.lambdas,100*(simu_A2_0.data - amplitude_truth_0)/amplitude_truth_0,c='gray',label='simu_A2=0')
plt.plot(simu_A2_05.lambdas,100*(simu_A2_05.data - amplitude_truth_05)/amplitude_truth_05,c='blue',label='simu_A2=0.4')
plt.plot(simu_A2_1.lambdas,100*(simu_A2_1.data - amplitude_truth_1)/amplitude_truth_1,c='green',label='simu_A2=0.1')
plt.plot(simu_A2_5.lambdas,100*(simu_A2_5.data - amplitude_truth_5)/amplitude_truth_5,c='red',label='simu_A2=0.5')
plt.plot(simu_A2_5.lambdas,100*(simu_A2_052.data - amplitude_truth_0)/amplitude_truth_0,c='pink',label='simu_A2=0.4_truth0')


plt.xlabel('$\lambda$ (nm)',fontsize=13)
plt.ylabel('pourcentage', fontsize=13)
plt.title("Résidus: (data - truth_contaminé)/truth_contaminé", fontsize=16)
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=[10, 10])
plt.plot(lambdas_truth_0, amplitude_truth_0, c='black', label='truth_0')
plt.plot(lambdas_truth_05, amplitude_truth_05, c='darkblue', label='truth_05')
plt.plot(lambdas_truth_1, amplitude_truth_1, c='darkgreen', label='truth_1')
plt.plot(lambdas_truth_5, amplitude_truth_5, c='darkred', label='truth_5')
plt.plot(simu_A2_0.lambdas,simu_A2_0.data,c='gray',label='simu_A2=0')
plt.plot(simu_A2_05.lambdas,simu_A2_05.data,c='blue',label='simu_A2=0.05')
plt.plot(simu_A2_1.lambdas,simu_A2_1.data,c='green',label='simu_A2=0.1')
plt.plot(simu_A2_5.lambdas,simu_A2_5.data,c='red',label='simu_A2=0.5')

plt.xlabel('$\lambda$ (nm)',fontsize=13)
plt.ylabel('erg/s/cm2/nm', fontsize=13)
plt.title("Analyse de l'extraction de l'ordre2 sur 134, label - simu_A2=0", fontsize=16)
plt.grid(True)
plt.legend()
plt.show()


