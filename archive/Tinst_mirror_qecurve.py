from scipy import signal  # filtre savgol pour enlever le bruit
from scipy.interpolate import interp1d  # interpolation
from spectractor.tools import wavelength_to_rgb  # couleurs des longueurs d'ondes
from SpectrumRangeAirmass import *
from Filters import *
#coding utf8

fig = plt.figure(figsize=[15, 10])
ax2 = fig.add_subplot(111)
lsst_mirror = open('../Spectractor/spectractor/simulation/CTIOThroughput/lsst_mirrorthroughput.txt','r')
qecurve = open('../Spectractor/spectractor/simulation/CTIOThroughput/qecurve.txt','r')
mypath = os.path.dirname(__file__)
THROUGHPUT_DIR = os.path.join(mypath, "throughput/")
rep_tel_name = os.path.join(THROUGHPUT_DIR, "ctio_throughput_basethor300_prod6.9_new.txt")

lsst_intensite = []
lsst_lambdas = []
for line in lsst_mirror:
    Line = line.split()
    lsst_intensite.append(float(Line[1]))
    lsst_lambdas.append(float(Line[0]))

qecurve_i = []
qecurve_l = []
for line in qecurve:
    Line = line.split()
    qecurve_l.append(float(Line[0]))
    qecurve_i.append(float(Line[1]))

tel = np.loadtxt(rep_tel_name)
Data_tel = sp.interpolate.interp1d(tel.T[0], tel.T[1], kind="linear", bounds_error=False,
                                           fill_value="extrapolate")

lambdas = np.arange(370,980,5)
Lsst_mirror = sp.interpolate.interp1d(lsst_lambdas, lsst_intensite, kind="linear", bounds_error=False,
                                               fill_value="extrapolate")
Qecurve = sp.interpolate.interp1d(qecurve_l, qecurve_i, kind="linear", bounds_error=False,
                                               fill_value="extrapolate")
qecurve_i = np.array(qecurve_i)
qecurve_l = np.array(qecurve_l)
x = 'throughput/20171006_RONCHI400_clear_45_median_tpt.txt'
a = np.loadtxt(x)
x = 'throughput/CBP_throughput.dat'
b = np.loadtxt(x)

def takePREMIER(elem):
    return elem[0]

A = [[a.T[0][i], a.T[1][i]] for i in range(len(a.T[0]))]
A.sort(key=takePREMIER)
a.T[0] = [A[i][0] for i in range(len(A))]
a.T[1] = [A[i][1] for i in range(len(A))]

if a.T[0][0] < b.T[0][0]:
    L = np.linspace(a.T[0][0] - 1, b.T[0][0], int(b.T[0][0] - a.T[0][0] + 1))
    Y_L = [0.006] * int((b.T[0][0] - a.T[0][0] + 1))
    X = np.concatenate((L, b.T[0]))
    Z = np.concatenate((Y_L, b.T[1]))

if a.T[0][len(a.T[0]) - 1] > X[len(X) - 1]:
    L = np.linspace(X[len(X) - 1], a.T[0][len(a.T[0]) - 1], int(a.T[0][len(a.T[0]) - 1] - X[len(X) - 1] + 1))
    Y_L = [0.0021] * int((a.T[0][len(a.T[0]) - 1] - X[len(X) - 1] + 1))
    M = np.concatenate((X, L))
    N = np.concatenate((Z, Y_L))

interpolation = interp1d(M, N)
Y2 = interpolation(a.T[0])
Ynew = np.array([a.T[1][i] / Y2[i] for i in range(len(Y2))])
Y3 = interp1d(a.T[0], Ynew)

X = lambdas
Y = Lsst_mirror(X)
Y1 = Data_tel(X)

I_tel = integrate.simps(Y1, X, dx=1) / (lambdas[-1] - lambdas[0])
I = Lsst_mirror(lambdas)**2*Qecurve(lambdas)
I_lsst_mirror = integrate.simps(I, X, dx=1) / (lambdas[-1] - lambdas[0])
I_cbp = integrate.simps(Ynew, a.T[0], dx=1) / (max(a.T[0]) - min(a.T[0]))
#* max(tel.T[1]) / max(qecurve_i)
plt.scatter(a.T[0],np.array(Ynew) * I_tel / I_cbp, c='purple', marker='.', label='T_inst CBP')
plt.scatter(lambdas, I * I_tel / I_lsst_mirror, c= 'blue', label='Tinst_mirror+qe')
plt.scatter(tel.T[0], tel.T[1], c='red', label = 'Tinst_photometric_night_fit')
plt.errorbar(tel.T[0], tel.T[1], xerr=None, yerr=tel.T[2], fmt='none', capsize=1,
                     ecolor='red', zorder=2, elinewidth=2)
ax2.get_xaxis().set_tick_params(labelsize=15)
ax2.get_yaxis().set_tick_params(labelsize=15)
plt.grid(True)
plt.legend(loc='best', prop={'size': 14})
plt.title("Instrument transmission of the CTIO", fontsize=22)
plt.xlabel('$\lambda$ [nm]', fontsize=24)
plt.ylabel("Instrument transmission", fontsize=22)
plt.grid(True)
ax2.axis([370,980,0,0.5])
fig.tight_layout()
plt.savefig('Comparaison_tinst.pdf')
plt.show()