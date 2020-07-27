# coding: utf8

import os
import matplotlib.pyplot as plt
import numpy as np
import parameters
from scipy.interpolate import interp1d
from scipy import integrate
import scipy as sp

class SpectrumAirmassFixed:

    def __init__(self, file_name=""):
        """Class to load a spectrum saved in a txt file.

        Parameters
        ----------
        file_name: str
            Spectrum file name (.txt)

        Examples
        --------

        >>> file_name = 'tests/data/reduc_20170530_134_spectrum.txt'
        >>> s = SpectrumAirmassFixed(file_name)
        >>> plot_spectrum(s)
        """
        self.target = None
        self.disperseur = None
        self.airmass = -1
        self.targetx = 0
        self.targetx = 0
        self.targety = 0
        self.D2CCD = 0
        self.PIXSHIFT = 0
        self.ROTANGLE = 0
        self.PARANGLE = 0
        self.lambdas = []
        self.data = []
        self.err = []
        self.tag = ""
        self.binwidths = parameters.BINWIDTHS
        self.lambda_min = parameters.LAMBDA_MIN
        self.lambda_max = parameters.LAMBDA_MAX
        self.Bin = parameters.BIN
        self.cov = np.load(file_name.replace('.txt', '.npy'))
        if file_name != "":
            self.file_name = file_name
            self.tag = file_name.split('/')[-1]
            self.load_spec_header(file_name)
            self.load_spec_data()

    def load_spec_header(self, input_file_name):
        """Load the header from the spectrum file.

        """
        if os.path.isfile(input_file_name) and input_file_name[-3:] == 'txt':

            spec = open(input_file_name, 'r')

            for line in spec:
                Line = line.split()
                self.target = Line[1]
                self.disperseur = Line[2]
                self.airmass = float(Line[3])
                try:
                    self.psf_reg = float(Line[10])
                except:
                    self.psf_reg = 1
                """
                self.targetx = float(Line[4])
                self.targetx = float(Line[4])
                self.targety = float(Line[5])
                self.D2CCD = float(Line[6])
                self.PIXSHIFT = float(Line[7])
                self.ROTANGLE = float(Line[8])
                self.PARANGLE = float(Line[9])
                """
                break

            spec.close()

        else:
            if input_file_name[-3:] != 'txt':
                raise FileNotFoundError(
                    f'\n\tSpectrum file {input_file_name} must be converted to a txt file with conversion_spec.py')
            else:
                raise FileNotFoundError(f'\n\tSpectrum file {input_file_name} not found')

    def load_spec_data(self):
        """Load the data from the spectrum file.

        """
        spec = open(self.file_name, 'r')
        lambdas = []
        data = []
        data_err = []
        lambdas_order2 = []

        for line in spec:
            Line = line.split()
            if Line[0] != '#' and len(Line) > 4:
                lambdas.append(float(Line[2]))
                data.append(float(Line[3]))
                data_err.append(float(Line[4]))
                try:
                    lambdas_order2.append(float(Line[5]))
                except:
                    lambdas_order2.append(float(Line[2])/2)
        self.lambdas = np.array(lambdas)
        self.data = np.array(data)
        self.err = np.array(data_err)
        self.lambdas_order2 = np.array(lambdas_order2)

    def adapt_from_lambdas_to_bin(self):
        """Bin data with a list of lambdas which delimited the edges of bins.

        Returns
        -------
        fluxlum_Binobs: array_like
            Array of the spectrum in FLAM (1D).
        fluxlumBin_err: array_like
            Array of the spectrum error in FLAM (1D).

        Examples
        --------

        >>> file_name = 'tests/data/reduc_20170530_134_spectrum.txt'
        >>> s = SpectrumAirmassFixed(file_name)
        >>> fluxlum_Binobs, fluxlumBin_err, cov_bin = s.adapt_from_lambdas_to_bin()
        """
        fluxlum_Binobs = np.zeros(len(self.Bin) - 1)
        fluxlumBin_err = np.zeros(len(self.Bin) - 1)
        interpolation_obs = sp.interpolate.interp1d(self.lambdas, self.data, kind="linear", bounds_error=False,
                                                    fill_value=(0, 0))
        cov_bin = np.zeros((len(self.Bin)-1,len(self.Bin)-1))
        JMIN = list(np.zeros(len(self.Bin)-1))
        JMAX = list(np.zeros(len(self.Bin)-1))

        for v in range(len(self.Bin) - 1):
            X = np.linspace(self.Bin[v], self.Bin[v + 1], int(self.binwidths * 100))
            Y = interpolation_obs(X)
            fluxlum_Binobs[v] = integrate.simps(Y, X, dx=1) / self.binwidths

            jmin = max(np.argmin(np.abs(self.lambdas - self.Bin[v])), 1)
            jmax = min(np.argmin(np.abs(self.lambdas - self.Bin[v + 1])), len(self.lambdas) - 1)
            JMIN[v] = int(jmin)
            JMAX[v] = int(jmax)

        #print(JMIN,JMAX)
        for v in range(len(self.Bin) - 1):
            for k in range(len(self.Bin) - 1):
                #print(self.cov.shape)
                #print(JMIN[v],JMIN[k],JMAX[v],JMAX[k])
                #print(self.cov[JMIN[v]:JMAX[v],JMIN[k]:JMAX[k]])
                if JMAX[v] != JMIN[v] and JMAX[k] != JMIN[k]:
                    S = np.sum(self.cov[JMIN[v]:JMAX[v],JMIN[k]:JMAX[k]])/((JMAX[v]-JMIN[v])*(JMAX[k]-JMIN[k]))
                else:
                    S = 0
                cov_bin[v][k], cov_bin[k][v] = S, S

            fluxlumBin_err[v] = np.sqrt(cov_bin[v][v])

        return fluxlum_Binobs, fluxlumBin_err, cov_bin

def plot_spectrum(s):
    """plot the data of a SpectrumAirmassFixed.

    """
    plt.figure(figsize=[10, 10])
    plt.plot(s.lambdas, s.data, c='black')
    plt.errorbar(s.lambdas, s.data, xerr=None, yerr=s.err, fmt='none', capsize=1, ecolor='black', zorder=2,
                 elinewidth=2)
    plt.xlabel('$\lambda$ (nm)', fontsize=13)
    plt.ylabel('erg/s/cm2/nm', fontsize=13)
    plt.title('spectra: ' + s.tag[:-13] + ' with version_' + parameters.PROD_NUM +' of ' + s.target, fontsize=16)
    plt.grid(True)
    plt.show()