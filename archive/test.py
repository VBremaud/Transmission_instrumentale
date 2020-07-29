# coding: utf8

import os #gestion de fichiers
import matplotlib.pyplot as plt #affichage
import numpy as np #calculs utilisant C
import glob
from astropy.io import fits
import parameters
from scipy import signal #filtre savgol pour enlever le bruit
from scipy.interpolate import interp1d #interpolation
from scipy import integrate #integation
import scipy as sp #calculs
from scipy import misc
import emcee
import corner
import multiprocessing
import sys
from schwimmbad import MPIPool

from spectractor.extractor.spectrum import Spectrum #importation des spectres
from spectractor.tools import wavelength_to_rgb #couleurs des longueurs d'ondes
from spectractor.simulation.simulator import AtmosphereGrid # grille d'atmosphère
import spectractor.parameters as parameterss
from spectractor.simulation.adr import adr_calib



class SpectrumAirmassFixed:

    """
    Examples
    ----------------
    Load a spectrum from a txt file
    >>> s = SpectrumAirmassFixed(file_name='tests/data/reduc_20170530_134_spectrum.txt')
    >>> s.load_spec_data()
    """

    def __init__(self, file_name=""):

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
        if file_name != "":
            self.file_name = file_name
            self.tag = file_name.split('/')[-1]
            self.load_spec_header(file_name)
            self.load_spec_data()


    def load_spec_header(self, input_file_name):
        if os.path.isfile(input_file_name) and input_file_name[-3:]=='txt':

            spec = open(input_file_name, 'r')

            for line in spec:
                Line = line.split()
                self.target = Line[1]
                self.disperseur = Line[2]
                self.airmass = float(Line[3])
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
                raise FileNotFoundError(f'\n\tSpectrum file {input_file_name} must be converted to a txt file with conversion_spec.py')
            else:
                raise FileNotFoundError(f'\n\tSpectrum file {input_file_name} not found')

    def load_spec_data(self):
            spec = open(self.file_name, 'r')
            lambdas=[]
            data=[]
            data_err=[]
            lambdas_order2 = []

            for line in spec:
                Line = line.split()
                if Line[0] != '#' and len(Line)>4:
                    lambdas.append(float(Line[2]))
                    data.append(float(Line[3]))
                    data_err.append(float(Line[4]))
                    lambdas_order2.append(float(Line[5]))

            self.lambdas = np.array(lambdas)
            self.data = np.array(data)
            self.err = np.array(data_err)
            self.lambdas_order2 = np.array(lambdas_order2)

    def adapt_from_lambdas_to_bin(self):
        fluxlum_Binobs = np.zeros(len(self.Bin)-1)
        fluxlumBin_err = np.zeros(len(self.Bin)-1)
        "tableaux de taille len(Bin)-1 et comportant les intensites par bin"
        "tableaux de taille len(Bin)-1 et comportant les erreurs d'intensites par bin"

        interpolation_obs = sp.interpolate.interp1d(self.lambdas,self.data,kind="linear", bounds_error=False, fill_value=(0, 0))
        print(self.tag)

        for v in range(len(self.Bin)-1):
            "On rempli les tableaux par bin de longueur d'onde"
            X = np.linspace(self.Bin[v],self.Bin[v+1], int(self.binwidths * 100))
            Y = interpolation_obs(X)
            fluxlum_Binobs[v] = integrate.simps(Y,X,dx=1)/self.binwidths

            jmin = max(np.argmin(np.abs(self.lambdas - self.Bin[v])),1)
            jmax = min(np.argmin(np.abs(self.lambdas - self.Bin[v+1])),len(self.lambdas)-1)
            S = 0
            "Propagation des incertitudes sur les intensites par bin, calcul sur les bords"
            for j in range(jmin, jmax):
                S += (self.err[j] * (self.lambdas[j + 1] - self.lambdas[j - 1]) / 2) ** 2

            fluxlumBin_err[v] = np.sqrt(S) / self.binwidths

        return fluxlum_Binobs, fluxlumBin_err

class SpectrumRangeAirmass:

    def __init__(self, prod_name="", sim="", disperseur="", target="", plot_specs="", prod_sim ="", prod_reduc=""):

        self.target = target
        self.disperseur = disperseur
        self.plot_specs = plot_specs
        self.binwidths = parameters.BINWIDTHS
        self.lambda_min = parameters.LAMBDA_MIN
        self.lambda_max = parameters.LAMBDA_MAX
        self.new_lambda = parameters.NEW_LAMBDA
        self.Bin = parameters.BIN
        self.sim = sim
        self.list_spectrum=[]
        self.data_mag = []
        self.range_airmass = []
        self.err_mag = []
        self.order2 = []
        self.names = []
        # ATTENTION à modifier #
        self.file_tdisp_order2 = os.path.join(parameters.THROUGHPUT_DIR, 'Thor300_order2_bis.txt') #self.disperseur +
        self.prod_sim = prod_sim
        self.prod_reduc = prod_reduc
        if prod_name != "":
            self.prod_name = prod_name
            self.init_spectrumrangeairmass()
            self.data_range_airmass()
            self.check_outliers()

    def init_spectrumrangeairmass(self):
        if self.sim:
            self.list_spectrum = self.prod_sim
        else:
            self.list_spectrum = self.prod_reduc

        self.Bin = np.arange(self.lambda_min, self.lambda_max + self.binwidths, self.binwidths)
        "Division de la plage spectrale en bin de longueur d'onde de longueur binwidths"

        for j in range(len(self.Bin) - 1):
            self.data_mag.append([])
            self.range_airmass.append([])
            self.err_mag.append([])
            self.order2.append([])

    def data_range_airmass(self):
        t_disp = np.loadtxt(self.file_tdisp_order2)
        T_disperseur = sp.interpolate.interp1d(t_disp.T[0], t_disp.T[1], kind="linear", bounds_error=False,
                                               fill_value="extrapolate")
        for i in range(len(self.list_spectrum)):
            s=SpectrumAirmassFixed(file_name = self.list_spectrum[i])

            if s.target == self.target and s.disperseur == self.disperseur:
                s.load_spec_data()
                data_bin, err_bin = s.adapt_from_lambdas_to_bin()

                data, err = convert_from_flam_to_mag(data_bin, err_bin)
                for v in range(len(self.Bin)-1):
                    self.data_mag[v].append(data[v])
                    self.range_airmass[v].append(s.airmass)
                    self.err_mag[v].append(err[v])

                    data_conv = interp1d(self.new_lambda, data_bin, kind="linear",
                                         bounds_error=False, fill_value=(0, 0))
                    lambdas_conv = interp1d(s.lambdas, s.lambdas_order2, kind="linear",
                                         bounds_error=False, fill_value=(0, 0))
                    LAMBDAS_ORDER2 = lambdas_conv(self.new_lambda)
                    I_order2 = data_conv(LAMBDAS_ORDER2) * T_disperseur(LAMBDAS_ORDER2)

                    self.order2[v].append(I_order2[v] * LAMBDAS_ORDER2[v] * np.gradient(LAMBDAS_ORDER2)[v]
                                          / np.gradient(self.new_lambda)[v] / self.new_lambda[v])


                self.names.append(self.list_spectrum[i])
                if self.plot_specs:
                    plot_spectrums(s)

        self.data_mag = np.array(self.data_mag)
        self.range_airmass = np.array(self.range_airmass)
        self.err_mag = np.array(self.err_mag)
        self.names = np.array(self.names)
        self.order2 = np.array(self.order2)

    def bouguer_line(self):
        def flin(x,a,b):
            return a * x + b

        slope = np.zeros(len(self.data_mag))  # on initialise la liste des coefficients à la liste vide.
        ord = np.zeros(len(self.data_mag))
        err_slope = np.zeros(len(self.data_mag))  # on initialise la liste des erreurs sur les ordonnees à l'origine à la liste vide.
        err_ord = np.zeros(len(self.data_mag))

        for i in range(len(self.data_mag)):
            popt, pcov = sp.optimize.curve_fit(flin, self.range_airmass[i], self.data_mag[i],sigma=self.err_mag[i])
            slope[i], ord[i] = popt[0], popt[1]
            err_ord[i] = np.sqrt(pcov[1][1])
            err_slope[i] = np.sqrt(pcov[0][0])

        return slope, ord, err_slope, err_ord

    def bouguer_line_order2(self):
        def forder2(x, a, b, A2=1):
            return np.log(np.exp(a * x + b) + A2 * order2)

        slope, ord, err_slope, err_ord = self.bouguer_line()
        A2 = np.zeros(len(self.data_mag))
        A2_err = np.zeros(len(self.data_mag))
        """
        Lambdas_order2 = self.new_lambda[j:] / 2
        lambdas_order2 = self.new_lambda[j:]
        sim_conv = interp1d(self.new_lambda, np.exp(self.data_mag) * self.new_lambda, kind="linear", bounds_error=False, fill_value=(0, 0))
        err_conv = interp1d(self.new_lambda, np.exp(self.err_mag) * self.new_lambda, kind="linear", bounds_error=False, fill_value=(0, 0))
        spectrum_order2 = sim_conv(Lambdas_order2) / lambdas_order2
        err_order2 = err_conv(Lambdas_order2) / lambdas_order2
        """

        for i in range(len(self.data_mag)):

            order2 = self.order2[i]
            #print(order2)
            Emcee = True
            if Emcee:
                def log_likelihood(theta, x, y, yerr):
                    a, b = theta
                    model = forder2(x, a, b)
                    sigma2 = yerr * yerr

                    return -0.5 * np.sum((y - model) ** 2 / sigma2)

                def log_prior(theta):
                    a, b = theta
                    if slope[i] < a < 0 and ord[i] - 2 < b < ord[i]:
                        return 0
                    return -np.inf

                def log_probability(theta, x, y, yerr=1):
                    lp = log_prior(theta)
                    if not np.isfinite(lp):
                        return -np.inf
                    return lp + log_likelihood(theta, x, y, yerr)

                p0 = np.array([slope[i], ord[i]])
                #print(p0)
                walker = 10
                init_a = p0[0] - p0[0] / 100 * 5 * abs(np.random.randn(walker))
                init_b = p0[1] + p0[1] / 100 * 5 * abs(np.random.randn(walker))
                # init_A2 = p0[2] + p0[2] / 100 * 50 * np.random.randn(walker)

                pos = np.array([[init_a[i], init_b[i]] for i in range(len(init_a))])
                # pos = p0 + np.array([init_a,init_b, init_A2])
                # pos = p0 + p0 / 100 * 5 * (2 * np.random.randn(20, 3) - 1)
                #print(pos)
                nwalkers, ndim = pos.shape

                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                                args=(self.range_airmass[i], self.data_mag[i], self.err_mag[i]))
                sampler.run_mcmc(pos, 1000, progress=False)
                """
                fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
                samples = sampler.get_chain()
                labels = ["a", "b"]
                for i in range(ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
                """
                flat_samples = sampler.get_chain(discard=200, thin=1, flat=True)
                """
                fig = corner.corner(
                    flat_samples, labels=labels, truths=p0, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}
                )
                plt.show()
                """
                mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
                mcmc2 = np.percentile(flat_samples[:, 1], [16, 50, 84])
                # mcmc3 = np.percentile(flat_samples[:, 2], [16, 50, 84])
                # q3 = np.diff(mcmc3)
                q1 = np.diff(mcmc)

                A2[i] = 1
                A2_err[i] = 0
                # (abs(q3[0])+abs(q3[1]))/2
                slope[i] = mcmc[1]
                ord[i] = mcmc2[1]
                err_slope[i] = (abs(q1[0]) + abs(q1[1])) / 2
                #print(slope[i])
            else:
                try:
                    popt, pcov = sp.optimize.curve_fit(forder2, self.range_airmass[i], self.data_mag[i],
                                                       p0=[slope[i], ord[i]], sigma=self.err_mag[i],
                                                       bounds=([slope[i], ord[i] - 2], [0, ord[i]]), verbose=2)
                    print(slope[i], ord[i], popt[0], popt[1])
                    slope[i], ord[i] = popt[0], popt[1]
                    err_ord[i] = min(np.sqrt(pcov[1][1]), err_ord[i])
                    err_slope[i] = min(np.sqrt(pcov[0][0]), err_slope[i])
                    A2_err[i] = 0
                    A2[i] = 1
                except RuntimeError:
                    slope[i], ord[i], A2[i] = slope[i], ord[i], 1
                    err_ord[i] = 0
                    err_slope[i] = 0
                    A2_err[i] = 0

        """
        fig = plt.figure(figsize=[15, 10])
        X=np.arange(1,len(self.Bin),1)
        print(A2)
        plt.plot(X, A2, color='blue', label='transmission libradtran typique')
        plt.show()
        """
        return slope, ord, err_slope, err_ord, A2, A2_err

    def megafit_emcee(self):
        nsamples = 8

        atm = []
        for j in range(len(self.names)):
            prod_name = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME)
            atmgrid = AtmosphereGrid(
                filename=(prod_name + '/' + self.names[j].split('/')[-1]).replace('sim','reduc').replace('spectrum.txt', 'atmsim.fits'))
            atm.append(atmgrid)

        def f_tinst_atm(Tinst, ozone, eau, aerosols, atm):
            model = np.zeros((len(self.data_mag),len(self.names)))
            for j in range(len(self.names)):
                a = atm[j].simulate(ozone, eau, aerosols)
                model[:,j] = Tinst * a(self.new_lambda)
            return model

        def log_likelihood(params_fit, atm, y, yerr):
            Tinst, ozone, eau, aerosols = params_fit[:-3], params_fit[-3], params_fit[-2], params_fit[-1]
            model = f_tinst_atm(Tinst, ozone, eau, aerosols, atm)
            sigma2 = yerr * yerr
            return -0.5 * np.sum((y - model) ** 2 / sigma2)

        def log_prior(params_fit):
            Tinst, ozone, eau, aerosols = params_fit[:-3], params_fit[-3], params_fit[-2], params_fit[-1]

            """
            if np.any(Tinst < 0) or np.any(Tinst > 1):
                return -np.inf
            """

            if 100 < ozone < 700 and 0 < eau < 10 and 0 < aerosols < 0.1:
                return 0
            else:
                return -np.inf

        def log_probability(params_fit, atm, y, yerr):
            lp = log_prior(params_fit)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params_fit, atm, y, yerr)

        slope, ord, err_slope, err_ord, A2, A2_err = self.bouguer_line_order2()
        p_ozone = 300
        p_eau = 5
        p_aerosols = 0.03
        p0 = np.array([np.exp(ord), p_ozone, p_eau, p_aerosols])
        walker = 300
        init_Tinst = []
        for i in range(walker):
            init_Tinst.append(p0[0] + 0.1 * np.random.randn(len(self.data_mag)))
        init_ozone = p0[1] + p0[1] / 5 * np.random.randn(walker)
        init_eau = p0[2] + p0[2] / 5 * np.random.randn(walker)
        init_aerosols = p0[3] + p0[3] / 5 * np.random.randn(walker)
        init_Tinst = np.array(init_Tinst)
        pos = []
        for i in range(len(init_Tinst)):
            T = []
            for j in range(len(init_Tinst[0])):
                T.append(init_Tinst[i][j])
            T.append(init_ozone[i])
            T.append(init_eau[i])
            T.append(init_aerosols[i])
            print(T[-3:])
            pos.append(T)
        p0 = np.array(pos)
        nwalkers, ndim = p0.shape

        """
        plt.errorbar(self.new_lambda, (np.exp(self.data_mag) - self.order2)[:,10], yerr=(self.err_mag * np.exp(self.data_mag))[:,10])
        plt.show()
        """

        filename = "sps/"+ self.disperseur +"_emcee.h5"
        backend = emcee.backends.HDFBackend(filename)
        """
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(atm, (np.exp(self.data_mag) - self.order2), self.err_mag * np.exp(self.data_mag)), threads=multiprocessing.cpu_count())
        sampler.run_mcmc(pos, 2000, progress=True)
        """
        try:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(atm, (np.exp(self.data_mag) - self.order2), self.err_mag * np.exp(self.data_mag)), pool=pool, backend=backend)
            if backend.iteration > 0:
                p0 = backend.get_last_sample()

            if nsamples - backend.iteration > 0:
                sampler.run_mcmc(p0, nsteps=max(0, nsamples - backend.iteration), progress=True)
            pool.close()
        except ValueError:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                        args=(atm, (np.exp(self.data_mag) - self.order2), self.err_mag * np.exp(self.data_mag)),
                                            threads=multiprocessing.cpu_count(), backend=backend)
            if backend.iteration > 0:
                p0 = sampler.get_last_sample()
            for _ in sampler.sample(p0, iterations=max(0, nsamples - backend.iteration), progress=True, store=True):
                continue

        flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)

        Tinst = np.mean(flat_samples, axis=0)[:-3]
        Tinst_err = np.std(flat_samples, axis=0)[:-3]

        """
        for i in range(5):
            plt.hist(flat_samples[:,i],bins=100)
            plt.show()
        """
        #Tinst = sp.signal.savgol_filter(Tinst, 31, 2)
        print(np.mean(flat_samples[:, -3]), np.std(flat_samples[:, -3]))
        print(np.mean(flat_samples[:, -2]), np.std(flat_samples[:, -2]))
        print(np.mean(flat_samples[:, -1]), np.std(flat_samples[:, -1]))
        return Tinst, Tinst_err

    def check_outliers(self):
        indice =[]
        outliers_detected = True
        while outliers_detected == True:
            for bin in range(len(self.Bin) - 1):
                if np.max(np.abs(self.data_mag[bin] - np.mean(self.data_mag[bin]))) > parameters.MAG_MAX:
                    outliers = np.argmax(np.abs(self.data_mag[bin] - np.mean(self.data_mag[bin])))
                    print("----> outliers detected:" + self.names[outliers].split('/')[-1] + "\t" + ", multiplicative factor compared to avg value: "+ str(np.exp(np.max(np.abs(self.data_mag[bin] - np.mean(self.data_mag[bin])))))[:3]+
                          " for "+str(self.Bin[bin])+"-"+str(self.Bin[bin+1])+" nm \n")
                    s = SpectrumAirmassFixed(file_name=self.names[outliers])
                    pl = input("Do you want check this spectra (y/n)? ")
                    if pl == 'y':
                        plot_spectrums(s)
                    rm = input("Do you want keep this spectra (y/n)? ")
                    if rm == 'n':
                        indice.append(outliers)
                    else:
                        continue
                    break
            if len(indice) == 0:
                outliers_detected = False
            else:
                outliers_detected = True
                self.names =np.delete(self.names, indice[0])
                self.data_mag = np.delete(self.data_mag, indice[0],1)
                self.range_airmass = np.delete(self.range_airmass, indice[0],1)
                self.err_mag = np.delete(self.err_mag, indice[0],1)
                self.order2 = np.delete(self.order2, indice[0],1)

                indice = []


class TransmissionInstrumentale:

    def __init__(self, prod_name="", sim="", disperseur="", target="", order2="", mega_fit="", plot_filt="", save_filter="",prod=""):
        self.target = target
        self.disperseur = disperseur
        self.order2 = order2
        self.mega_fit = mega_fit
        self.A2 = []
        self.A2_err = []
        self.t_disp_order2 = []
        self.plot_filt = plot_filt
        self.save_filter = save_filter
        self.binwidths = parameters.BINWIDTHS
        self.lambda_min = parameters.LAMBDA_MIN
        self.lambda_max = parameters.LAMBDA_MAX
        self.Bin = parameters.BIN
        self.new_lambda = parameters.NEW_LAMBDA
        self.sim = sim
        self.lambdas_calspec = []
        self.data_calspec = []
        self.data_calspec_org = []
        self.data_calspec_mag = []
        self.data = []
        self.data_order2 = []
        self.data_tel = []
        self.data_tel_err = []
        self.data_disp = []
        self.data_disp_err = []
        self.lambdas = []
        self.slope = []
        self.slope2 = []
        self.ord = []
        self.ord2 = []
        self.err_slope = []
        self.err_slope2 = []
        self.err_ord = []
        self.err_ord2 = []
        self.err = []
        self.err_order2 = []
        self.data_bouguer = []
        self.prod_name = prod_name
        self.file_calspec = prod[0]
        self.rep_tel_name = parameters.rep_tel_name
        self.rep_disp_ref = parameters.rep_disp_ref
        self.rep_disp_name = os.path.join(parameters.THROUGHPUT_DIR, self.disperseur + '.txt')
        self.spec_calspec()

    def spec_calspec(self):
        spec = open(self.file_calspec, 'r')
        lambdas = []
        data = []

        for line in spec:
            Line = line.split()
            if Line[0] != '#':
                lambdas.append(float(Line[0]))
                data.append(float(Line[1]))

        self.lambdas_calspec = np.array(lambdas)
        self.data_calspec_org = np.array(data)
        #self.data_calspec = filter_detect_lines(self.lambdas_calspec, self.data_calspec_org)
        self.data_calspec = self.data_calspec_org #ATTENTION
        fluxlum_Binreel = np.zeros(len(self.Bin) - 1)
        interpolation_reel = sp.interpolate.interp1d(self.lambdas_calspec, self.data_calspec, kind="linear", bounds_error=False,
                                                    fill_value=(0, 0))
        for v in range(len(self.Bin) - 1):
            "On rempli les tableaux par bin de longueur d'onde"
            X = np.linspace(self.Bin[v], self.Bin[v + 1], int(self.binwidths * 100))
            Y = interpolation_reel(X)
            fluxlum_Binreel[v] = integrate.simps(Y, X, dx=1) / self.binwidths

        self.data_calspec = fluxlum_Binreel
        self.data_calspec_mag = convert_from_flam_to_mag(fluxlum_Binreel,np.zeros(len(fluxlum_Binreel)))

    def calcul_throughput(self, spectrumrangeairmass):
        data_mag = spectrumrangeairmass.data_mag.T
        data_mag -= self.data_calspec_mag[0]
        spectrumrangeairmass.data_mag = data_mag.T

        data_order2 = spectrumrangeairmass.order2.T
        data_order2 /= self.data_calspec
        spectrumrangeairmass.order2 = data_order2.T

        self.slope, self.ord, self.err_slope, self.err_ord = spectrumrangeairmass.bouguer_line()
        disp = np.loadtxt(self.rep_disp_name)
        Data_disp = sp.interpolate.interp1d(disp.T[0], disp.T[1], kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        Err_disp = sp.interpolate.interp1d(disp.T[0], disp.T[2], kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        tel = np.loadtxt(self.rep_tel_name)
        Data_tel = sp.interpolate.interp1d(tel.T[0], tel.T[1], kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        Err_tel = sp.interpolate.interp1d(tel.T[0], tel.T[2], kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        """
        data_disp_new_lambda = Data_disp(self.new_lambda)
        err_disp_new_lambda = Err_disp(self.new_lambda)
        data_tel_new_lambda = Data_tel(self.new_lambda)
        err_tel_new_lambda = Err_tel(self.new_lambda)
        """
        self.data_bouguer = np.exp(self.ord)
        err_bouguer = self.err_ord * self.data_bouguer
        self.data = self.data_bouguer
        self.lambdas = self.new_lambda
        self.err = err_bouguer
        #self.data = filter_detect_lines(self.lambdas, self.data, self.plot_filt, self.save_filter)

        Data = sp.interpolate.interp1d(self.lambdas, self.data, kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        Data_bouguer = sp.interpolate.interp1d(self.lambdas, self.data_bouguer, kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        Err = sp.interpolate.interp1d(self.lambdas, self.err, kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")
        self.lambdas = np.arange(self.lambda_min, self.lambda_max, 1)

        self.data_tel = Data_tel(self.lambdas)
        self.data_tel_err = Err_tel(self.lambdas)
        self.data_disp = Data_disp(self.lambdas)
        self.data_disp_err = Err_disp(self.lambdas)
        self.data = Data(self.lambdas)
        self.err = Err(self.lambdas)
        self.data_bouguer = Data_bouguer(self.lambdas)
        #self.data = sp.signal.savgol_filter(self.data, 111, 2)

        if self.order2 and not self.mega_fit:
            self.slope2, self.ord2, self.err_slope2, self.err_ord2, self.A2, self.A2_err = spectrumrangeairmass.bouguer_line_order2()
            self.data_bouguer = np.exp(self.ord2)
            err_bouguer = self.err_ord2 * self.data_bouguer
            self.data_order2 = self.data_bouguer
            self.lambdas = self.new_lambda
            self.err_order2 = err_bouguer
            # self.data_order2 = filter_detect_lines(self.lambdas, self.data_order2, self.plot_filt, self.save_filter)
            Data = sp.interpolate.interp1d(self.lambdas, self.data_order2, kind="linear", bounds_error=False,
                                           fill_value="extrapolate")
            Data_bouguer = sp.interpolate.interp1d(self.lambdas, self.data_bouguer, kind="linear", bounds_error=False,
                                                   fill_value="extrapolate")
            Err = sp.interpolate.interp1d(self.lambdas, self.err_order2, kind="linear", bounds_error=False,
                                          fill_value="extrapolate")
            self.lambdas = np.arange(self.lambda_min, self.lambda_max, 1)
            self.data_order2 = Data(self.lambdas)
            self.err_order2 = Err(self.lambdas)
            self.data_bouguer = Data_bouguer(self.lambdas)
            # self.data_order2 = sp.signal.savgol_filter(self.data_order2, 111, 2)

        if self.mega_fit:
            self.ord2, self.err_ord2 = spectrumrangeairmass.megafit_emcee()
            print(self.ord2)
            print(len(self.ord2))
            self.data_order2 = self.ord2
            self.lambdas = self.new_lambda
            # self.data_order2 = filter_detect_lines(self.lambdas, self.data_order2, self.plot_filt, self.save_filter)
            Data = sp.interpolate.interp1d(self.lambdas, self.data_order2, kind="linear", bounds_error=False,
                                           fill_value="extrapolate")
            """
            Data_bouguer = sp.interpolate.interp1d(self.lambdas, self.data_bouguer, kind="linear", bounds_error=False,
                                                   fill_value="extrapolate")
            Err = sp.interpolate.interp1d(self.lambdas, self.err_order2, kind="linear", bounds_error=False,
                                          fill_value="extrapolate")
            """
            self.lambdas = np.arange(self.lambda_min, self.lambda_max, 1)
            self.data_order2 = Data(self.lambdas)
            self.err_order2 = Err(self.lambdas)
            self.data_bouguer = Data_bouguer(self.lambdas)
            # self.data_order2 = sp.signal.savgol_filter(self.data_order2, 111, 2)



def convert_from_flam_to_mag(data,err):

    for i in range(len(data)):
        if data[i]<1e-15:
            data[i] = 1e-15

    return np.log(data), err/data

def filter_spec(lambdas, data):

    """
    Data = sp.interpolate.interp1d(lambdas, data, kind="linear", bounds_error=False,
                                           fill_value=(0, 0))
    lambdas2 = np.arange(parameters.LAMBDA_MIN-100, parameters.LAMBDA_MAX+100, 1)
    data_savgol = sp.signal.savgol_filter(Data(lambdas2), parameters.SAVGOL_LENGTH, parameters.SAVGOL_ORDER)  # filtre savgol (enlève le bruit)
    data_filt = smooth(data_savgol, parameters.SMOOTH_LENGTH, parameters.SMOOTH_WINDOW, sigma=parameters.SMOOTH_SIGMA)
    Data = sp.interpolate.interp1d(lambdas2, data_filt, kind="linear", bounds_error=False,
                                   fill_value=(0, 0))
    data_filt = Data(lambdas)
    """

    intensite_reel = data
    lambda_reel = lambdas

    filtre3_window = 11
    filtre3_order = 3
    filtre3avg_window = 50
    lambda_mid = 475
    filtre2avg_window = 10
    filtre2_window = 61
    filtre2_order = 3

    intensite_reel_savgol = sp.signal.savgol_filter(intensite_reel, filtre3_window,
                                                    filtre3_order)  # filtre savgol (enlève le bruit)
    intensite_reel_moy = smooth(intensite_reel, filtre3avg_window, 'flat', 1)

    intensite_reel_1 = (intensite_reel_savgol + intensite_reel_moy) / 2

    j = 0
    for i in range(len(intensite_reel_1)):
        if lambda_reel[i] > lambda_mid:
            j = i
            break

    intensite_reel_1[j:] = intensite_reel_savgol[j:]

    intensite_reel_1 = sp.interpolate.interp1d(lambda_reel, intensite_reel_1, kind='cubic')

    lambda_complet = np.linspace(lambda_reel[0], lambda_reel[-1],
                                 int((lambda_reel[-1] - lambda_reel[0]) * 10 + 1))  # précision Angtrom

    Intensite_reel = intensite_reel_1(lambda_complet)

    intensite_tronque = [Intensite_reel[0]]
    lambda_tronque = [lambda_complet[0]]
    c = 0
    for i in range(1, len(lambda_complet) - 1):
        if (Intensite_reel[i + 1] - Intensite_reel[i - 1]) / (
                lambda_complet[i + 1] - lambda_complet[i - 1]) > 0:
            c = 1

        elif c == 1 and (Intensite_reel[i + 1] - Intensite_reel[i - 1]) / (
                lambda_complet[i + 1] - lambda_complet[i - 1]) < 0:
            intensite_tronque.append(Intensite_reel[i])
            lambda_tronque.append(lambda_complet[i])
            c = 0

    intensite_tronque.append(Intensite_reel[-1])
    lambda_tronque.append(lambda_complet[-1])

    for j in range(100):
        intensite_tronque2 = [Intensite_reel[0]]
        lambda_tronque2 = [lambda_complet[0]]
        c = 0
        for i in range(1, len(lambda_tronque) - 1):
            if intensite_tronque[i - 1] < intensite_tronque[i] or intensite_tronque[i + 1] < \
                    intensite_tronque[i]:
                intensite_tronque2.append(intensite_tronque[i])
                lambda_tronque2.append(lambda_tronque[i])

        intensite_tronque2.append(Intensite_reel[-1])
        lambda_tronque2.append(lambda_complet[-1])

        intensite_tronque = intensite_tronque2
        lambda_tronque = lambda_tronque2

    Intensite_reels = sp.interpolate.interp1d(lambda_tronque, intensite_tronque, bounds_error=False,
                                              fill_value="extrapolate")
    Intensite_reel = Intensite_reels(lambda_complet)
    INTENSITE_reel = smooth(Intensite_reel, filtre2avg_window, 'flat', 1)
    INTENSITE_reelS = sp.signal.savgol_filter(INTENSITE_reel, filtre2_window, filtre2_order)

    interpolation_reel = sp.interpolate.interp1d(lambda_complet, INTENSITE_reelS)
    data_filt = interpolation_reel(lambdas)

    return data_filt

def filter_detect_lines(lambdas, data, plot=False, save=False):

    intensite_obs = data
    lambda_obs = lambdas

    intensite_obs_savgol = sp.signal.savgol_filter(intensite_obs, parameters.SAVGOL_LENGTH_DL, parameters.SAVGOL_ORDER_DL)  # filtre savgol (enlève le bruit)
    # filtre moy 2
    # entre 2 pts, 1.4 nm

    intensite_obs_savgol1 = sp.interpolate.interp1d(lambda_obs, intensite_obs_savgol, kind='quadratic')

    for i in range(len(lambda_obs)):
        # début du filtre "global"
        if intensite_obs[i] > parameters.START_FILTER * max(intensite_obs):
            debut_filtre_global = lambda_obs[i]
            break
    k = np.argmin(np.array(lambda_obs) - debut_filtre_global)

    moy_raies = 1
    intensite_obs_savgol2 = sp.signal.savgol_filter(intensite_obs[k:], parameters.SAVGOL_LENGTH_GLOB, parameters.SAVGOL_ORDER_GLOB)

    intensite_obs_sagol_3 = sp.interpolate.interp1d(lambda_obs[k:], intensite_obs_savgol2, kind='quadratic',
                                                    bounds_error=False, fill_value="extrapolate")

    lambda_complet = np.linspace(lambda_obs[0], lambda_obs[-1],
                                 int((lambda_obs[-1] - lambda_obs[0]) + 1))  # précison Angtrom

    intensite_obss = intensite_obs_savgol

    D_intensite_obss = [(intensite_obss[1] - intensite_obss[0]) / (lambda_obs[1] - lambda_obs[0])]
    for i in range(1, len(intensite_obss) - 1):
        D_intensite_obss.append(
            (intensite_obss[i + 1] - intensite_obss[i - 1]) / (lambda_obs[i + 1] - lambda_obs[i - 1]))

    D_intensite_obss.append(0)

    intensite_derivee = sp.interpolate.interp1d(lambda_obs, D_intensite_obss)

    intensite_obss = intensite_obs_savgol1(lambda_complet)

    D_intensite_obss = intensite_derivee(lambda_complet)

    D_mean = misc.derivative(intensite_obs_sagol_3, lambda_complet[moy_raies:-moy_raies])

    S = np.std(D_mean[:moy_raies * parameters.STD_LENGTH])
    D_sigma = []
    for i in range(moy_raies * parameters.STD_LENGTH):
        D_sigma.append(S)

    for i in range(moy_raies * parameters.STD_LENGTH, len(D_mean) - moy_raies * parameters.STD_LENGTH):
        D_sigma.append(np.std(D_mean[i - moy_raies * parameters.STD_LENGTH:i + moy_raies * parameters.STD_LENGTH]))

    for i in range(len(D_mean) - moy_raies * parameters.STD_LENGTH, len(D_mean)):
        D_sigma.append(np.std(D_mean[-moy_raies * parameters.STD_LENGTH:]))

    Raies = [False]

    i = moy_raies
    while i < len(D_intensite_obss) - moy_raies:
        var_signe = 0
        Raies.append(False)

        if D_intensite_obss[i] < D_mean[i - moy_raies] - parameters.TRIGGER * D_sigma[i - moy_raies]:

            k = i
            while lambda_complet[k] - lambda_complet[i] < parameters.HALF_LENGTH_MAX and k < len(lambda_complet) - moy_raies:
                k += 1

            for j in range(i, k):
                if D_intensite_obss[j + 1] - D_intensite_obss[j] > 0 and var_signe == 0:
                    var_signe = 1

                if var_signe == 1 and D_intensite_obss[j + 1] - D_intensite_obss[j] < 0:
                    var_signe = 2

                if var_signe == 2 and D_intensite_obss[j + 1] - D_intensite_obss[j] > 0:
                    var_signe = 3

                if var_signe == 3 and D_intensite_obss[j + 1] - D_intensite_obss[j] < 0:
                    break

                if D_intensite_obss[j] > D_mean[j - moy_raies] + parameters.TRIGGER * D_sigma[j - moy_raies]:

                    if len(lambda_complet) - moy_raies > j + k - i:
                        indice = j + k - i
                    else:
                        indice = len(lambda_complet) - moy_raies
                    for v in range(j, indice):

                        if D_intensite_obss[v + 1] - D_intensite_obss[v] < 0:
                            if var_signe == 1:
                                var_signe = 2

                            if var_signe == 3:
                                var_signe = 4

                        if D_intensite_obss[v + 1] - D_intensite_obss[v] > 0:
                            if var_signe == 2:
                                var_signe = 3

                            if var_signe == 4:
                                break

                        if D_intensite_obss[v] < D_mean[v - moy_raies] + parameters.TRIGGER * D_sigma[v - moy_raies]:
                            indice = v
                            break
                    parameters.AROUND_LINES = 4
                    if indice != j + k - i and indice != len(lambda_complet) - 1:
                        if var_signe == 2 or var_signe == 4:
                            for loop in range(i + 1, indice + parameters.AROUND_LINES + 1):
                                Raies.append(True)
                            for loop in range(i - parameters.AROUND_LINES - 1, i + 1):
                                Raies[i] = True
                            i = indice + 1 + parameters.AROUND_LINES
                            Raies.append(False)
                    break
        i += 1

    intensite_coupe_obs = []
    lambda_coupe_obs = []
    D_intensite_coupe = []

    if len(Raies) < len(lambda_complet):
        for j in range(len(Raies), len(lambda_complet)):
            Raies.append(False)

    if len(Raies) < len(lambda_complet):
        for j in range(len(Raies), len(lambda_complet)):
            Raies.append(False)

    for i in range(parameters.AROUND_LINES, len(Raies)-parameters.AROUND_LINES-1):
        if Raies[i] == False:
            stop = 0
            for j in range(i - parameters.AROUND_LINES, i + 1):
                if Raies[j] == True:
                    stop = 1
            if stop == 1:
                for j in range(i + 1, i + parameters.AROUND_LINES + 1):
                    if Raies[j] == True:
                        stop = 2
            if stop == 2:
                Raies[i] = True

    for i in range(len(lambda_complet)):
        if Raies[i] == False or lambda_complet[i] < parameters.LEFT or lambda_complet[i] > parameters.RIGHT:
            if lambda_complet[i] > parameters.END_WATER or lambda_complet[i] < parameters.START_WATER:
                intensite_coupe_obs.append(intensite_obss[i])
                lambda_coupe_obs.append(lambda_complet[i])
                D_intensite_coupe.append(D_intensite_obss[i])

    intensite_obsSpline = sp.interpolate.interp1d(lambda_coupe_obs, intensite_coupe_obs, bounds_error=False,
                                                  fill_value="extrapolate")
    intensite_obsSpline2 = sp.interpolate.interp1d(lambda_coupe_obs, intensite_coupe_obs, kind='quadratic',
                                                   bounds_error=False, fill_value="extrapolate")
    INTENSITE_OBSS = (np.array(intensite_obsSpline(lambda_obs)) + np.array(intensite_obsSpline2(lambda_obs))) / 2

    data_filt = INTENSITE_OBSS

    if plot:
        plot_filter(save)

    return data_filt

def smooth(x,window_len,window,sigma=1):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    elif window == 'gaussian':
        if sigma==0:
            return x
        else:
            w=signal.gaussian(window_len,sigma)
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    if window_len%2==0: # even case
        y=y[int(window_len/2):-int(window_len/2)+1]
        return y
    else:           #odd case
        y=y[int(window_len/2-1)+1:-int(window_len/2-1)-1]
        return y

def plot_atmosphere(Throughput, save_atmo, sim):

    fig = plt.figure(figsize=[15, 10])
    ax = fig.add_subplot(111)

    T = Throughput
    Lambda = np.arange(T.new_lambda[0], T.new_lambda[-1], 1)

    atmgrid = AtmosphereGrid(filename="tests/data/reduc_20170530_060_atmsim.fits")
    if sim:
        b = atmgrid.simulate(300, 5, 0.03)
        def fatmosphere(lambdas, ozone, eau, aerosol):
            return np.exp(np.log(atmgrid.simulate(ozone, eau, aerosol)(lambdas)) / 1.047)

        popt, pcov = sp.optimize.curve_fit(fatmosphere, T.new_lambda, np.exp(T.slope), p0=[300, 5, 0.03], sigma=T.err_slope * np.exp(T.slope),
                                           bounds=([0,0,0], [700,20,0.5]), verbose=2)
        print(popt[0], np.sqrt(pcov[0][0]))
        print(popt[1], np.sqrt(pcov[1][1]))
        print(popt[2], np.sqrt(pcov[2][2]))
        c = atmgrid.simulate(*popt)
        plt.plot(Lambda, np.exp(np.log(b(Lambda)) / 1.047), color='blue', label='transmission libradtran typique')
    else:
        def fatmosphere(lambdas, ozone, eau, aerosol):
            return np.exp(np.log(atmgrid.simulate(ozone, eau, aerosol)(lambdas)) / 1.047)

        popt, pcov = sp.optimize.curve_fit(fatmosphere, T.new_lambda, np.exp(T.slope), p0=[300, 5, 0.03], sigma=T.err_slope * np.exp(T.slope),
                                           bounds=([0,0,0], [700,20,0.5]), verbose=2)
        print(popt[0], np.sqrt(pcov[0][0]))
        print(popt[1], np.sqrt(pcov[1][1]))
        print(popt[2], np.sqrt(pcov[2][2]))
        c= atmgrid.simulate(*popt)

    plt.plot(Lambda, np.exp(np.log(c(Lambda)) / 1.047), color='black', label='transmission libradtran ajustée')
    plt.plot(T.new_lambda, np.exp(T.slope), color='red', label='transmission atmosphérique droites bouguer')
    plt.errorbar(T.new_lambda, np.exp(T.slope), xerr=None, yerr=T.err_slope * np.exp(T.slope), fmt='none', capsize=1,
                 ecolor='red', zorder=2, elinewidth=2)
    plt.fill_between(T.new_lambda, np.exp(T.slope) + T.err_slope * np.exp(T.slope),
                     np.exp(T.slope) - T.err_slope * np.exp(T.slope), color='red')

    if T.order2 :
        plt.plot(T.new_lambda, np.exp(T.slope2), color='green', label='transmission atmosphérique correction ordre2')
        plt.errorbar(T.new_lambda, np.exp(T.slope2), xerr=None, yerr=T.err_slope2 * np.exp(T.slope2), fmt='none',
                     capsize=1,
                     ecolor='green', zorder=2, elinewidth=2)
        plt.fill_between(T.new_lambda, np.exp(T.slope2) + T.err_slope2 * np.exp(T.slope2),
                         np.exp(T.slope2) - T.err_slope2 * np.exp(T.slope2), color='green')

    if T.sim:
        plt.title('Transmission atmosphérique, simulation, ' + T.disperseur +', '+parameters.PROD_NAME, fontsize=18)
    else:
        plt.title('Transmission atmosphérique, données, ' + T.disperseur +', '+parameters.PROD_NAME, fontsize=18)

    ax.get_xaxis().set_tick_params(labelsize=17)
    ax.get_yaxis().set_tick_params(labelsize=14)
    plt.xlabel('$\lambda$ (nm)', fontsize=17)
    plt.ylabel('Transmission atmosphérique', fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 15}, loc='lower right')

    if save_atmo:
        if T.sim:
            if os.path.exists(parameters.OUTPUTS_ATM_SIM):
                plt.savefig(parameters.OUTPUTS_ATM_SIM+'atm_simu, ' + T.disperseur +', '+parameters.PROD.split('_')[-4]+'.png')
            else:
                os.makedirs(parameters.OUTPUTS_ATM_SIM)
                plt.savefig(parameters.OUTPUTS_ATM_SIM + 'atm_simu, ' + T.disperseur + ', ' +parameters.PROD.split('_')[-4] + '.png')
        else:
            if os.path.exists(parameters.OUTPUTS_ATM_REDUC):
                plt.savefig(
                    parameters.OUTPUTS_ATM_REDUC + 'atm_reduc, ' + T.disperseur + ', ' + parameters.PROD.split('_')[-4] + '.png')
            else:
                os.makedirs(parameters.OUTPUTS_ATM_REDUC)
                plt.savefig(
                    parameters.OUTPUTS_ATM_REDUC + 'atm_reduc, ' + T.disperseur + ', ' +parameters.PROD.split('_')[-4] + '.png')

    plt.show()

def plot_bouguer_lines(spectrumrangeairmass, Throughput, save_bouguer):
    fig = plt.figure(figsize=[15, 10])
    ax = fig.add_subplot(111)

    def flin(x, a, b):
        return a * x + b

    def forder2(x, a, b, A2):
        return a * x + b + np.log(1 + A2 * t_disperseur * np.exp((a_p - a) * x + b_p - b))

    S = spectrumrangeairmass
    T = Throughput
    Z = np.linspace(0,2.2,1000)

    for i in range(len(T.new_lambda)):
        MAG = flin(Z, T.slope[i], T.ord[i])
        MAG_sup = flin(Z, T.slope[i] + T.err_slope[i], T.ord[i] - T.err_ord[i])
        MAG_inf = flin(Z, T.slope[i] - T.err_slope[i], T.ord[i] + T.err_ord[i])
        if i % 10 == 0:
            plt.plot(Z, MAG, c=wavelength_to_rgb(T.new_lambda[i]))
            plt.plot(Z, MAG_sup, c=wavelength_to_rgb(T.new_lambda[i]), linestyle=':')
            plt.plot(Z, MAG_inf, c=wavelength_to_rgb(T.new_lambda[i]), linestyle=':')

            plt.fill_between(Z, MAG_sup, MAG_inf, color=[wavelength_to_rgb(T.new_lambda[i])])
            "la commande ci-dessus grise donne la bonne couleur la zone où se trouve la droite de Bouguer"

            plt.scatter(S.range_airmass[i], S.data_mag[i], c=[wavelength_to_rgb(T.new_lambda[i])], label=f'{T.Bin[i]}-{T.Bin[i + 1]} nm',
                        marker='o', s=30)
            plt.errorbar(S.range_airmass[i], S.data_mag[i], xerr=None, yerr=S.err_mag[i], fmt='none', capsize=1,
                         ecolor=(wavelength_to_rgb(T.new_lambda[i])), zorder=2, elinewidth=2)
        """
        if T.order2:
            t_disperseur = T.t_disp_order2[i]
            if i % 2 == 0:
                a_p = T.slope[int(i / 2)]
                b_p = T.ord[int(i / 2)]
            else:
                a_p = (T.slope[int(i / 2)] + T.slope[int(i / 2) + 1]) / 2
                b_p = (T.ord[int(i / 2)] + T.ord[int(i / 2) + 1]) / 2
            MAG = forder2(Z, T.slope2[i], T.ord2[i], T.A2[i])
            if i % 10 == 0:
                plt.plot(Z, MAG, c=wavelength_to_rgb(T.new_lambda[i]), linestyle='--')
        """
    if T.sim:
        plt.title('Droites de Bouguer, simulation, ' + T.disperseur+', '+parameters.PROD.split('_')[-4], fontsize=18)
    else:
        plt.title('Droites de Bouguer, données, ' + T.disperseur+', '+parameters.PROD.split('_')[-4], fontsize=18)

    ax.get_xaxis().set_tick_params(labelsize=17)
    ax.get_yaxis().set_tick_params(labelsize=14)
    plt.xlabel('$\lambda$ (nm)', fontsize=17)
    plt.ylabel('ln(flux)', fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 12}, loc='upper right')

    if save_bouguer:
        if T.sim:
            if os.path.exists(parameters.OUTPUTS_BOUGUER_SIM):
                plt.savefig(parameters.OUTPUTS_BOUGUER_SIM+'bouguer_simu, ' + T.disperseur +', '+parameters.PROD.split('_')[-4]+'.png')
            else:
                os.makedirs(parameters.OUTPUTS_BOUGUER_SIM)
                plt.savefig(parameters.OUTPUTS_BOUGUER_SIM + 'bouguer_simu, ' + T.disperseur + ', ' +parameters.PROD.split('_')[-4]+ '.png')
        else:
            if os.path.exists(parameters.OUTPUTS_BOUGUER_REDUC):
                plt.savefig(
                    parameters.OUTPUTS_BOUGUER_REDUC + 'bouguer_reduc, ' + T.disperseur + ', ' +parameters.PROD.split('_')[-4]+ '.png')
            else:
                os.makedirs(parameters.OUTPUTS_BOUGUER_REDUC)
                plt.savefig(parameters.OUTPUTS_BOUGUER_REDUC + 'bouguer_reduc, ' + T.disperseur + ', ' +parameters.PROD.split('_')[-4]+ '.png')
    plt.show()

def plot_spectrums(s):
    plt.figure(figsize=[10, 10])
    plt.plot(s.lambdas, s.data, c='black')
    plt.errorbar(s.lambdas, s.data, xerr=None, yerr=s.err, fmt='none', capsize=1, ecolor='black', zorder=2,
                   elinewidth=2)
    plt.xlabel('$\lambda$ (nm)',fontsize=13)
    plt.ylabel('erg/s/cm2/Hz', fontsize=13)
    plt.title('spectra: '+s.tag[:-13]+' with '+parameters.PROD.split('_')[-4] +' of '+s.target, fontsize=16)
    plt.grid(True)
    plt.show()

def plot_spec_target(Throughput, save_target):
    plt.figure(figsize=[10, 10])
    plt.plot(Throughput.lambdas_calspec, Throughput.data_calspec_org, c='red', label='CALSPEC')
    plt.plot(Throughput.new_lambda, Throughput.data_calspec, c='black', label='CALSPEC filtered')
    plt.plot(Throughput.lambdas, Throughput.data_bouguer / (Throughput.data_disp * Throughput.data_tel), c='blue', label='CALSPEC exact')
    plt.axis([Throughput.lambda_min,Throughput.lambda_max,0,max(Throughput.data_calspec_org)*1.1])
    plt.xlabel('$\lambda$ (nm)', fontsize=13)
    plt.ylabel('erg/s/cm2/Hz', fontsize=13)
    plt.title('spectra CALSPEC: ' + Throughput.target, fontsize=16)
    plt.grid(True)
    plt.legend(prop={'size': 12}, loc='upper right')

    if save_target:
        if os.path.exists(parameters.OUTPUTS_TARGET):
            plt.savefig(parameters.OUTPUTS_TARGET+'CALSPEC, ' + Throughput.target+'.png')
        else:
            os.makedirs(parameters.OUTPUTS_TARGET)
            plt.savefig(parameters.OUTPUTS_TARGET+'CALSPEC, ' + Throughput.target+'.png')

    plt.show()


def plot_filter(save_filter):
    return 1

def plot_throughput_sim(Throughput, save_Throughput):
    T = Throughput

    gs_kw = dict(height_ratios=[4, 1], width_ratios=[1])
    fig, ax = plt.subplots(2, 1, sharex="all", figsize=[14, 12], constrained_layout=True, gridspec_kw=gs_kw)

    ax[0].scatter(T.lambdas, T.data / T.data_tel, c='black', label='T_disp Vincent', s=15, zorder=2)
    ax[0].errorbar(T.lambdas, T.data / T.data_tel, xerr=None, yerr=T.err / T.data_tel, fmt='none', capsize=1, ecolor='black', zorder=2,
                       elinewidth=2)

    ax[0].scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='T_disp exacte')
    ax[0].errorbar(T.lambdas, T.data_disp, xerr=None, yerr=T.data_disp_err, fmt='none', capsize=1, ecolor='deepskyblue', zorder=1,
                       elinewidth=2)

    if T.order2:
        ax[0].scatter(T.lambdas, T.data_order2 / T.data_tel, c='red', label='T_disp Vincent ordre2', s=15, zorder=2)
        ax[0].errorbar(T.lambdas, T.data_order2 / T.data_tel, xerr=None, yerr=T.err_order2 / T.data_tel, fmt='none', capsize=1,
                   ecolor='red', zorder=2,elinewidth=2)

    ax[0].set_xlabel('$\lambda$ (nm)', fontsize=20)
    ax[0].set_ylabel('Transmission instrumentale', fontsize=20)
    ax[0].set_title('Transmission du '+ T.disperseur +', '+parameters.PROD_NAME, fontsize=18)
    ax[0].get_xaxis().set_tick_params(labelsize=17)
    ax[0].get_yaxis().set_tick_params(labelsize=14)
    ax[0].grid(True)
    ax[0].legend(prop={'size': 22}, loc='upper right')

    """On cherche les points de la reponse ideale (celle de Sylvie) les plus proches des longueurs d'ondes de la rep
    simulee"""

    "Tableaux avec les ecarts relatifs"

    Rep_sim_norm = (T.data / (T.data_tel * T.data_disp)  - 1) * 100
    if T.order2:
        Rep_sim_norm_bis = (T.data_order2 / (T.data_tel * T.data_disp)  - 1) * 100
    zero = np.zeros(1000)

    X_2 = 0
    if T.order2:
        X_2_bis = 0
    for i in range(len(T.lambdas)):
        X_2 += Rep_sim_norm[i] ** 2
        if T.order2:
            X_2_bis += Rep_sim_norm_bis[i] ** 2

    X_2 = np.sqrt(X_2 / len(Rep_sim_norm))  # correspond au sigma
    if T.order2:
        X_2_bis = np.sqrt(X_2_bis / len(Rep_sim_norm_bis))

    ax[1].plot(np.linspace(T.lambdas[0], T.lambdas[-1], 1000), zero, c='black')

    NewErr = T.err / (T.data_tel * T.data_disp) * 100
    if T.order2:
        NewErr_bis = T.err_order2 / (T.data_tel * T.data_disp) * 100

    ax[1].scatter(T.lambdas, Rep_sim_norm, c='black', marker='o')
    ax[1].errorbar(T.lambdas, Rep_sim_norm, xerr=None, yerr=NewErr, fmt='none', capsize=1,
                   ecolor='black', zorder=2, elinewidth=2)

    if T.order2:
        ax[1].scatter(T.lambdas, Rep_sim_norm_bis, c='red', marker='o')
        ax[1].errorbar(T.lambdas, Rep_sim_norm_bis, xerr=None, yerr=NewErr_bis, fmt='none', capsize=1,
                       ecolor='red', zorder=2, elinewidth=2)

    ax[1].set_xlabel('$\lambda$ (nm)', fontsize=17)
    ax[1].set_ylabel('Ecart relatif (%)', fontsize=15)
    ax[1].get_xaxis().set_tick_params(labelsize=18)
    ax[1].get_yaxis().set_tick_params(labelsize=10)

    ax[1].grid(True)

    ax[1].yaxis.set_ticks(range(int(min(Rep_sim_norm)) - 2, int(max(Rep_sim_norm)) + 4,
                                    (int(max(Rep_sim_norm)) + 6 - int(min(Rep_sim_norm))) // 8))

    ax[1].text(550, max(Rep_sim_norm) * 3 / 4, '$\sigma$= ' + str(X_2)[:4] + '%', color='black', fontsize=20)
    if T.order2:
        ax[1].text(850, max(Rep_sim_norm) * 3 / 4, '$\sigma$= ' + str(X_2_bis)[:4] + '%', color='red', fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_Throughput:
        if os.path.exists(parameters.OUTPUTS_THROUGHPUT_SIM):
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_SIM+'throughput_sim, '+ T.disperseur +', '+parameters.PROD.split('_')[-4]+'.png')
        else:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_SIM)
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_SIM+'throughput_sim, '+ T.disperseur +', '+parameters.PROD.split('_')[-4]+'.png')

    plt.show()

def plot_throughput_reduc(Throughput, save_Throughput):
    T = Throughput
    if T.disperseur == 'Thor300':
        fig = plt.figure(figsize=[15, 10])
        ax2 = fig.add_subplot(111)

        thorlab = np.loadtxt(T.rep_disp_ref)

        thorlab_data = sp.interpolate.interp1d(thorlab.T[0], thorlab.T[1], bounds_error=False, fill_value="extrapolate")(T.lambdas)

        Tinst = sp.signal.savgol_filter(T.data / thorlab_data, 81, 3)
        ax2.scatter(T.lambdas, Tinst, c='black', label='rep tel')
        ax2.errorbar(T.lambdas,  Tinst, xerr=None, yerr=T.err / thorlab_data, fmt='none', capsize=1,
                       ecolor='black', zorder=2, elinewidth=2)
        if T.order2:
            Tinst_order2 = sp.signal.savgol_filter(T.data_order2 / thorlab_data, 81, 3)
            Tinst_order2_err = sp.signal.savgol_filter(T.err_order2 / thorlab_data, 81, 3)
            ax2.scatter(T.lambdas, Tinst_order2, c='red', label='rep tel en enlevant ordre2')
            ax2.errorbar(T.lambdas, Tinst_order2, xerr=None, yerr=Tinst_order2_err, fmt='none', capsize=1,
                       ecolor='red', zorder=2, elinewidth=2)
        ax2.scatter(T.lambdas, T.data_tel, c='blue', label='rep CTIO Sylvie')
        ax2.errorbar(T.lambdas, T.data_tel, xerr=None, yerr=T.data_tel_err, fmt='none', capsize=1,
                       ecolor='blue', zorder=2, elinewidth=2)

        ax2.set_xlabel('$\lambda$ (nm)', fontsize=24)
        ax2.set_ylabel("Transmission telescope", fontsize=22)
        ax2.set_title("Transmission instrumentale du telescope, "+T.disperseur+', '+parameters.PROD.split('_')[-4] , fontsize=22)
        ax2.get_xaxis().set_tick_params(labelsize=20)
        ax2.get_yaxis().set_tick_params(labelsize=20)
        ax2.legend(prop={'size': 17}, loc='upper right')
        plt.grid(True)
        fig.tight_layout()
        if save_Throughput:
            if os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC):
                plt.savefig(
                    parameters.OUTPUTS_THROUGHPUT_REDUC + 'ctio_throughput, ' +parameters.PROD.split('_')[-4] + '.png')
                fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'ctio_thrpoughput_basethor300'), 'w')

                for i in range(len(T.lambdas)):
                    fichier.write(
                        str(T.lambdas[i]) + '\t' + str(Tinst_order2[i]) + '\t' + str(Tinst_order2_err[i]) + '\n')
                fichier.close()
            else:
                os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
                fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'ctio_throughput_basethor300.txt'), 'w')

                for i in range(len(T.lambdas)):
                    fichier.write(str(T.lambdas[i]) + '\t' + str(Tinst_order2[i]) + '\t' + str(Tinst_order2_err[i]) + '\n')
                fichier.close()
                plt.savefig(
                    parameters.OUTPUTS_THROUGHPUT_REDUC + 'ctio_throughput, ' +parameters.PROD.split('_')[-4] + '.png')
        plt.show()

    fig = plt.figure(figsize=[15, 10])
    ax2 = fig.add_subplot(111)

    ax2.scatter(T.lambdas, T.data / T.data_tel, c='black', marker='.', label='Tinst_Vincent')
    ax2.errorbar(T.lambdas, T.data / T.data_tel, xerr=None, yerr=T.err / T.data_tel, fmt='none', capsize=1,
                    ecolor='black',zorder=1, elinewidth=2)

    if T.disperseur == 'Thor300':
        T.data_disp = thorlab_data
        T.data_disp_err = thorlab_data*0.01
        ax2.scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='Banc_LPNHE')
    else:
        ax2.scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='Tinst_Sylvie')
    ax2.errorbar(T.lambdas, T.data_disp, xerr=None, yerr=T.data_disp_err, fmt='none', capsize=1, ecolor='deepskyblue', zorder=1,
                     elinewidth=2)

    if T.order2:
        ax2.scatter(T.lambdas, T.data_order2 / T.data_tel, c='red', marker='.', label='Tinst_Vincent_ordre2')
        ax2.errorbar(T.lambdas, T.data_order2 / T.data_tel, xerr=None, yerr=T.err_order2 / T.data_tel,
        fmt = 'none', capsize = 1, ecolor = 'red',zorder = 1, elinewidth = 2)

    ax2.set_xlabel('$\lambda$ (nm)', fontsize=24)
    ax2.set_ylabel("Transmission disperseur", fontsize=22)
    ax2.set_title("Transmission instrumentale du, " + T.disperseur + ', ' + parameters.PROD_NAME,
                      fontsize=22)
    ax2.get_xaxis().set_tick_params(labelsize=20)
    ax2.get_yaxis().set_tick_params(labelsize=20)
    ax2.legend(prop={'size': 17}, loc='upper right')
    plt.grid(True)
    fig.tight_layout()

    if save_Throughput:
        if os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC):
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_REDUC+'throughput_reduc, '+ T.disperseur +', '+parameters.PROD.split('_')[-4]+'.png')
        else:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_REDUC+'throughput_reduc, '+ T.disperseur +', '+parameters.PROD.split('_')[-4]+'.png')
    plt.show()

def convert_from_fits_to_txt(prod_name, prod_txt):

    to_convert_list = []
    Lsimutxt = glob.glob(prod_txt + "/sim*spectrum.txt")
    Lreductxt = glob.glob(prod_txt + "/reduc*spectrum.txt")
    Lsimufits = glob.glob(prod_name + "/sim*spectrum.fits")
    Lreducfits = glob.glob(prod_name + "/reduc*spectrum.fits")

    Ldefaut = glob.glob(prod_name + "/*bestfit*.txt") + glob.glob(prod_name + "/*fitting*.txt") + glob.glob(prod_name + "/*A2=0*.fits") + glob.glob(prod_name + "/*20170530_201_spectrum*")+ glob.glob(prod_name + "/*20170530_200_spectrum*")+ glob.glob(prod_name + "/*20170530_205_spectrum*")

    Lsimutxt = [i for i in Lsimutxt if i not in Ldefaut]
    Lreductxt = [i for i in Lreductxt if i not in Ldefaut]
    Lsimufits = [i for i in Lsimufits if i not in Ldefaut]
    Lreducfits = [i for i in Lreducfits if i not in Ldefaut]

    if len(Lsimutxt) != len(Lsimufits) or len(Lreductxt) != len(Lreducfits):
        for file in Lsimufits:
            tag = file.split('/')[-1]
            fichier = os.path.join(prod_txt, tag.replace('fits', 'txt'))
            if fichier not in Lsimutxt:
                to_convert_list.append(file)

        for file in Lreducfits:
            tag = file.split('/')[-1]
            fichier = os.path.join(prod_txt, tag.replace('fits', 'txt'))
            if fichier not in Lreductxt:
                to_convert_list.append(file)


        for i in range(len(to_convert_list)):
            startest = to_convert_list[i]
            s = Spectrum(startest)
            hdu = fits.open(startest)
            airmass = hdu[0].header["AIRMASS"]
            TARGETX = hdu[0].header["TARGETX"]
            TARGETY = hdu[0].header["TARGETY"]
            D2CCD = hdu[0].header["D2CCD"]
            PIXSHIFT = hdu[0].header["PIXSHIFT"]
            ROTANGLE = hdu[0].header["ROTANGLE"]
            psf_transverse = s.chromatic_psf.table['fwhm']
            PARANGLE = hdu[0].header["PARANGLE"]

            x0 = [TARGETX,TARGETY]
            disperser = s.disperser
            distance = disperser.grating_lambda_to_pixel(s.lambdas, x0=x0, order=1)
            distance -= adr_calib(s.lambdas, s.adr_params, parameterss.OBS_LATITUDE, lambda_ref=s.lambda_ref)
            distance += adr_calib(s.lambdas/2, s.adr_params, parameterss.OBS_LATITUDE, lambda_ref=s.lambda_ref)
            lambdas_order2 = disperser.grating_pixel_to_lambda(distance, x0=x0, order=2)

            print(to_convert_list[i][:len(to_convert_list[i]) - 5])
            disperseur = s.disperser_label
            star = s.header['TARGET']
            lambda_obs = s.lambdas
            intensite_obs = s.data
            intensite_err = s.err

            if s.target.wavelengths == []:
                print('CALSPEC error')

            else:
                lambda_reel = s.target.wavelengths[0]
                intensite_reel = s.target.spectra[0]
                tag = to_convert_list[i].split('/')[-1]
                fichier = open(os.path.join(prod_txt, tag.replace('fits','txt')), 'w')
                fichier.write('#' + '\t' + star + '\t' + disperseur + '\t' + str(airmass) + '\t' + str(
                    TARGETX) + '\t' + str(TARGETY) + '\t' + str(D2CCD) + '\t' + str(PIXSHIFT) + '\t' + str(
                    ROTANGLE) + '\t' + str(PARANGLE) + '\n')
                for j in range(len(lambda_reel)):
                    if len(lambda_obs) > j:
                        if len(psf_transverse) > j:
                            fichier.write(str(lambda_reel[j]) + '\t' + str(intensite_reel[j]) + '\t' + str(
                                        lambda_obs[j]) + '\t' + str(intensite_obs[j]) + '\t' + str(
                                        intensite_err[j]) + '\t' + str(lambdas_order2[j]) + '\t' + str(psf_transverse[j]) + '\n')
                        else:
                            fichier.write(str(lambda_reel[j]) + '\t' + str(intensite_reel[j]) + '\t' + str(
                                        lambda_obs[j]) + '\t' + str(intensite_obs[j]) + '\t' + str(
                                        intensite_err[j]) + '\t' + str(lambdas_order2[j]) + '\n')

                    else:
                        fichier.write(str(lambda_reel[j]) + '\t' + str(intensite_reel[j]) + '\n')

                fichier.close()
        return True
    else:
        print('already done')
        return True,Lsimutxt,Lreductxt

def extract_throughput(prod_name, sim, disperseur, prod_sim, prod_reduc, target="HD111980", order2=False, mega_fit=False,
                       plot_atmo=False, plot_bouguer=False, plot_specs=False, plot_target=False, plot_filt=False, plot_Throughput=True,
                       save_atmo=False, save_bouguer=False, save_target=False, save_Throughput=False, save_filter=False):

    spectrumrangeairmass = SpectrumRangeAirmass(prod_name=prod_name, sim=sim, disperseur=disperseur, target=target, plot_specs=plot_specs, prod_sim = prod_sim, prod_reduc = prod_reduc)
    Throughput = TransmissionInstrumentale(prod_name=prod_name, sim=sim, disperseur=disperseur, target=target, order2=order2, mega_fit=mega_fit, plot_filt=plot_filt, save_filter=save_filter, prod=prod_sim)
    Throughput.calcul_throughput(spectrumrangeairmass)

    if plot_atmo:
        plot_atmosphere(Throughput, save_atmo, sim)

    if plot_bouguer:
        plot_bouguer_lines(spectrumrangeairmass, Throughput, save_bouguer)

    if plot_target:
        plot_spec_target(Throughput, save_target)

    if plot_Throughput:
        if sim:
            plot_throughput_sim(Throughput, save_Throughput)
        else:
            plot_throughput_reduc(Throughput, save_Throughput)

def prod_analyse(prod_name, prod_txt):
    CFT = convert_from_fits_to_txt(prod_name, prod_txt)
    if CFT[0]:

        for disperser in parameters.DISPERSER:
            extract_throughput(prod_txt, True, disperser, CFT[1], CFT[2], plot_bouguer=False, plot_atmo=False, plot_target=False,save_atmo=False, save_bouguer=False, save_target=False, save_Throughput=False, order2=True)
        """
        for disperser in parameters.DISPERSER:
            extract_throughput(prod_txt, False, disperser, CFT[1], CFT[2], plot_bouguer=True, plot_atmo=True, plot_target=True,save_atmo=True, save_bouguer=False, save_target=False, save_Throughput=True, order2=True)
        """
    else:
        print('relaunch, convert_fits_to_txt step')

prod_txt = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_TXT)
prod_name = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME)
extract_throughput(prod_txt, True, 'Thor300', glob.glob(prod_txt + "/sim*spectrum.txt"), glob.glob(prod_txt + "/reduc*spectrum.txt"), plot_specs = False, plot_bouguer=False, plot_atmo=False, order2=True, mega_fit=True, save_Throughput=False, plot_Throughput=True)
#prod_analyse(prod_name, prod_txt)