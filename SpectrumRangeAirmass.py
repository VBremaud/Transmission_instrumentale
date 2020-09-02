# coding: utf8

import glob
import emcee
import multiprocessing
import sys
from schwimmbad import MPIPool
from SpectrumAirmassFixed import *
from spectractor.simulation.simulator import AtmosphereGrid
from numpy.linalg import inv
import matplotlib.colors
from scipy import integrate

class SpectrumRangeAirmass:

    def __init__(self, subtract_order2=False):

        self.target = parameters.target
        self.disperseur = parameters.DISP
        self.plot_specs = parameters.plot_specs
        self.binwidths = parameters.BINWIDTHS
        self.lambda_min = parameters.LAMBDA_MIN
        self.lambda_max = parameters.LAMBDA_MAX
        self.new_lambda = parameters.NEW_LAMBDA
        self.Bin = parameters.BIN
        self.sim = parameters.SIM
        self.list_spectrum = []
        self.data = []
        self.range_airmass = []
        self.err_mag = []
        self.order2 = []
        self.names = []
        self.cov = []
        self.INVCOV = []
        self.PSF_REG = []
        self.atm = None
        self.params_atmo = np.zeros(3)
        self.err_params_atmo = np.zeros(3)
        self.prod_sim = glob.glob(parameters.PROD_TXT + "/sim*spectrum.txt")
        self.prod_reduc = glob.glob(parameters.PROD_TXT + "/reduc*spectrum.txt")
        self.file_tdisp_order2 = None
        if subtract_order2:
            if self.sim:
                self.file_tdisp_order2 = os.path.join(parameters.THROUGHPUT_DIR, parameters.DISPERSER_ORDER2_SIM)
            else:
                self.file_tdisp_order2 = os.path.join(parameters.THROUGHPUT_DIR, parameters.DISPERSER_ORDER2)
        self.init_spectrumrangeairmass()
        self.data_range_airmass()
        self.check_outliers()
        self.invcov()

    def init_spectrumrangeairmass(self):
        if self.sim:
            self.list_spectrum = self.prod_sim
        else:
            self.list_spectrum = self.prod_reduc

        for j in range(len(self.Bin) - 1):
            self.data.append([])
            self.range_airmass.append([])
            self.err_mag.append([])
            self.order2.append([])

    def data_range_airmass(self):
        if self.file_tdisp_order2 is not None:
            t_disp = np.loadtxt(self.file_tdisp_order2)
            T_disperseur = sp.interpolate.interp1d(t_disp.T[0], t_disp.T[1], kind="linear", bounds_error=False,
                                                    fill_value="extrapolate")

        for i in range(len(self.list_spectrum)):
            s = SpectrumAirmassFixed(file_name=self.list_spectrum[i])

            if s.target == self.target and s.disperseur == self.disperseur:
                print(s.tag)
                s.load_spec_data()
                data, err, cov_bin = s.adapt_from_lambdas_to_bin()
                for v in range(len(self.Bin) - 1):
                    self.data[v].append(data[v])
                    self.range_airmass[v].append(s.airmass)
                    self.err_mag[v].append(err[v])

                    data_conv = interp1d(self.new_lambda, data, kind="linear",
                                         bounds_error=False, fill_value=(0, 0))
                    lambdas_conv = interp1d(s.lambdas, s.lambdas_order2, kind="linear",
                                            bounds_error=False, fill_value=(0, 0))
                    LAMBDAS_ORDER2 = lambdas_conv(self.new_lambda)
                    if self.file_tdisp_order2 is not None:
                        I_order2 = data_conv(LAMBDAS_ORDER2) * T_disperseur(LAMBDAS_ORDER2)

                        self.order2[v].append(I_order2[v] * LAMBDAS_ORDER2[v] * np.gradient(LAMBDAS_ORDER2)[v]
                                          / np.gradient(self.new_lambda)[v] / self.new_lambda[v])
                    else:
                        self.order2[v].append(np.zeros_like(LAMBDAS_ORDER2))

                self.cov.append(cov_bin)
                self.names.append(self.list_spectrum[i])
                self.PSF_REG.append(s.psf_reg)
                if self.plot_specs:
                    plot_spectrum(s)

        self.data = np.array(self.data)
        self.range_airmass = np.array(self.range_airmass)
        self.err_mag = np.array(self.err_mag)
        self.names = np.array(self.names)
        self.order2 = np.array(self.order2)
        self.cov = np.array(self.cov)
        self.PSF_REG = np.array(self.PSF_REG)

    def check_outliers(self):
        indice = []
        outliers_detected = True
        while outliers_detected:
            for bin in range(len(self.Bin) - 1):
                if np.max(self.data[bin] / np.mean(self.data[bin])) > parameters.MULT_MAX or np.min(self.data[bin] / np.mean(self.data[bin])) < 1/parameters.MULT_MAX:
                    outliers = np.argmax(np.abs(self.data[bin] - np.mean(self.data[bin])))
                    print("----> outliers detected:" + self.names[outliers].split('/')[
                        -1] + "\t" + ", multiplicative factor compared to avg value: " + str(
                        self.data[bin][outliers] / np.mean(self.data[bin])) +
                          " for " + str(self.Bin[bin]) + "-" + str(self.Bin[bin + 1]) + " nm \n")
                    s = SpectrumAirmassFixed(file_name=self.names[outliers])
                    pl = input("Do you want check this spectra (y/n)? ")
                    if pl == 'y':
                        plot_spectrum(s)
                    rm = input("Do you want keep this spectra (y/n)? ")
                    if rm == 'n':
                        indice.append(outliers)
                    else:
                        continue
                    break
            if len(indice) == 0:
                outliers_detected = False
                test = [int(self.names[i][-16:-13]) for i in range(len(self.names))]
                i=0
                while i<len(self.names):
                    if test[i] in parameters.REMOVE_SPECTRA:
                        self.names = np.delete(self.names, i)
                        self.cov = np.delete(self.cov, i, 0)
                        self.data = np.delete(self.data, i, 1)
                        self.range_airmass = np.delete(self.range_airmass, i, 1)
                        self.err_mag = np.delete(self.err_mag, i, 1)
                        self.order2 = np.delete(self.order2, i, 1)
                        self.PSF_REG = np.delete(self.PSF_REG, i)
                        test = [int(self.names[i][-16:-13]) for i in range(len(self.names))]
                    else:
                        i+=1
            else:
                outliers_detected = True
                self.names = np.delete(self.names, indice[0])
                self.cov = np.delete(self.cov, indice[0], 0)
                self.data = np.delete(self.data, indice[0], 1)
                self.range_airmass = np.delete(self.range_airmass, indice[0], 1)
                self.err_mag = np.delete(self.err_mag, indice[0], 1)
                self.order2 = np.delete(self.order2, indice[0], 1)
                self.PSF_REG = np.delete(self.PSF_REG, indice[0])
                indice = []

    def invcov(self):
        self.INVCOV = []
        for j in range(len(self.names)):
            self.INVCOV.append(inv(self.cov[j]))

        "Remove star for HoloAmAg"
        if self.disperseur == 'HoloAmAg' and self.sim == False:
            for j in range(len(self.names)):
                a, b = np.argmin(abs(self.new_lambda-537.5)), np.argmin(abs(self.new_lambda-542.5))
                self.INVCOV[j][a,:], self.INVCOV[j][:,a] = self.INVCOV[j][a,:] / 10000000000, self.INVCOV[j][:,a] / 10000000000
                self.INVCOV[j][b, :], self.INVCOV[j][:, b] = self.INVCOV[j][b, :] / 10000000000, self.INVCOV[j][:, b] / 10000000000

    def megafit_emcee(self):
        nsamples = 300

        atm = []
        for j in range(len(self.names)):
            prod_name = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME)
            atmgrid = AtmosphereGrid(
                filename=(prod_name + '/' + self.names[j].split('/')[-1]).replace('sim', 'reduc').replace(
                    'spectrum.txt', 'atmsim.fits'))
            atm.append(atmgrid)

        def matrice_data():
            y = self.data - self.order2
            nb_spectre = len(self.names)
            nb_bin = len(self.data)
            D = np.zeros(nb_bin * nb_spectre)
            for j in range(nb_spectre):
                D[j * nb_bin: (j + 1) * nb_bin] = y[:, j]
            return D

        def Atm(atm, ozone, eau, aerosols):
            nb_spectre = len(self.names)
            nb_bin = len(self.data)
            M = np.zeros((nb_spectre, nb_bin, nb_bin))
            M_p = np.zeros((nb_spectre * nb_bin, nb_bin))
            for j in range(nb_spectre):
                Atmo = np.zeros(len(self.new_lambda))
                Lambdas = np.arange(self.Bin[0],self.Bin[-1],0.2)
                Atm = atm[j].simulate(ozone, eau, aerosols)(Lambdas)
                for i in range(len(self.new_lambda)):
                    Atmo[i] = np.mean(Atm[i*int(self.binwidths/0.2):(i+1)*int(self.binwidths/0.2)])
                a = np.diagflat(Atmo)
                M[j, :, :] = a
                M_p[nb_bin * j:nb_bin * (j+1),:] = a
            return M, M_p

        def log_likelihood(params_fit, atm):
            nb_spectre = len(self.names)
            nb_bin = len(self.data)
            ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
            D = matrice_data()
            M, M_p = Atm(atm, ozone, eau, aerosols)
            prod = np.zeros((nb_bin, nb_spectre * nb_bin))
            for spec in range(nb_spectre):
                prod[:,spec * nb_bin : (spec+1) * nb_bin] = M[spec] @ self.INVCOV[spec]
            COV = inv(prod @ M_p)
            A = COV @ prod @ D

            chi2 = 0
            for spec in range(nb_spectre):
                mat = D[spec * nb_bin : (spec+1) * nb_bin] - M[spec] @ A
                chi2 += mat @ self.INVCOV[spec] @ mat

            n = np.random.randint(0, 100)
            if n > 97:
                print(chi2 / (nb_spectre * nb_bin))
                print(ozone, eau, aerosols)
            return -0.5 * chi2

        def log_prior(params_fit):
            ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
            if 100 < ozone < 700 and 0 < eau < 10 and 0 < aerosols < 0.1:
                return 0
            else:
                return -np.inf

        def log_probability(params_fit, atm):
            lp = log_prior(params_fit)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params_fit, atm)

        if self.sim:
            filename = "sps/" + self.disperseur + "_"+ "sim_"+ parameters.PROD_NUM + "_emcee.h5"
        else:
            filename = "sps/" + self.disperseur + "_" + "reduc_" + parameters.PROD_NUM + "_emcee.h5"

        p_ozone = 300
        p_eau = 5
        p_aerosols = 0.03
        p0 = np.array([p_ozone, p_eau, p_aerosols])
        walker = 10

        init_ozone = p0[0] + p0[0] / 5 * np.random.randn(walker)
        init_eau = p0[1] + p0[1] / 5 * np.random.randn(walker)
        init_aerosols = p0[2] + p0[2] / 5 * np.random.randn(walker)

        p0 = np.array([[init_ozone[i], init_eau[i], init_aerosols[i]] for i in range(walker)])
        nwalkers, ndim = p0.shape

        backend = emcee.backends.HDFBackend(filename)
        try:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            args=(atm,), pool=pool, backend=backend)
            if backend.iteration > 0:
                p0 = backend.get_last_sample()

            if nsamples - backend.iteration > 0:
                sampler.run_mcmc(p0, nsteps=max(0, nsamples - backend.iteration), progress=True)
            pool.close()
        except ValueError:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            args=(atm,),
                                            threads=multiprocessing.cpu_count(), backend=backend)
            if backend.iteration > 0:
                p0 = sampler.get_last_sample()
            for _ in sampler.sample(p0, iterations=max(0, nsamples - backend.iteration), progress=True, store=True):
                continue

        flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)

        ozone, d_ozone = np.mean(flat_samples[:, -3]), np.std(flat_samples[:, -3])
        eau, d_eau = np.mean(flat_samples[:, -2]), np.std(flat_samples[:, -2])
        aerosols, d_aerosols = np.mean(flat_samples[:, -1]), np.std(flat_samples[:, -1])
        print(ozone, d_ozone)
        print(eau, d_eau)
        print(aerosols, d_aerosols)
        self.params_atmo = np.array([ozone, eau, aerosols])
        self.err_params_atmo = np.array([d_ozone, d_eau, d_aerosols])

        nb_spectre = len(self.names)
        nb_bin = len(self.data)
        M, M_p = Atm(atm, ozone, eau, aerosols)
        prod = np.zeros((nb_bin, nb_spectre * nb_bin))
        chi2 = 0
        for spec in range(nb_spectre):
            prod[:, spec * nb_bin: (spec + 1) * nb_bin] = M[spec] @ self.INVCOV[spec]

        COV = inv(prod @ M_p)
        D = matrice_data()
        Tinst = COV @ prod @ D
        Tinst_err = np.array([np.sqrt(COV[i,i]) for i in range(len(Tinst))])

        if self.disperseur == 'HoloAmAg' and self.sim == False:
            a, b = np.argmin(abs(self.new_lambda - 537.5)), np.argmin(abs(self.new_lambda - 542.5))
            Tinst_err[a], Tinst_err[b] = Tinst_err[a-1], Tinst_err[b+1]

        for spec in range(nb_spectre):
            mat = D[spec * nb_bin: (spec + 1) * nb_bin] - M[spec] @ Tinst
            chi2 += mat @ self.INVCOV[spec] @ mat
        print(chi2 / (nb_spectre * nb_bin))

        err = np.zeros_like(D)
        for j in range(len(self.names)):
            for i in range(len(self.data)):
                if self.disperseur == 'HoloAmAg' and self.sim == False:
                    if self.new_lambda[i] == 537.5 or self.new_lambda[i] == 542.5:
                        err[j * len(self.data) + i] = 1
                    else:
                        err[j * len(self.data) + i] = np.sqrt(self.cov[j][i, i])
                else:
                    err[j * len(self.data) + i] = np.sqrt(self.cov[j][i, i])

        model = M_p @ Tinst
        Err = (D - model) / err

        if parameters.plot_residuals:
            self.Plot_residuals(model, D, Err, COV)

        return Tinst, Tinst_err

    def Plot_residuals(self, model, D, Err, COV):

        def compute_correlation_matrix(cov):
            rho = np.zeros_like(cov)
            for i in range(cov.shape[0]):
                for j in range(cov.shape[1]):
                    rho[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
            return rho

        def plot_correlation_matrix_simple(ax, rho, axis_names, ipar=None):
            if ipar is None:
                ipar = np.arange(rho.shape[0]).astype(int)
            im = plt.imshow(rho[ipar[:, None], ipar], interpolation="nearest", cmap='bwr', vmin=-1, vmax=1)
            ax.set_title("Correlation matrix")
            names = [axis_names[ip] for ip in ipar]
            plt.xticks(np.arange(ipar.size), names, rotation='vertical', fontsize=11)
            plt.yticks(np.arange(ipar.size), names, fontsize=11)
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=9)
            plt.gcf().tight_layout()

        fig = plt.figure(figsize=[15, 10])
        ax = fig.add_subplot(111)
        axis_names = [str(i) for i in range(len(COV))]
        plot_correlation_matrix_simple(ax, compute_correlation_matrix(COV), axis_names, ipar=None)

        rho = np.zeros((len(self.names), len(self.data)))
        test = [int(self.names[i][-16:-13]) for i in range(len(self.names))]
        test2 = test.copy()
        for Test in test:
            C = 0
            for Test2 in test:
                if Test > Test2:
                    C += 1
            test2[C] = Test

        axis_names_vert = []
        for i in range(rho.shape[0]):
            k = np.argmin(abs(np.array(test) - test2[i]))
            axis_names_vert.append(str(test[k]))
            for j in range(rho.shape[1]):
                rho[i, j] = Err[k * len(self.data) + j]

        vert = np.arange(rho.shape[0]).astype(int)
        hor = np.arange(rho.shape[1]).astype(int)

        fig = plt.figure(figsize=[15, 7])
        ax = fig.add_subplot(111)
        axis_names_hor = [str(self.new_lambda[i]) for i in range(len(self.new_lambda))]
        norm = matplotlib.colors.SymLogNorm(vmin=-np.max(abs(rho)), vmax=np.max(abs(rho)), linthresh=10)
        im = plt.imshow(rho[vert[:, None], hor], interpolation="nearest", cmap='bwr',
                        vmin=-5, vmax=5)
        if self.sim:
            ax.set_title(self.disperseur +' '+ parameters.PROD_NUM, fontsize=21)
        else:
            ax.set_title(self.disperseur +' '+ parameters.PROD_NUM, fontsize=21)
        print(np.mean(rho))
        print(np.std(rho))
        names_vert = [axis_names_vert[ip] for ip in vert]
        names_hor = [axis_names_hor[ip] for ip in hor]
        plt.xticks(np.arange(0, hor.size, 3), names_hor[::3], rotation='vertical', fontsize=14)
        plt.yticks(np.arange(0, vert.size, 3), names_vert[::3], fontsize=14)
        plt.xlabel('$\lambda$ [nm]', fontsize=17)
        plt.ylabel('Spectrum index', fontsize=17)
        cbar = plt.colorbar(im, orientation='horizontal')
        cbar.set_label('Residuals in #$\sigma$', fontsize=20)
        cbar.ax.tick_params(labelsize=13)
        plt.gcf().tight_layout()
        fig.tight_layout()
        if os.path.exists(parameters.OUTPUTS_THROUGHPUT_SIM) == False or os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC) == False:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_SIM)
        if self.sim and parameters.save_residuals:
            plt.savefig(
                parameters.OUTPUTS_THROUGHPUT_SIM + 'throughput_sim, ' + self.disperseur + ',résidus, version_' + parameters.PROD_NUM + '.pdf')
        elif parameters.save_residuals:
            plt.savefig(
                parameters.OUTPUTS_THROUGHPUT_REDUC + 'throughput_reduc, ' + self.disperseur + ',résidus, version_' + parameters.PROD_NUM + '.pdf')

        Kplot = parameters.COMPARISON_MD_SPECTRA
        for i in range(len(Kplot)):
            k = np.argmin(abs(np.array(test) - Kplot[i]))
            fig = plt.figure(figsize=[15, 15])
            ax = fig.add_subplot(111)
            plt.plot(self.new_lambda, model[k * len(self.new_lambda) : (k+1) * len(self.new_lambda)], c='red', label='model')
            plt.plot(self.new_lambda, D[k * len(self.new_lambda) : (k+1) * len(self.new_lambda)], c='blue', label='data')
            plt.title("spectrum :"+str(Kplot[i]))
            plt.grid(True)
            plt.legend()
        plt.show()

    def fit_spectrum(self):
        nsamples = 300

        def spec_calspec():
            spec = open(self.prod_reduc[0], 'r')
            lambdas = []
            data = []
            for line in spec:
                Line = line.split()
                if Line[0] != '#':
                    lambdas.append(float(Line[0]))
                    data.append(float(Line[1]))

            lambdas_calspec = np.array(lambdas)
            data_calspec_org = np.array(data)
            data_calspec = data_calspec_org
            fluxlum_Binreel = np.zeros(len(self.Bin) - 1)
            interpolation_reel = sp.interpolate.interp1d(lambdas_calspec, data_calspec, kind="linear",
                                                         bounds_error=False,
                                                         fill_value=(0, 0))
            for v in range(len(self.Bin) - 1):
                "On rempli les tableaux par bin de longueur d'onde"
                X = np.linspace(self.Bin[v], self.Bin[v + 1], int(self.binwidths * 100))
                Y = interpolation_reel(X)
                fluxlum_Binreel[v] = integrate.simps(Y, X, dx=1) / self.binwidths
            data_calspec = fluxlum_Binreel
            return data_calspec

        def Ordonnee():
            data_calspec = spec_calspec()
            if self.disperseur != parameters.DISPERSER_REF:
                disp = np.loadtxt(parameters.DISPERSER_EXTRACTION)
            else:
                disp = np.loadtxt(parameters.DISPERSER_REF_BANC)
            Tctio = np.loadtxt(parameters.THROUGHPUT_REDUC)
            Tctio_data = sp.interpolate.interp1d(Tctio.T[0], Tctio.T[1], bounds_error=False,
                                                 fill_value="extrapolate")(self.new_lambda)
            disp_data = sp.interpolate.interp1d(disp.T[0], disp.T[1], bounds_error=False,
                                                    fill_value="extrapolate")(self.new_lambda)
            return data_calspec * Tctio_data * disp_data

        OZONE = np.zeros(len(self.names))
        AEROSOLS = np.zeros(len(self.names))
        EAU = np.zeros(len(self.names))
        ERR_OZONE = np.zeros(len(self.names))
        ERR_AEROSOLS = np.zeros(len(self.names))
        ERR_EAU = np.zeros(len(self.names))

        for spec in range(len(self.names)):
            atm = AtmosphereGrid(
                filename=(os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME) + '/' + self.names[spec].split('/')[-1]).replace('sim', 'reduc').replace(
                    'spectrum.txt', 'atmsim.fits'))

            self.atm = atm
            ORDO = Ordonnee()
            def Atm(atm, ozone, eau, aerosols):
                nb_bin = len(self.data)
                Atmo = np.zeros(nb_bin)
                Lambdas = np.arange(self.Bin[0], self.Bin[-1], 0.2)
                Atm = atm.simulate(ozone, eau, aerosols)(Lambdas)
                for i in range(len(self.new_lambda)):
                    Atmo[i] = np.mean(Atm[i * int(self.binwidths / 0.2):(i + 1) * int(self.binwidths / 0.2)])
                M = np.diagflat(Atmo)
                return M

            def log_likelihood(params_fit):
                nb_bin = len(self.data)
                ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
                M = Atm(self.atm,ozone, eau, aerosols)
                D = self.data[:,spec] - self.order2[:,spec]
                mat = D - M @ ORDO

                chi2 = mat @ self.INVCOV[spec] @ mat

                n = np.random.randint(0, 100)
                if n > 97:
                    print(chi2 / (nb_bin))
                    print(ozone, eau, aerosols)

                return -0.5 * chi2

            def log_prior(params_fit):
                ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
                if 100 < ozone < 700 and 0 < eau < 10 and 0 < aerosols < 0.1:
                    return 0
                else:
                    return -np.inf

            def log_probability(params_fit):
                lp = log_prior(params_fit)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(params_fit)

            p_ozone = 300
            p_eau = 5
            p_aerosols = 0.03
            p0 = np.array([p_ozone, p_eau, p_aerosols])
            walker = 10

            init_ozone = p0[0] + p0[0] / 5 * np.random.randn(walker)
            init_eau = p0[1] + p0[1] / 5 * np.random.randn(walker)
            init_aerosols = p0[2] + p0[2] / 5 * np.random.randn(walker)

            p0 = np.array([[init_ozone[i], init_eau[i], init_aerosols[i]] for i in range(walker)])
            nwalkers, ndim = p0.shape

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                            threads=multiprocessing.cpu_count())
            sampler.run_mcmc(p0, nsamples, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)

            ozone, d_ozone = np.mean(flat_samples[:, -3]), np.std(flat_samples[:, -3])
            eau, d_eau = np.mean(flat_samples[:, -2]), np.std(flat_samples[:, -2])
            aerosols, d_aerosols = np.mean(flat_samples[:, -1]), np.std(flat_samples[:, -1])
            print(ozone, d_ozone)
            print(eau, d_eau)
            print(aerosols, d_aerosols)
            OZONE[spec] = ozone
            AEROSOLS[spec] = aerosols
            EAU[spec] = eau
            ERR_OZONE[spec] = d_ozone
            ERR_EAU[spec] = d_eau
            ERR_AEROSOLS[spec] = d_aerosols

        if self.sim:
            chemin = parameters.OUTPUTS_FITSPECTRUM_SIM
        else:
            chemin = parameters.OUTPUTS_FITSPECTRUM_REDUC

        NUM = np.zeros(len(self.names))
        for i in range(len(self.names)):
            NUM[i] = self.names[i][-16:-13]
        fig = plt.figure(figsize=[10, 10])
        ax2 = fig.add_subplot(111)
        ax2.get_xaxis().set_tick_params(labelsize=12)
        ax2.get_yaxis().set_tick_params(labelsize=12)
        plt.grid(True)
        plt.title("Fit spectrum with bins", fontsize=22)
        plt.xlabel('Spectrum index', fontsize=20)
        plt.ylabel("Ozone [db]", fontsize=20)
        plt.scatter(NUM, OZONE, c='blue')
        plt.errorbar(NUM, OZONE, xerr=None, yerr=ERR_OZONE, fmt='none', capsize=1,
                     ecolor='blue', zorder=2, elinewidth=2)
        fig.tight_layout()
        if os.path.exists(chemin) == False:
            os.makedirs(chemin)
        plt.savefig(chemin + 'Ozone_fit, ' + self.disperseur + ', version_' + parameters.PROD_NUM + '.pdf')

        fig = plt.figure(figsize=[10, 10])
        ax2 = fig.add_subplot(111)
        ax2.get_xaxis().set_tick_params(labelsize=12)
        ax2.get_yaxis().set_tick_params(labelsize=12)
        plt.grid(True)
        plt.title("Fit spectrum with bins", fontsize=22)
        plt.xlabel('Spectrum index', fontsize=20)
        plt.ylabel("Aerosols [VAOD]", fontsize=20)
        plt.scatter(NUM, AEROSOLS, c='blue')
        plt.errorbar(NUM, AEROSOLS, xerr=None, yerr=ERR_AEROSOLS, fmt='none', capsize=1,
                     ecolor='blue', zorder=2, elinewidth=2)
        fig.tight_layout()
        plt.savefig(chemin + 'Aerosols_fit, ' + self.disperseur + ', version_' + parameters.PROD_NUM + '.pdf')

        fig = plt.figure(figsize=[10, 10])
        ax2 = fig.add_subplot(111)
        ax2.get_xaxis().set_tick_params(labelsize=12)
        ax2.get_yaxis().set_tick_params(labelsize=12)
        plt.grid(True)
        plt.title("Fit spectrum with bins", fontsize=22)
        plt.xlabel('Spectrum index', fontsize=20)
        plt.ylabel("PWV [mm]", fontsize=20)
        plt.scatter(NUM, EAU, c='blue')
        plt.errorbar(NUM, EAU, xerr=None, yerr=ERR_EAU, fmt='none', capsize=1,
                     ecolor='blue', zorder=2, elinewidth=2)
        fig.tight_layout()
        plt.savefig(chemin + 'Eau_fit, ' + self.disperseur + ', version_' + parameters.PROD_NUM + '.pdf')

        plt.show()
