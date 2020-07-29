# coding: utf8

from scipy.interpolate import interp1d  # interpolation
import emcee
import multiprocessing
import sys
from schwimmbad import MPIPool
from SpectrumAirmassFixed import *
from spectractor.simulation.simulator import AtmosphereGrid  # grille d'atmosphère
from numpy.linalg import inv
import matplotlib.colors
from scipy import integrate

class SpectrumRangeAirmass:

    def __init__(self, prod_name="", sim="", disperseur="", target="", plot_specs="", prod_sim="", prod_reduc=""):

        self.target = target
        self.disperseur = disperseur
        self.plot_specs = plot_specs
        self.binwidths = parameters.BINWIDTHS
        self.lambda_min = parameters.LAMBDA_MIN
        self.lambda_max = parameters.LAMBDA_MAX
        self.new_lambda = parameters.NEW_LAMBDA
        self.Bin = parameters.BIN
        self.sim = sim
        self.list_spectrum = []
        self.data_mag = []
        self.range_airmass = []
        self.err_mag = []
        self.order2 = []
        self.names = []
        self.cov = []
        self.INVCOV = []
        self.atm = None
        # ATTENTION à modifier #
        if self.sim:# and self.disperseur != 'Thor300':
            self.file_tdisp_order2 = os.path.join(parameters.THROUGHPUT_DIR, 'Thor300_order2_bis.txt')  # self.disperseur +
        else:
            print('cest bon')
            self.file_tdisp_order2 = os.path.join(parameters.THROUGHPUT_DIR, 'Thor300_order2.txt')
        self.prod_sim = prod_sim
        self.prod_reduc = prod_reduc
        self.PSF_REG = []
        if prod_name != "":
            self.prod_name = prod_name
            self.init_spectrumrangeairmass()
            self.data_range_airmass()
            self.check_outliers()
            self.invcov()

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

        if self.disperseur == 'Thor300':
            t_disp = np.loadtxt(self.file_tdisp_order2)
            T_disperseur = sp.interpolate.interp1d(t_disp.T[0], t_disp.T[1], kind="linear", bounds_error=False,
                                               fill_value="extrapolate")
        else:
            t_disp = np.loadtxt(self.file_tdisp_order2)
            T_disperseur = sp.interpolate.interp1d(t_disp.T[0], np.zeros(len(t_disp.T[0])), kind="linear", bounds_error=False,
                                                   fill_value="extrapolate")

        t_disp = np.loadtxt(self.file_tdisp_order2)
        T_disperseur = sp.interpolate.interp1d(t_disp.T[0], t_disp.T[1], kind="linear", bounds_error=False,
                                               fill_value="extrapolate")

        for i in range(len(self.list_spectrum)):
            s = SpectrumAirmassFixed(file_name=self.list_spectrum[i])

            if s.target == self.target and s.disperseur == self.disperseur :#and s.airmass != 1.024 and s.airmass != 1.050 :
                print(s.tag)
                s.load_spec_data()
                data_bin, err_bin, cov_bin = s.adapt_from_lambdas_to_bin()
                data, err = convert_from_flam_to_mag(data_bin, err_bin)
                for v in range(len(self.Bin) - 1):
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

                self.cov.append(cov_bin)
                self.names.append(self.list_spectrum[i])
                self.PSF_REG.append(s.psf_reg)
                if self.plot_specs:
                    plot_spectrum(s)

        self.data_mag = np.array(self.data_mag)
        self.range_airmass = np.array(self.range_airmass)
        self.err_mag = np.array(self.err_mag)
        self.names = np.array(self.names)
        self.order2 = np.array(self.order2)
        self.cov = np.array(self.cov)
        self.PSF_REG = np.array(self.PSF_REG)

    def bouguer_line(self):
        def flin(x, a, b):
            return a * x + b

        slope = np.zeros(len(self.data_mag))  # on initialise la liste des coefficients à la liste vide.
        ord = np.zeros(len(self.data_mag))
        err_slope = np.zeros(
            len(self.data_mag))  # on initialise la liste des erreurs sur les ordonnees à l'origine à la liste vide.
        err_ord = np.zeros(len(self.data_mag))

        for i in range(len(self.data_mag)):
            popt, pcov = sp.optimize.curve_fit(flin, self.range_airmass[i], self.data_mag[i], sigma=self.err_mag[i])
            slope[i], ord[i] = popt[0], popt[1]
            err_ord[i] = np.sqrt(pcov[1][1])
            err_slope[i] = np.sqrt(pcov[0][0])

        return slope, ord, err_slope, err_ord

    def bouguer_line_order2(self):
        def forder2(x, a, b, A2=1):
            return np.log(np.exp(a * x + b) + A2 * order2)
            """
            if np.exp(a * x + b) + A2 * order2 > 0:
                return np.log(np.exp(a * x + b) + A2 * order2)
            else:
                return -np.inf
            """
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
            # print(order2)
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
                # print(p0)
                walker = 10
                init_a = p0[0] - p0[0] / 100 * 5 * abs(np.random.randn(walker))
                init_b = p0[1] + p0[1] / 100 * 5 * abs(np.random.randn(walker))
                # init_A2 = p0[2] + p0[2] / 100 * 50 * np.random.randn(walker)

                pos = np.array([[init_a[i], init_b[i]] for i in range(len(init_a))])
                # pos = p0 + np.array([init_a,init_b, init_A2])
                # pos = p0 + p0 / 100 * 5 * (2 * np.random.randn(20, 3) - 1)
                # print(pos)
                nwalkers, ndim = pos.shape

                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                                args=(self.range_airmass[i], self.data_mag[i], self.err_mag[i]))
                sampler.run_mcmc(pos, 1000, progress=True)
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
                # print(slope[i])
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

    def invcov(self):
        self.INVCOV = []
        for j in range(len(self.names)):
            self.INVCOV.append(inv(self.cov[j]))
        for j in range(len(self.names)):
            a, b = np.argmin(abs(self.new_lambda-537.5)), np.argmin(abs(self.new_lambda-542.5))
            self.INVCOV[j][a,:], self.INVCOV[j][:,a] = self.INVCOV[j][a,:] / 10000000000, self.INVCOV[j][:,a] / 10000000000
            self.INVCOV[j][b, :], self.INVCOV[j][:, b] = self.INVCOV[j][b, :] / 10000000000, self.INVCOV[j][:, b] / 10000000000
    def matrice_data(self, A2):
        y = (np.exp(self.data_mag) - self.order2)
        nb_spectre = len(self.names)
        nb_bin = len(self.data_mag)
        D = np.zeros(nb_bin * nb_spectre)
        for j in range(nb_spectre):
            D[j * nb_bin: (j+1)*nb_bin] = y[:,j]
        return D

    def megafit_emcee(self):
        nsamples = 300

        atm = []
        for j in range(len(self.names)):
            prod_name = os.path.join(parameters.PROD_DIRECTORY, parameters.PROD_NAME)
            atmgrid = AtmosphereGrid(
                filename=(prod_name + '/' + self.names[j].split('/')[-1]).replace('sim', 'reduc').replace(
                    'spectrum.txt', 'atmsim.fits'))
            atm.append(atmgrid)

        #D = self.matrice_data()

        """
        def f_tinst_atm(Tinst, ozone, eau, aerosols, atm):
            model = np.zeros((len(self.data_mag), len(self.names)))
            for j in range(len(self.names)):
                a = atm[j].simulate(ozone, eau, aerosols)
                model[:, j] = Tinst * a(self.new_lambda)
            return model
        """

        def Atm(atm, ozone, eau, aerosols):
            nb_spectre = len(self.names)
            nb_bin = len(self.data_mag)
            M = np.zeros((nb_spectre, nb_bin, nb_bin))
            M_p = np.zeros((nb_spectre * nb_bin, nb_bin))
            for j in range(nb_spectre):
                Atmo = np.zeros(len(self.new_lambda))
                Lambdas = np.arange(self.Bin[0],self.Bin[-1],0.2)
                Atm = atm[j].simulate(ozone, eau, aerosols)(Lambdas)
                for i in range(len(self.new_lambda)):
                    #step = int(self.binwidths * 10)
                    #X = np.linspace(self.Bin[i], self.Bin[i + 1] + step, step)
                    Atmo[i] = np.mean(Atm[i*int(self.binwidths/0.2):(i+1)*int(self.binwidths/0.2)])
                a = np.diagflat(Atmo)
                M[j, :, :] = a
                M_p[nb_bin * j:nb_bin * (j+1),:] = a
            return M, M_p

        def log_likelihood(params_fit, atm):
            nb_spectre = len(self.names)
            nb_bin = len(self.data_mag)
            ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
            A2 = 0
            D = self.matrice_data(A2)
            M, M_p = Atm(atm, ozone, eau, aerosols)
            #A = np.zeros(nb_bin)
            #COV = np.zeros((nb_bin, nb_bin))
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
                print(A2, ozone, eau, aerosols)

            return -0.5 * chi2

        def log_prior(params_fit):
            ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
            if 100 < ozone < 700 and 0 < eau < 10 and 0 < aerosols < 0.1:# and 0<A2<5:
                return 0
            else:
                return -np.inf

        def log_probability(params_fit, atm):
            lp = log_prior(params_fit)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params_fit, atm)

        if self.sim:
            filename = "sps/" + self.disperseur + "_"+ "sim_new_"+ parameters.PROD_NUM + "_emcee.h5"#+ "sim_new_"
        else:
            filename = "sps/" + self.disperseur + "_" + "reduc_new_" + parameters.PROD_NUM + "_emcee.h5"

        if os.path.exists(filename):
            slope, ord, err_slope, err_ord = self.bouguer_line()
        else:
            slope, ord, err_slope, err_ord, A2, A2_err = self.bouguer_line_order2()
        p_ozone = 300
        p_eau = 5
        p_aerosols = 0.03
        #p_A2 = 1
        p0 = np.array([p_ozone, p_eau, p_aerosols])
        walker = 10

        #init_A2 = p0[0] + p0[0] / 5 * np.random.randn(walker)
        init_ozone = p0[0] + p0[0] / 5 * np.random.randn(walker)
        init_eau = p0[1] + p0[1] / 5 * np.random.randn(walker)
        init_aerosols = p0[2] + p0[2] / 5 * np.random.randn(walker)

        p0 = np.array([[init_ozone[i], init_eau[i], init_aerosols[i]] for i in range(walker)])
        nwalkers, ndim = p0.shape

        """
        plt.errorbar(self.new_lambda, (np.exp(self.data_mag) - self.order2)[:,10], yerr=(self.err_mag * np.exp(self.data_mag))[:,10])
        plt.show()
        """
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

        #A2, A2_err = np.mean(flat_samples[:, -4]), np.std(flat_samples[:, -4])
        ozone, d_ozone = np.mean(flat_samples[:, -3]), np.std(flat_samples[:, -3])
        eau, d_eau = np.mean(flat_samples[:, -2]), np.std(flat_samples[:, -2])
        aerosols, d_aerosols = np.mean(flat_samples[:, -1]), np.std(flat_samples[:, -1])
        print(ozone, d_ozone)
        print(eau, d_eau)
        print(aerosols, d_aerosols)
        #print(A2, A2_err)
        nb_spectre = len(self.names)
        nb_bin = len(self.data_mag)
        M, M_p = Atm(atm, ozone, eau, aerosols)
        prod = np.zeros((nb_bin, nb_spectre * nb_bin))
        chi2 = 0
        for spec in range(nb_spectre):
            prod[:, spec * nb_bin: (spec + 1) * nb_bin] = M[spec] @ self.INVCOV[spec]

        COV = inv(prod @ M_p)
        A2=0
        D = self.matrice_data(A2)
        Tinst = COV @ prod @ D
        Tinst_err = np.array([np.sqrt(COV[i,i]) for i in range(len(Tinst))])
        a, b = np.argmin(abs(self.new_lambda - 537.5)), np.argmin(abs(self.new_lambda - 542.5))
        Tinst_err[a], Tinst_err[b] = 1e-16, 1e-16
        A = COV @ prod @ D
        for spec in range(nb_spectre):
            mat = D[spec * nb_bin: (spec + 1) * nb_bin] - M[spec] @ A
            chi2 += mat @ self.INVCOV[spec] @ mat
        print(chi2 / (nb_spectre * nb_bin))

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

        def plot_err(err, ipar=None):
            rho = np.zeros((len(self.names), len(self.data_mag)))
            test = [int(self.names[i][-16:-13]) for i in range(len(self.names))]
            print(test)
            test2 = test.copy()
            for Test in test:
                C = 0
                for Test2 in test:
                    if Test > Test2:
                        C+=1
                test2[C] = Test
            print(test2)
            axis_names_vert = []
            for i in range(rho.shape[0]):
                k = np.argmin(abs(np.array(test) - test2[i]))
                axis_names_vert.append(str(test[k]))
                for j in range(rho.shape[1]):
                    rho[i, j] = err[k * len(self.data_mag) + j]
            if ipar is None:
                vert = np.arange(rho.shape[0]).astype(int)
                hor = np.arange(rho.shape[1]).astype(int)
            """
            gs_kw = dict(height_ratios=[1], width_ratios=[5,1])
            fig, ax = plt.subplots(1, 2, figsize=[15, 15], constrained_layout=True, gridspec_kw=gs_kw)
            ax1, ax2 = ax[0], ax[1]
            """
            fig = plt.figure(figsize=[15, 7])
            ax = fig.add_subplot(111)
            axis_names_hor = [str(self.new_lambda[i]) for i in range(len(self.new_lambda))]
            norm = matplotlib.colors.SymLogNorm(vmin=-np.max(abs(rho)), vmax=np.max(abs(rho)), linthresh=10)
            im = plt.imshow(rho[vert[:, None], hor], interpolation="nearest", cmap='bwr',
                            vmin=-5, vmax=5)
            if self.sim:
                #ax.set_title("Résidus: (data - model) / err sur des simulations du "+self.disperseur+" version "+parameters.PROD_NUM)
                ax.set_title(self.disperseur, fontsize=21)
            else:
                ax.set_title(self.disperseur, fontsize = 21)
            print(np.mean(rho))
            print(np.std(rho))
            names_vert = [axis_names_vert[ip] for ip in vert]
            names_hor = [axis_names_hor[ip] for ip in hor]
            plt.xticks(np.arange(0, hor.size, 3), names_hor[::3], rotation='vertical', fontsize=14)
            plt.yticks(np.arange(0, vert.size, 3), names_vert[::3], fontsize=14)
            plt.xlabel('$\lambda$ [nm]', fontsize = 17)
            plt.ylabel('Spectrum index', fontsize = 17)
            cbar = plt.colorbar(im, orientation = 'horizontal')
            cbar.set_label('Residuals in #$\sigma$', fontsize = 20)
            cbar.ax.tick_params(labelsize=13)
            # cbar.ax.set_yticklabels(['{:.0f}'.format(x) for x in np.linspace(np.min(rho), np.max(rho), 10)])                                    #fontsize=16, weight='bold')
            plt.gcf().tight_layout()
            fig.tight_layout()
            if self.sim and 1==1:
                plt.savefig(
                    parameters.OUTPUTS_THROUGHPUT_SIM + 'throughput_simb, ' + self.disperseur + ',résidus, version_' + parameters.PROD_NUM + '.pdf')
            elif 1==1:
                plt.savefig(
                    parameters.OUTPUTS_THROUGHPUT_REDUC + 'throughput_reduc, ' + self.disperseur + ',résidus, version_' + parameters.PROD_NUM + '.pdf')

        fig = plt.figure(figsize=[15, 10])
        ax = fig.add_subplot(111)
        axis_names = [str(i) for i in range(len(COV))]
        plot_correlation_matrix_simple(ax, compute_correlation_matrix(COV), axis_names, ipar=None)

        err = np.zeros_like(D)
        for j in range(len(self.names)):
            for i in range(len(self.data_mag)):
                #if self.new_lambda[i]==537.5 or self.new_lambda[i]==542.5:
                 #   err[j * len(self.data_mag) + i] = 1
                #else:
                err[j * len(self.data_mag) + i] = np.sqrt(self.cov[j][i,i])
        model = M_p @ Tinst
        Err = (D - model) / err
        plot_err(Err)
        fig = plt.figure(figsize=[15, 15])
        ax = fig.add_subplot(111)
        ax.hist(Err, bins=np.arange(-10, 10, 1))
        ax.set_title("Histogramme des résidus: (data - model) / err sur des données du HoloAmAg version " + parameters.PROD_NUM)
        plt.xlabel('Ecart aux données', fontsize=14)
        plt.grid(True)
        test = [int(self.names[i][-16:-13]) for i in range(len(self.names))]
        Kplot = [181, 186, 191, 196]
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
        return Tinst, Tinst_err

    def spec_calspec(self):
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
        # self.data_calspec = filter_detect_lines(self.lambdas_calspec, self.data_calspec_org)
        data_calspec = data_calspec_org  # ATTENTION
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
        data_calspec_mag = convert_from_flam_to_mag(fluxlum_Binreel, np.zeros(len(fluxlum_Binreel)))
        return data_calspec

    def Ordonnee(self):
        data_calspec = self.spec_calspec()
        holoamag = np.loadtxt('throughput/HoloAmAg_basectiothor300, version_6.9.txt')
        Tctio = np.loadtxt('throughput/ctio_throughput_basethor300_prod6.9_new.txt')
        Tctio_data = sp.interpolate.interp1d(Tctio.T[0], Tctio.T[1], bounds_error=False,
                                               fill_value="extrapolate")(self.new_lambda)
        holoamag_data = sp.interpolate.interp1d(holoamag.T[0], holoamag.T[1], bounds_error=False,
                                               fill_value="extrapolate")(self.new_lambda)

        return data_calspec * Tctio_data * holoamag_data

    def archi_mega_fit(self):
        nsamples = 300

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
            ORDO = self.Ordonnee()
            def Atm(atm, ozone, eau, aerosols):
                nb_bin = len(self.data_mag)
                Atmo = np.zeros(nb_bin)
                Lambdas = np.arange(self.Bin[0], self.Bin[-1], 0.2)
                Atm = atm.simulate(ozone, eau, aerosols)(Lambdas)
                for i in range(len(self.new_lambda)):
                    Atmo[i] = np.mean(Atm[i * int(self.binwidths / 0.2):(i + 1) * int(self.binwidths / 0.2)])
                M = np.diagflat(Atmo)
                return M

            def log_likelihood(params_fit):
                nb_bin = len(self.data_mag)
                ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
                M = Atm(self.atm,ozone, eau, aerosols)
                D = np.exp(self.data_mag[:,spec]) - 0 * self.order2[:,spec]
                mat = D - M @ ORDO

                chi2 = mat @ self.INVCOV[spec] @ mat

                n = np.random.randint(0, 100)
                if n > 97:
                    print(chi2 / (nb_bin))
                    print(ozone, eau, aerosols)

                return -0.5 * chi2

            def log_prior(params_fit):
                ozone, eau, aerosols = params_fit[-3], params_fit[-2], params_fit[-1]
                if 100 < ozone < 700 and 0 < eau < 10 and 0 < aerosols < 0.1:  # and 0<A2<5:
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
        plt.savefig('Ozone_fit.pdf')

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
        plt.savefig('Aerosols_fit.pdf')

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
        plt.savefig('Eau_fit.pdf')

        plt.show()
    def check_outliers(self):
        indice = []
        outliers_detected = True
        while outliers_detected == True:
            for bin in range(len(self.Bin) - 1):
                if np.max(np.abs(self.data_mag[bin] - np.mean(self.data_mag[bin]))) > parameters.MAG_MAX:
                    outliers = np.argmax(np.abs(self.data_mag[bin] - np.mean(self.data_mag[bin])))
                    print("----> outliers detected:" + self.names[outliers].split('/')[
                        -1] + "\t" + ", multiplicative factor compared to avg value: " + str(
                        np.exp(np.max(np.abs(self.data_mag[bin] - np.mean(self.data_mag[bin])))))[:3] +
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
                    if test[i] == 58 or test[i] == 86 or test[i] == 66 or test[i] == 121 or test[i] == 106: #(58 - 86 -66 -121: prod6.9, 106 - 58: prod6.7)
                        self.names = np.delete(self.names, i)
                        self.cov = np.delete(self.cov, i, 0)
                        self.data_mag = np.delete(self.data_mag, i, 1)
                        self.range_airmass = np.delete(self.range_airmass, i, 1)
                        self.err_mag = np.delete(self.err_mag, i, 1)
                        self.order2 = np.delete(self.order2, i, 1)
                        self.PSF_REG = np.delete(self.PSF_REG, i)
                        print('ok')
                        test = [int(self.names[i][-16:-13]) for i in range(len(self.names))]
                    else:
                        i+=1
            else:
                outliers_detected = True
                #print(self.names.shape,self.cov.shape,self.data_mag.shape,self.range_airmass.shape)
                self.names = np.delete(self.names, indice[0])
                self.cov = np.delete(self.cov, indice[0], 0)
                self.data_mag = np.delete(self.data_mag, indice[0], 1)
                self.range_airmass = np.delete(self.range_airmass, indice[0], 1)
                self.err_mag = np.delete(self.err_mag, indice[0], 1)
                self.order2 = np.delete(self.order2, indice[0], 1)
                self.PSF_REG = np.delete(self.PSF_REG, indice[0])
                indice = []

def convert_from_flam_to_mag(data, err):
    for i in range(len(data)):
        if data[i] < 1e-15:
            data[i] = 1e-15

    return np.log(data), err / data