# coding: utf8

from scipy import signal  # filtre savgol pour enlever le bruit
from scipy.interpolate import interp1d  # interpolation
from spectractor.tools import wavelength_to_rgb  # couleurs des longueurs d'ondes
from SpectrumRangeAirmass import *
from Filters import *

class TransmissionInstrumentale:

    def __init__(self, prod_name="", sim="", disperseur="", target="", order2="", mega_fit="", plot_filt="",
                 save_filter="", prod=""):
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
        # self.data_calspec = filter_detect_lines(self.lambdas_calspec, self.data_calspec_org)
        self.data_calspec = self.data_calspec_org  # ATTENTION
        fluxlum_Binreel = np.zeros(len(self.Bin) - 1)
        interpolation_reel = sp.interpolate.interp1d(self.lambdas_calspec, self.data_calspec, kind="linear",
                                                     bounds_error=False,
                                                     fill_value=(0, 0))
        for v in range(len(self.Bin) - 1):
            "On rempli les tableaux par bin de longueur d'onde"
            X = np.linspace(self.Bin[v], self.Bin[v + 1], int(self.binwidths * 100))
            Y = interpolation_reel(X)
            fluxlum_Binreel[v] = integrate.simps(Y, X, dx=1) / self.binwidths

        self.data_calspec = fluxlum_Binreel
        self.data_calspec_mag = convert_from_flam_to_mag(fluxlum_Binreel, np.zeros(len(fluxlum_Binreel)))

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

        # self.data = filter_detect_lines(self.lambdas, self.data, self.plot_filt, self.save_filter)

        Data = sp.interpolate.interp1d(self.lambdas, self.data, kind="linear", bounds_error=False,
                                       fill_value="extrapolate")
        Data_bouguer = sp.interpolate.interp1d(self.lambdas, self.data_bouguer, kind="linear", bounds_error=False,
                                               fill_value="extrapolate")
        Err = sp.interpolate.interp1d(self.lambdas, self.err, kind="linear", bounds_error=False,
                                      fill_value="extrapolate")
        #self.lambdas = self.new_lambda
        self.lambdas = np.arange(self.lambda_min, self.lambda_max, 1)

        self.data_tel = Data_tel(self.lambdas)
        self.data_tel_err = Err_tel(self.lambdas)
        self.data_disp = Data_disp(self.lambdas)
        self.data_disp_err = Err_disp(self.lambdas)
        self.data = Data(self.lambdas)
        self.err = Err(self.lambdas)
        self.data_bouguer = Data_bouguer(self.lambdas)
        # self.data = sp.signal.savgol_filter(self.data, 111, 2)

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
            self.ord2, self.err_order2 = spectrumrangeairmass.megafit_emcee()
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
            """
            Err = sp.interpolate.interp1d(self.lambdas, self.err_order2, kind="linear", bounds_error=False,
                                          fill_value="extrapolate")

            #self.lambdas = self.new_lambda
            self.lambdas = np.arange(self.lambda_min, self.lambda_max, 1)
            self.data_order2 = Data(self.lambdas)
            self.err_order2 = Err(self.lambdas)
            self.data_bouguer = Data_bouguer(self.lambdas)
            # self.data_order2 = sp.signal.savgol_filter(self.data_order2, 111, 2)


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

        popt, pcov = sp.optimize.curve_fit(fatmosphere, T.new_lambda, np.exp(T.slope), p0=[300, 5, 0.03],
                                           sigma=T.err_slope * np.exp(T.slope),
                                           bounds=([0, 0, 0], [700, 20, 0.5]), verbose=2)
        print(popt[0], np.sqrt(pcov[0][0]))
        print(popt[1], np.sqrt(pcov[1][1]))
        print(popt[2], np.sqrt(pcov[2][2]))
        c = atmgrid.simulate(*popt)
        plt.plot(Lambda, np.exp(np.log(b(Lambda)) / 1.047), color='blue', label='transmission libradtran typique')
    else:
        def fatmosphere(lambdas, ozone, eau, aerosol):
            return np.exp(np.log(atmgrid.simulate(ozone, eau, aerosol)(lambdas)) / 1.047)

        popt, pcov = sp.optimize.curve_fit(fatmosphere, T.new_lambda, np.exp(T.slope), p0=[300, 5, 0.03],
                                           sigma=T.err_slope * np.exp(T.slope),
                                           bounds=([0, 0, 0], [700, 20, 0.5]), verbose=2)
        print(popt[0], np.sqrt(pcov[0][0]))
        print(popt[1], np.sqrt(pcov[1][1]))
        print(popt[2], np.sqrt(pcov[2][2]))
        c = atmgrid.simulate(*popt)

    plt.plot(Lambda, np.exp(np.log(c(Lambda)) / 1.047), color='black', label='transmission libradtran ajustée')
    plt.plot(T.new_lambda, np.exp(T.slope), color='red', label='transmission atmosphérique droites bouguer')
    plt.errorbar(T.new_lambda, np.exp(T.slope), xerr=None, yerr=T.err_slope * np.exp(T.slope), fmt='none', capsize=1,
                 ecolor='red', zorder=2, elinewidth=2)
    plt.fill_between(T.new_lambda, np.exp(T.slope) + T.err_slope * np.exp(T.slope),
                     np.exp(T.slope) - T.err_slope * np.exp(T.slope), color='red')

    if T.order2:
        plt.plot(T.new_lambda, np.exp(T.slope2), color='green', label='transmission atmosphérique correction ordre2')
        plt.errorbar(T.new_lambda, np.exp(T.slope2), xerr=None, yerr=T.err_slope2 * np.exp(T.slope2), fmt='none',
                     capsize=1,
                     ecolor='green', zorder=2, elinewidth=2)
        plt.fill_between(T.new_lambda, np.exp(T.slope2) + T.err_slope2 * np.exp(T.slope2),
                         np.exp(T.slope2) - T.err_slope2 * np.exp(T.slope2), color='green')

    if T.sim:
        plt.title('Transmission atmosphérique, simulation, ' + T.disperseur + ', version_' + parameters.PROD_NUM, fontsize=18)
    else:
        plt.title('Transmission atmosphérique, données, ' + T.disperseur + ', version_' + parameters.PROD_NUM, fontsize=18)

    ax.get_xaxis().set_tick_params(labelsize=17)
    ax.get_yaxis().set_tick_params(labelsize=14)
    plt.xlabel('$\lambda$ (nm)', fontsize=17)
    plt.ylabel('Transmission atmosphérique', fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 15}, loc='lower right')

    if save_atmo:
        if T.sim:
            if os.path.exists(parameters.OUTPUTS_ATM_SIM):
                plt.savefig(
                    parameters.OUTPUTS_ATM_SIM + 'atm_simu, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
            else:
                os.makedirs(parameters.OUTPUTS_ATM_SIM)
                plt.savefig(
                    parameters.OUTPUTS_ATM_SIM + 'atm_simu, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
        else:
            if os.path.exists(parameters.OUTPUTS_ATM_REDUC):
                plt.savefig(
                    parameters.OUTPUTS_ATM_REDUC + 'atm_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
            else:
                os.makedirs(parameters.OUTPUTS_ATM_REDUC)
                plt.savefig(
                    parameters.OUTPUTS_ATM_REDUC + 'atm_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')

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
    Z = np.linspace(0, 2.2, 1000)

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

            plt.scatter(S.range_airmass[i], S.data_mag[i], c=[wavelength_to_rgb(T.new_lambda[i])],
                        label=f'{T.Bin[i]}-{T.Bin[i + 1]} nm',
                        marker='o', s=30)
            plt.errorbar(S.range_airmass[i], S.data_mag[i], xerr=None, yerr=S.err_mag[i], fmt='none', capsize=1,
                         ecolor=(wavelength_to_rgb(T.new_lambda[i])), zorder=2, elinewidth=2)

    if T.sim:
        plt.title('Droites de Bouguer, simulation, ' + T.disperseur + ', version_' + parameters.PROD_NUM,
                  fontsize=18)
    else:
        plt.title('Droites de Bouguer, données, ' + T.disperseur + ', version_' + parameters.PROD_NUM, fontsize=18)

    ax.get_xaxis().set_tick_params(labelsize=17)
    ax.get_yaxis().set_tick_params(labelsize=14)
    plt.xlabel('$\lambda$ (nm)', fontsize=17)
    plt.ylabel('ln(flux)', fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 12}, loc='upper right')

    if save_bouguer:
        if T.sim:
            if os.path.exists(parameters.OUTPUTS_BOUGUER_SIM):
                plt.savefig(parameters.OUTPUTS_BOUGUER_SIM + 'bouguer_simu, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
            else:
                os.makedirs(parameters.OUTPUTS_BOUGUER_SIM)
                plt.savefig(parameters.OUTPUTS_BOUGUER_SIM + 'bouguer_simu, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
        else:
            if os.path.exists(parameters.OUTPUTS_BOUGUER_REDUC):
                plt.savefig(
                    parameters.OUTPUTS_BOUGUER_REDUC + 'bouguer_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
            else:
                os.makedirs(parameters.OUTPUTS_BOUGUER_REDUC)
                plt.savefig(parameters.OUTPUTS_BOUGUER_REDUC + 'bouguer_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
    plt.show()

def plot_spec_target(Throughput, save_target):
    plt.figure(figsize=[10, 10])
    plt.plot(Throughput.lambdas_calspec, Throughput.data_calspec_org, c='red', label='CALSPEC')
    plt.plot(Throughput.new_lambda, Throughput.data_calspec, c='black', label='CALSPEC filtered')
    plt.plot(Throughput.lambdas, Throughput.data_bouguer / (Throughput.data_disp * Throughput.data_tel), c='blue',
             label='CALSPEC exact')
    plt.axis([Throughput.lambda_min, Throughput.lambda_max, 0, max(Throughput.data_calspec_org) * 1.1])
    plt.xlabel('$\lambda$ (nm)', fontsize=13)
    plt.ylabel('erg/s/cm2/Hz', fontsize=13)
    plt.title('spectra CALSPEC: ' + Throughput.target, fontsize=16)
    plt.grid(True)
    plt.legend(prop={'size': 12}, loc='upper right')

    if save_target:
        if os.path.exists(parameters.OUTPUTS_TARGET):
            plt.savefig(parameters.OUTPUTS_TARGET + 'CALSPEC, ' + Throughput.target + '.png')
        else:
            os.makedirs(parameters.OUTPUTS_TARGET)
            plt.savefig(parameters.OUTPUTS_TARGET + 'CALSPEC, ' + Throughput.target + '.png')

    plt.show()

def plot_throughput_sim(Throughput, save_Throughput):
    T = Throughput

    gs_kw = dict(height_ratios=[4, 1], width_ratios=[1])
    fig, ax = plt.subplots(2, 1, sharex="all", figsize=[14, 12], constrained_layout=True, gridspec_kw=gs_kw)

    ax[0].scatter(T.lambdas, T.data / T.data_tel, c='black', label='T_disp Vincent', s=15, zorder=2)
    ax[0].errorbar(T.lambdas, T.data / T.data_tel, xerr=None, yerr=T.err / T.data_tel, fmt='none', capsize=1,
                   ecolor='black', zorder=2,
                   elinewidth=2)

    ax[0].scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='T_disp exacte')
    ax[0].errorbar(T.lambdas, T.data_disp, xerr=None, yerr=T.data_disp_err, fmt='none', capsize=1, ecolor='deepskyblue',
                   zorder=1,
                   elinewidth=2)

    if T.order2:
        ax[0].scatter(T.lambdas, T.data_order2 / T.data_tel, c='red', label='T_disp Vincent ordre2', s=15, zorder=2)
        ax[0].errorbar(T.lambdas, T.data_order2 / T.data_tel, xerr=None, yerr=T.err_order2 / T.data_tel, fmt='none',
                       capsize=1,
                       ecolor='red', zorder=2, elinewidth=2)

    ax[0].set_xlabel('$\lambda$ (nm)', fontsize=20)
    ax[0].set_ylabel('Transmission instrumentale', fontsize=20)
    ax[0].set_title('Transmission du ' + T.disperseur + ', version_' + parameters.PROD_NUM, fontsize=18)
    ax[0].get_xaxis().set_tick_params(labelsize=17)
    ax[0].get_yaxis().set_tick_params(labelsize=14)
    ax[0].grid(True)
    ax[0].legend(prop={'size': 22}, loc='upper right')

    """On cherche les points de la reponse ideale (celle de Sylvie) les plus proches des longueurs d'ondes de la rep
    simulee"""

    "Tableaux avec les ecarts relatifs"

    Rep_sim_norm = (T.data / (T.data_tel * T.data_disp) - 1) * 100
    if T.order2:
        Rep_sim_norm_bis = (T.data_order2 / (T.data_tel * T.data_disp) - 1) * 100
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
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_SIM + 'throughput_sim, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
        else:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_SIM)
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_SIM + 'throughput_sim, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')

    plt.show()

def plot_throughput_reduc(Throughput, save_Throughput):
    T = Throughput
    if T.disperseur == 'Thor300':
        fig = plt.figure(figsize=[15, 10])
        ax2 = fig.add_subplot(111)

        thorlab = np.loadtxt(T.rep_disp_ref)

        thorlab_data = sp.interpolate.interp1d(thorlab.T[0], thorlab.T[1], bounds_error=False,
                                               fill_value="extrapolate")(T.lambdas)

        Tinst = sp.signal.savgol_filter(T.data / thorlab_data, 81, 3)
        ax2.scatter(T.lambdas, Tinst, c='black', label='rep tel')
        ax2.errorbar(T.lambdas, Tinst, xerr=None, yerr=T.err / thorlab_data, fmt='none', capsize=1,
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
        ax2.set_title(
            "Transmission instrumentale du telescope, " + T.disperseur + ', version_' + parameters.PROD_NUM,
            fontsize=22)
        ax2.get_xaxis().set_tick_params(labelsize=20)
        ax2.get_yaxis().set_tick_params(labelsize=20)
        ax2.legend(prop={'size': 17}, loc='upper right')
        plt.grid(True)
        fig.tight_layout()
        if save_Throughput:
            if os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC):
                plt.savefig(
                    parameters.OUTPUTS_THROUGHPUT_REDUC + 'ctio_throughput, version_' + parameters.PROD_NUM + '.png')
                fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'ctio_thrpoughput_basethor300'), 'w')

                for i in range(len(T.lambdas)):
                    fichier.write(
                        str(T.lambdas[i]) + '\t' + str(Tinst_order2[i]) + '\t' + str(Tinst_order2_err[i]) + '\n')
                fichier.close()
            else:
                os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
                fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'ctio_throughput_basethor300.txt'), 'w')

                for i in range(len(T.lambdas)):
                    fichier.write(
                        str(T.lambdas[i]) + '\t' + str(Tinst_order2[i]) + '\t' + str(Tinst_order2_err[i]) + '\n')
                fichier.close()
                plt.savefig(
                    parameters.OUTPUTS_THROUGHPUT_REDUC + 'ctio_throughput, version_' + parameters.PROD_NUM + '.png')
        plt.show()

    fig = plt.figure(figsize=[15, 10])
    ax2 = fig.add_subplot(111)

    ax2.scatter(T.lambdas, T.data / T.data_tel, c='black', marker='.', label='Tinst_Vincent')
    ax2.errorbar(T.lambdas, T.data / T.data_tel, xerr=None, yerr=T.err / T.data_tel, fmt='none', capsize=1,
                 ecolor='black', zorder=1, elinewidth=2)

    if T.disperseur == 'Thor300':
        T.data_disp = thorlab_data
        T.data_disp_err = thorlab_data * 0.01
        ax2.scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='Banc_LPNHE')
    else:
        ax2.scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='Tinst_Sylvie')
    ax2.errorbar(T.lambdas, T.data_disp, xerr=None, yerr=T.data_disp_err, fmt='none', capsize=1, ecolor='deepskyblue',
                 zorder=1,
                 elinewidth=2)

    if T.order2:
        ax2.scatter(T.lambdas, T.data_order2 / T.data_tel, c='red', marker='.', label='Tinst_Vincent_ordre2')
        ax2.errorbar(T.lambdas, T.data_order2 / T.data_tel, xerr=None, yerr=T.err_order2 / T.data_tel,
                     fmt='none', capsize=1, ecolor='red', zorder=1, elinewidth=2)

    ax2.set_xlabel('$\lambda$ (nm)', fontsize=24)
    ax2.set_ylabel("Transmission disperseur", fontsize=22)
    ax2.set_title("Transmission instrumentale du, " + T.disperseur + ', version_' + parameters.PROD_NUM,
                  fontsize=22)
    ax2.get_xaxis().set_tick_params(labelsize=20)
    ax2.get_yaxis().set_tick_params(labelsize=20)
    ax2.legend(prop={'size': 17}, loc='upper right')
    plt.grid(True)
    fig.tight_layout()

    if save_Throughput:
        if os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC):
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_REDUC + 'throughput_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.png')
        else:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
            plt.savefig(parameters.OUTPUTS_THROUGHPUT_REDUC + 'throughput_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM+ '.png')
    plt.show()
