# coding: utf8

from scipy import signal
from scipy.interpolate import interp1d
from SpectrumRangeAirmass import *

class TransmissionInstrumentale:

    def __init__(self):

        self.target = parameters.target
        self.disperseur = parameters.DISP
        self.binwidths = parameters.BINWIDTHS
        self.lambda_min = parameters.LAMBDA_MIN
        self.lambda_max = parameters.LAMBDA_MAX
        self.Bin = parameters.BIN
        self.new_lambda = parameters.NEW_LAMBDA
        self.sim = parameters.SIM
        self.lambdas_calspec = []
        self.data_calspec = []
        self.data_order2 = []
        self.data_tel = []
        self.data_tel_err = []
        self.data_disp = []
        self.data_disp_err = []
        self.lambdas = []
        self.ord2 = []
        self.err_ord2 = []
        self.err_order2 = []
        self.file_calspec = glob.glob(parameters.PROD_TXT + "/sim*spectrum.txt")[0]
        if self.sim:
            self.rep_tel_name = os.path.join(parameters.THROUGHPUT_DIR, parameters.THROUGHPUT_SIM)
        else:
            self.rep_tel_name = os.path.join(parameters.THROUGHPUT_DIR, parameters.THROUGHPUT_REDUC)
        if self.sim:
            self.rep_disp_ref = os.path.join(parameters.THROUGHPUT_DIR, parameters.DISPERSER_REF_SIM)
        elif self.disperseur == parameters.DISPERSER_REF:
            self.rep_disp_ref = os.path.join(parameters.THROUGHPUT_DIR, parameters.DISPERSER_REF_BANC)
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
        self.data_calspec = np.array(data)
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

    def calcul_throughput(self, spectrumrangeairmass):
        self.lambdas = self.new_lambda
        if (self.disperseur == parameters.DISPERSER_REF and self.sim == False) or self.sim == True:
            disp = np.loadtxt(self.rep_disp_ref)
            Data_disp = sp.interpolate.interp1d(disp.T[0], disp.T[1], kind="linear", bounds_error=False,
                                                fill_value="extrapolate")
            try:
                Err_disp = sp.interpolate.interp1d(disp.T[0], disp.T[2], kind="linear", bounds_error=False,
                                               fill_value="extrapolate")
            except:
                Err_disp = sp.interpolate.interp1d(disp.T[0], disp.T[1]/100, kind="linear", bounds_error=False,
                                                   fill_value="extrapolate")
            self.data_disp = Data_disp(self.lambdas)
            self.data_disp_err = Err_disp(self.lambdas)

        tel = np.loadtxt(self.rep_tel_name)
        Data_tel = sp.interpolate.interp1d(tel.T[0], tel.T[1], kind="linear", bounds_error=False,
                                           fill_value="extrapolate")
        Err_tel = sp.interpolate.interp1d(tel.T[0], tel.T[2], kind="linear", bounds_error=False,
                                          fill_value="extrapolate")

        self.ord2, self.err_order2 = spectrumrangeairmass.megafit_emcee()
        self.data_order2 = self.ord2 / self.data_calspec
        self.err_order2 = self.err_order2 / self.data_calspec

        Data = sp.interpolate.interp1d(self.lambdas, self.data_order2, kind="linear", bounds_error=False,
                                       fill_value="extrapolate")
        Err = sp.interpolate.interp1d(self.lambdas, self.err_order2, kind="linear", bounds_error=False,
                                      fill_value="extrapolate")
        self.lambdas = self.new_lambda
        self.data_order2 = Data(self.lambdas)
        self.err_order2 = Err(self.lambdas)

        if self.sim == False and self.disperseur == parameters.DISPERSER_REF:
            self.data_order2[21:-21] = sp.signal.savgol_filter(self.data_order2[21:-21], 11, 2)

        self.data_tel = Data_tel(self.lambdas)
        self.data_tel_err = Err_tel(self.lambdas)

        if parameters.plot_fitspectrum:
            spectrumrangeairmass.fit_spectrum()


def plot_atmosphere(Throughput, save_atmo, sim):
    fig = plt.figure(figsize=[12, 7])
    ax = fig.add_subplot(111)

    T = Throughput
    Lambda = np.arange(T.new_lambda[0], T.new_lambda[-1], 1)

    atmgrid = AtmosphereGrid(filename="tests/data/reduc_20170530_060_atmsim.fits")
    if sim:
        b = atmgrid.simulate(300, 5, 0.03)
        plt.plot(Lambda, np.exp(np.log(b(Lambda)) / 1.047), color='blue', label='transmission libradtran injectée')

    c = atmgrid.simulate(*T.params_atmo)
    d = atmgrid.simulate(*(T.params_atmo+T.err_params_atmo))
    e= atmgrid.simulate(*(T.params_atmo-T.err_params_atmo))
    plt.plot(Lambda, np.exp(np.log(c(Lambda)) / 1.047), color='red', label='Atmospheric transmission fit')
    plt.plot(Lambda, np.exp(np.log(d(Lambda)) / 1.047), color='darkred', linestyle='--')
    plt.plot(Lambda, np.exp(np.log(e(Lambda)) / 1.047), color='darkred', linestyle='--')

    if T.sim:
        plt.title('Atmospheric transmission with '+parameters.DISP+' simulations'+ ', version_' + parameters.PROD_NUM,fontsize=25)
    else:
        plt.title('Atmospheric transmission with '+parameters.DISP+', version_' + parameters.PROD_NUM,fontsize=25)

    ax.get_xaxis().set_tick_params(labelsize=18)
    ax.get_yaxis().set_tick_params(labelsize=17)
    plt.axis([370,980,0.38,1.3])
    plt.xlabel('$\lambda$ (nm)', fontsize=17)
    plt.ylabel('Atmospheric transmission', fontsize=17)
    plt.grid(True)
    plt.legend(prop={'size': 16}, loc='lower right')
    fig.tight_layout()
    if save_atmo:
        if os.path.exists(parameters.OUTPUTS_ATM_SIM) == False:
            os.makedirs(parameters.OUTPUTS_ATM_SIM)
        if T.sim:
            plt.savefig(
                    parameters.OUTPUTS_ATM_SIM + 'atm_simu, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.pdf')
        else:
             plt.savefig(
                    parameters.OUTPUTS_ATM_REDUC + 'atm_reduc, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.pdf')

    plt.show()

def plot_spec_target(Throughput, save_target):
    plt.figure(figsize=[10, 10])
    plt.plot(Throughput.lambdas_calspec, Throughput.data_calspec_org, c='red', label='CALSPEC')
    plt.plot(Throughput.new_lambda, Throughput.data_calspec, c='black', label='CALSPEC filtered')
    plt.plot(Throughput.lambdas, Throughput.data_bouguer / (Throughput.data_disp * Throughput.data_tel), c='blue',
             label='CALSPEC exact')
    plt.axis([Throughput.lambda_min, Throughput.lambda_max, 0, max(Throughput.data_calspec_org) * 1.1])
    plt.xlabel('$\lambda$ (nm)', fontsize=13)
    plt.ylabel('ADU', fontsize=13)
    plt.title('spectra CALSPEC: ' + Throughput.target, fontsize=16)
    plt.grid(True)
    plt.legend(prop={'size': 12}, loc='upper right')

    if save_target:
        if os.path.exists(parameters.OUTPUTS_TARGET) == False:
            os.makedirs(parameters.OUTPUTS_TARGET)
        plt.savefig(parameters.OUTPUTS_TARGET + 'CALSPEC, ' + Throughput.target + '.pdf')
    plt.show()

def plot_throughput_sim(Throughput, save_Throughput):
    T = Throughput

    gs_kw = dict(height_ratios=[4, 1], width_ratios=[1])
    fig, ax = plt.subplots(2, 1, sharex="all", figsize=[15, 8], constrained_layout=True, gridspec_kw=gs_kw)

    ax[0].scatter(T.lambdas, T.data_disp, c='deepskyblue', marker='.', label='Truth transmission')
    ax[0].errorbar(T.lambdas, T.data_disp, xerr=None, yerr=T.data_disp_err, fmt='none', capsize=1, ecolor='deepskyblue',
                   zorder=1,
                   elinewidth=2)

    ax[0].scatter(T.lambdas, T.data_order2 / T.data_tel, c='red', label='fit photometric night', marker='x',s=15, zorder=2)
    ax[0].errorbar(T.lambdas, T.data_order2 / T.data_tel, xerr=None, yerr=T.err_order2 / T.data_tel, fmt='none',
                       capsize=1,
                       ecolor='red', zorder=2, elinewidth=2)

    ax[0].set_xlabel('$\lambda$ [nm]]', fontsize=22)
    ax[0].set_ylabel('Grating transmission', fontsize=22)
    ax[0].set_title('Grating transmission of the '+parameters.DISP+' with simulations, version_' + parameters.PROD_NUM,fontsize=24)
    ax[0].get_xaxis().set_tick_params(labelsize=19)
    ax[0].get_yaxis().set_tick_params(labelsize=16)
    ax[0].grid(True)
    ax[0].legend(prop={'size': 20}, loc='upper right')

    """On cherche les points de la reponse ideale (celle de Sylvie) les plus proches des longueurs d'ondes de la rep
    simulee"""

    "Tableaux avec les ecarts relatifs"

    Rep_sim_norm_bis = (T.data_order2 / (T.data_tel * T.data_disp) - 1) * 100
    zero = np.zeros(1000)

    X_2_bis = 0
    for i in range(len(T.lambdas)):
        X_2_bis += Rep_sim_norm_bis[i] ** 2

    X_2_bis = np.sqrt(X_2_bis / len(Rep_sim_norm_bis))

    ax[1].plot(np.linspace(T.lambdas[0], T.lambdas[-1], 1000), zero, c='black')

    NewErr_bis = T.err_order2 / (T.data_tel * T.data_disp) * 100

    ax[1].scatter(T.lambdas, Rep_sim_norm_bis, c='red', marker='x')
    ax[1].errorbar(T.lambdas, Rep_sim_norm_bis, xerr=None, yerr=NewErr_bis, fmt='none', capsize=1,
                       ecolor='red', zorder=2, elinewidth=2)
    ax[1].set_xlabel('$\lambda$ [nm]', fontsize=22)
    ax[1].set_ylabel('Residuals [%]', fontsize=16)
    ax[1].get_xaxis().set_tick_params(labelsize=19)
    ax[1].get_yaxis().set_tick_params(labelsize=11)

    ax[1].grid(True)
    ax[1].text(850, max(Rep_sim_norm_bis) * 3 / 4, '$\sigma$= ' + str(X_2_bis)[:4] + '%', color='red', fontsize=20)

    fig.tight_layout()

    plt.subplots_adjust(wspace=0, hspace=0)
    if save_Throughput:
        if os.path.exists(parameters.OUTPUTS_THROUGHPUT_SIM) == False:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_SIM)
        plt.savefig(parameters.OUTPUTS_THROUGHPUT_SIM + 'throughput_sim, ' + T.disperseur + ', version_' + parameters.PROD_NUM + '.pdf')
    plt.show()

def plot_throughput_reduc(Throughput, save_Throughput):
    T = Throughput
    if T.disperseur == parameters.DISPERSER_REF:
        fig = plt.figure(figsize=[15, 10])
        ax2 = fig.add_subplot(111)

        disp_ref = np.loadtxt(T.rep_disp_ref)

        disp_ref_data = sp.interpolate.interp1d(disp_ref.T[0], disp_ref.T[1], bounds_error=False,
                                               fill_value="extrapolate")(T.lambdas)

        Tinst_order2 = sp.signal.savgol_filter(T.data_order2 / disp_ref_data, 17, 3)
        Tinst_order2_err = sp.signal.savgol_filter(T.err_order2 / disp_ref_data, 17, 3)
        ax2.scatter(T.lambdas, Tinst_order2, c='red', label='Tinst_Mega_fit')
        ax2.errorbar(T.lambdas, Tinst_order2, xerr=None, yerr=Tinst_order2_err, fmt='none', capsize=1,
                         ecolor='red', zorder=2, elinewidth=2)
        ax2.scatter(T.lambdas, T.data_tel, c='blue', label=parameters.THROUGHPUT_REDUC)
        ax2.errorbar(T.lambdas, T.data_tel, xerr=None, yerr=T.data_tel_err, fmt='none', capsize=1,
                     ecolor='blue', zorder=2, elinewidth=2)

        #TO DO A COMPARISON WITH CBP
        """
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
        Y = interpolation(a.T[0])
        Ynew = [a.T[1][i] / Y[i] for i in range(len(Y))]
        ax2.scatter(a.T[0], Ynew / Ynew[int(len(Ynew) / 2)] * max(T.data_tel), c='purple', marker='.', label='T_inst CBP')
        """

        ax2.set_xlabel('$\lambda$ (nm)', fontsize=24)
        ax2.set_ylabel("Instrumental transmission", fontsize=22)
        ax2.set_title(
            "Instrumental transmission of the telescope, " + T.disperseur + ', version_' + parameters.PROD_NUM,
            fontsize=22)
        ax2.get_xaxis().set_tick_params(labelsize=20)
        ax2.get_yaxis().set_tick_params(labelsize=20)
        ax2.legend(prop={'size': 17}, loc='upper right')
        plt.grid(True)
        fig.tight_layout()
        if save_Throughput:
            if os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC) == False:
                os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
            plt.savefig(
                parameters.OUTPUTS_THROUGHPUT_REDUC + 'ctio_throughput'+'_base'+T.disperseur+'_prod' + parameters.PROD_NUM + '.pdf')
            fichier = open(os.path.join(parameters.THROUGHPUT_DIR, 'ctio_throughput'+'_base'+T.disperseur+'_prod' + parameters.PROD_NUM + '.txt'), 'w')
            for i in range(len(T.lambdas)):
                fichier.write(
                    str(T.lambdas[i]) + '\t' + str(Tinst_order2[i]) + '\t' + str(Tinst_order2_err[i]) + '\n')
            fichier.close()
        plt.show()

    fig = plt.figure(figsize=[12, 7])
    ax2 = fig.add_subplot(111)

    if parameters.COMPARISON_BANC:
        disp = np.loadtxt(os.path.join(parameters.THROUGHPUT_DIR,parameters.DISPERSER_BANC))
        ax2.scatter(disp.T[0], disp.T[1], c='blue', marker='.', label='Measurement on optical test bench')
        ax2.errorbar(disp.T[0], disp.T[1], xerr=None, yerr=disp.T[2],
                     fmt='none', capsize=1, ecolor='blue', zorder=1, elinewidth=2)

    ax2.scatter(T.lambdas, T.data_order2 / T.data_tel, c='red', marker='.', label='Photometric_night_fit')
    ax2.errorbar(T.lambdas, T.data_order2 / T.data_tel, xerr=None, yerr=T.err_order2 / T.data_tel,
                     fmt='none', capsize=1, ecolor='red', zorder=1, elinewidth=2)
    Tinst_order2 = T.data_order2 / T.data_tel
    Tinst_order2_err = T.err_order2 / T.data_tel
    ax2.set_xlabel('$\lambda$ [nm]', fontsize=22)
    ax2.set_ylabel("Grating transmission", fontsize=22)
    ax2.set_title("Instrument Transmission of the "+ T.disperseur + ', version_' + parameters.PROD_NUM,
                  fontsize=24)
    ax2.get_xaxis().set_tick_params(labelsize=18)
    ax2.get_yaxis().set_tick_params(labelsize=18)
    ax2.legend(prop={'size': 15}, loc='upper left')
    plt.grid(True)
    fig.tight_layout()

    if save_Throughput:
        if os.path.exists(parameters.OUTPUTS_THROUGHPUT_REDUC) == False:
            os.makedirs(parameters.OUTPUTS_THROUGHPUT_REDUC)
        plt.savefig(
                parameters.OUTPUTS_THROUGHPUT_REDUC + T.disperseur+'_basectio_'+parameters.DISPERSER_REF+', version_' + parameters.PROD_NUM + '.pdf')
        fichier = open(os.path.join(parameters.THROUGHPUT_DIR, T.disperseur+'_basectio_'+parameters.DISPERSER_REF+', version_' + parameters.PROD_NUM + '.txt'), 'w')
        for i in range(len(T.lambdas)):
            fichier.write(str(T.lambdas[i]) + '\t' + str(Tinst_order2[i]) + '\t' + str(Tinst_order2_err[i]) + '\n')
        fichier.close()

    plt.show()
