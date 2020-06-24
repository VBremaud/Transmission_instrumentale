# coding: utf8

import numpy as np  # calculs utilisant C
import parameters
from scipy import signal  # filtre savgol pour enlever le bruit
from scipy.interpolate import interp1d  # interpolation
import scipy as sp  # calculs
from scipy import misc

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

    intensite_obs_savgol = sp.signal.savgol_filter(intensite_obs, parameters.SAVGOL_LENGTH_DL,
                                                   parameters.SAVGOL_ORDER_DL)  # filtre savgol (enlève le bruit)
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
    intensite_obs_savgol2 = sp.signal.savgol_filter(intensite_obs[k:], parameters.SAVGOL_LENGTH_GLOB,
                                                    parameters.SAVGOL_ORDER_GLOB)

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
            while lambda_complet[k] - lambda_complet[i] < parameters.HALF_LENGTH_MAX and k < len(
                    lambda_complet) - moy_raies:
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

    for i in range(parameters.AROUND_LINES, len(Raies) - parameters.AROUND_LINES - 1):
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

    return data_filt


def smooth(x, window_len, window, sigma=1):
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    elif window == 'gaussian':
        if sigma == 0:
            return x
        else:
            w = signal.gaussian(window_len, sigma)
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    if window_len % 2 == 0:  # even case
        y = y[int(window_len / 2):-int(window_len / 2) + 1]
        return y
    else:  # odd case
        y = y[int(window_len / 2 - 1) + 1:-int(window_len / 2 - 1) - 1]
        return y