# coding: utf8

import glob
from astropy.io import fits
from spectractor.extractor.spectrum import Spectrum  # importation des spectres
import spectractor.parameters as parameterss
from spectractor.simulation.adr import adr_calib
from TransmissionInstrumentale import *

def convert_from_fits_to_txt(prod_name, prod_txt):
    to_convert_list = []
    Lsimutxt = glob.glob(prod_txt + "/sim*spectrum.txt")
    Lreductxt = glob.glob(prod_txt + "/reduc*spectrum.txt")
    Lsimufits = glob.glob(prod_name + "/sim*spectrum.fits")
    Lreducfits = glob.glob(prod_name + "/reduc*spectrum.fits")

    Ldefaut = glob.glob(prod_name + "/*bestfit*.txt") + glob.glob(prod_name + "/*fitting*.txt") + glob.glob(
        prod_name + "/*A2=0*.fits") + glob.glob(prod_name + "/*20170530_201_spectrum*") + glob.glob(
        prod_name + "/*20170530_200_spectrum*") + glob.glob(prod_name + "/*20170530_205_spectrum*")

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
            airmass = s.header["AIRMASS"]
            TARGETX = s.header["TARGETX"]
            TARGETY = s.header["TARGETY"]
            D2CCD = s.header["D2CCD"]
            PIXSHIFT = s.header["PIXSHIFT"]
            ROTANGLE = s.header["ROTANGLE"]
            psf_transverse = s.chromatic_psf.table['fwhm']
            PARANGLE = s.header["PARANGLE"]
            PSF_REG = s.header['PSF_REG']

            x0 = [TARGETX, TARGETY]
            disperser = s.disperser
            distance = disperser.grating_lambda_to_pixel(s.lambdas, x0=x0, order=1)
            distance += adr_calib(s.lambdas, s.adr_params, parameterss.OBS_LATITUDE, lambda_ref=s.lambda_ref)
            distance -= adr_calib(s.lambdas / 2, s.adr_params, parameterss.OBS_LATITUDE, lambda_ref=s.lambda_ref)
            lambdas_order2 = disperser.grating_pixel_to_lambda(distance, x0=x0, order=2)

            print(to_convert_list[i][:len(to_convert_list[i]) - 5])
            disperseur = s.disperser_label
            star = s.header['TARGET']
            lambda_obs = s.lambdas
            intensite_obs = s.data
            intensite_err = s.err

            cov = s.cov_matrix
            if s.target.wavelengths == []:
                print('CALSPEC error')

            ###### ATTENTION PSF_REG ######
            else:
                lambda_reel = s.target.wavelengths[0]
                intensite_reel = s.target.spectra[0]
                tag = to_convert_list[i].split('/')[-1]
                fichier = open(os.path.join(prod_txt, tag.replace('fits', 'txt')), 'w')
                fichier.write('#' + '\t' + star + '\t' + disperseur + '\t' + str(airmass) + '\t' + str(
                    TARGETX) + '\t' + str(TARGETY) + '\t' + str(D2CCD) + '\t' + str(PIXSHIFT) + '\t' + str(
                    ROTANGLE) + '\t' + str(PARANGLE) + '\t' + str(PSF_REG) + '\n')
                for j in range(len(lambda_reel)):
                    if len(lambda_obs) > j:
                        if len(psf_transverse) > j:
                            fichier.write(str(lambda_reel[j]) + '\t' + str(intensite_reel[j]) + '\t' + str(
                                lambda_obs[j]) + '\t' + str(intensite_obs[j]) + '\t' + str(
                                intensite_err[j]) + '\t' + str(lambdas_order2[j]) + '\t' + str(
                                psf_transverse[j]) + '\n')
                        else:
                            fichier.write(str(lambda_reel[j]) + '\t' + str(intensite_reel[j]) + '\t' + str(
                                lambda_obs[j]) + '\t' + str(intensite_obs[j]) + '\t' + str(
                                intensite_err[j]) + '\t' + str(lambdas_order2[j]) + '\n')

                    else:
                        fichier.write(str(lambda_reel[j]) + '\t' + str(intensite_reel[j]) + '\n')

                fichier.close()

                np.save(os.path.join(prod_txt, tag.replace('fits', 'npy')),cov)
        return True
    else:
        print('already done')
        return True, Lsimutxt, Lreductxt


def extract_throughput(prod_name, sim, disperseur, prod_sim, prod_reduc, target="HD111980", order2=False,
                       mega_fit=False,
                       plot_atmo=False, plot_bouguer=False, plot_specs=False, plot_target=False, plot_filt=False,
                       plot_Throughput=True,
                       save_atmo=False, save_bouguer=False, save_target=False, save_Throughput=False,
                       save_filter=False):
    spectrumrangeairmass = SpectrumRangeAirmass(prod_name=prod_name, sim=sim, disperseur=disperseur, target=target,
                                                plot_specs=plot_specs, prod_sim=prod_sim, prod_reduc=prod_reduc)
    Throughput = TransmissionInstrumentale(prod_name=prod_name, sim=sim, disperseur=disperseur, target=target,
                                           order2=order2, mega_fit=mega_fit, plot_filt=plot_filt,
                                           save_filter=save_filter, prod=prod_sim)
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


def prod_analyse(prod_name, prod_txt, data='all'):
    CFT = convert_from_fits_to_txt(prod_name, prod_txt)
    if CFT[0]:
        if data == 'all' or data == 'sim':
            for disperser in parameters.DISPERSER:
                extract_throughput(prod_txt, True, disperser, CFT[1], CFT[2], plot_bouguer=False, plot_atmo=True,
                                   plot_target=False, save_atmo=True, save_bouguer=False, save_target=False,
                                   save_Throughput=False, order2=True, mega_fit=False)
        if data == 'all' or data == 'reduc':
            for disperser in parameters.DISPERSER:
                extract_throughput(prod_txt, False, disperser, CFT[1], CFT[2], plot_bouguer=False, plot_atmo=False,
                                   plot_target=False, save_atmo=False, save_bouguer=False, save_target=False,
                                   save_Throughput=False, order2=True, mega_fit=True)

    else:
        print('relaunch, convert_fits_to_txt step')