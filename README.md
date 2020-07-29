Structure des dossiers avant la première exécution: 

runTinst.py 

TransmissionInstrumentale.py

SpectrumRangeAirmass.py

SpectrumAirmassFixed.py

parameters.py

sps/ (vide)

prod/data_txt/ (vide)
    
prod/data/(lien symbolique vers CTIO...)
    
throughput/

tests/data/reduc_20170530_060_atmsim.fits

Pour lancer: runTinst.py (lance tout les fits d'une prod d'abord simu puis données), -D spécifie un disperseur, -s spécifie s'il s'agit d'une simulation. 
