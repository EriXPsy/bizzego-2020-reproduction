from config import PATH

import os
import gc

import pyphysio as ph

#%%
ECGDIR = f'{PATH}/data/pkl/ecg'
OUTDIR = f'{PATH}/data/pkl/ibi'

MANUAL = True #manually correct the detected ibi

#%%
# filters and estimators
lp_filter = ph.IIRFilter(fp=48, fs=50, ftype='ellip')
hp_filter = ph.IIRFilter(fp=10, fs=8.5, ftype='ellip')

ibi_detector = ph.BeatFromECG(bpm_max=140, k=0.7)

#%%
subjects = os.listdir(ECGDIR)

#%%
for SUB in subjects:
    try:
        ecg = ph.from_pickle(f'{ECGDIR}/{SUB}') #load
        ecg = lp_filter(hp_filter(ecg)) #filter
        
        ibi = ibi_detector(ecg) #extract ibi
        id_bad_ibi = ph.BeatOutliers(sensitivity=0.75)(ibi) #detect outliers
        ibi = ph.FixIBI(id_bad_ibi)(ibi) #remove outliers

        if MANUAL:
            ibi = ph.Annotate(ecg, ibi).ibi_ok
        
        ibi.to_pickle(f'{OUTDIR}/{SUB}')
        del ecg, ibi
        gc.collect()
    except:
        print(SUB)
        
