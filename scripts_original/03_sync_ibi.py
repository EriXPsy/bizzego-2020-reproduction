from config import PATH

import os
import pandas as pd
import pyphysio as ph
from pyphysio.metrics.Metrics import compute_IAAFT_surrogates, compute_distances_golland
import numpy as np

def preprocess_signal(signal):
    signal = lp_filter(signal.resample(F_RESAMP))
    return(signal)

def standard_scale(x):
    return( (x-np.mean(x))/np.std(x))

#%%
# folders
DATADIR = f'{PATH}/data/pkl/ibi'
OUTDIR = f'{PATH}/data/synchrony/cc_IBI_complete'
TRGDIR = f'{PATH}/data/pkl/trg'

video_names    = ['HS',    'TIT', 'WD',   'RELAX', 'NH',   'PEN']
video_emotions = ['EMBRS', 'SAD', 'FEAR', 'CALMNESS', 'LOVE', 'PRIDE']

# preprocessing functions and parameters
lp_filter = ph.IIRFilter(fp = 0.04, fs= 0.1) # see golland2014, page 2, 'Physiological measures: Collection and preprocessing'
F_RESAMP = 2 # see golland2014
CONV_WINDOW = 5

# distance parameters
DISTANCE = 'cc'
MAX_DELAY = 10
LAG = MAX_DELAY*F_RESAMP
STANDARDIZE = True
NORMALIZE = True

#%%
# create relation folders in OUTDIR
for R in ['U', 'F', 'L']:
    os.makedirs(f'{OUTDIR}/{R}')

#%% LOAD ALL DATA
dyad_list = np.unique([x.split('_')[0] for x in os.listdir(DATADIR)])
relation = np.array([D[0] for D in dyad_list])

data = {}
for RELATION in ['U', 'F', 'L']:
    selected_dyads = dyad_list[relation==RELATION]
    data_rel = {}
    for D in selected_dyads:
        print(D)
        signal_M = ph.from_pickle(f'{DATADIR}/{D}_M.pkl')
        signal_M = preprocess_signal(signal_M)
        
        signal_F = ph.from_pickle(f'{DATADIR}/{D}_F.pkl')
        signal_F = preprocess_signal(signal_F)
        
        trg = ph.from_pickle(f'{TRGDIR}/{D}.pkl')
        t_trg = trg.get_times()
        
        data_dyad = {}
        for v in range(6):
            V = v+1
            t_video = t_trg[np.where(trg==V)[0]]
            if len(t_video)>0: 
                t_start = t_video[0]+10
                t_stop = t_video[-1]
                
                signal_video_M = standard_scale(signal_M.segment_time(t_start, t_stop).get_values())
                if len(signal_video_M)<235*2:
                    signal_video_M = None
                    signal_video_M_s = None
                else:
                    signal_video_M_s = compute_IAAFT_surrogates(signal_video_M)
                    signal_video_M_s = np.convolve(signal_video_M_s, np.ones(int(CONV_WINDOW*F_RESAMP))/int(CONV_WINDOW*F_RESAMP))
                    
                signal_video_F = standard_scale(signal_F.segment_time(t_start, t_stop).get_values())
                if len(signal_video_F)<235*2:
                    signal_video_F = None
                    signal_video_F_s = None
                else:
                    signal_video_F_s = compute_IAAFT_surrogates(signal_video_F)
                    signal_video_F_s = np.convolve(signal_video_F_s, np.ones(int(CONV_WINDOW*F_RESAMP))/int(CONV_WINDOW*F_RESAMP))
                
                data_dyad[V] = {'M': [signal_video_M, signal_video_M_s], 'F': [signal_video_F, signal_video_F_s]}
        
        data_rel[D] = data_dyad
        
    data[RELATION] = data_rel

#%% GROUP SUBJECTS BY RELATION, GENDER AND VIDEO
group_1 = {}
group_2 = {}

for RELATION in ['U', 'F', 'L']:
    selected_dyads = dyad_list[relation==RELATION]
    group_1_R = {}
    group_2_R = {}
    for v in range(len(video_names)):
        group_1_v = []
        group_2_v = []
        
        V=v+1
        for dyad in selected_dyads:
            group_1_v.append([data[RELATION][dyad][V]['M'][0], data[RELATION][dyad][V]['M'][1]])
            group_2_v.append([data[RELATION][dyad][V]['F'][0], data[RELATION][dyad][V]['F'][1]])
        group_1_R[V] = group_1_v
        group_2_R[V] = group_2_v
    group_1[RELATION] = group_1_R
    group_2[RELATION] = group_2_R

for i_r, RELATION in enumerate(['U', 'F', 'L']):
    selected_dyads = dyad_list[relation==RELATION]
    
    group_1_R = group_1[RELATION]
    group_2_R = group_2[RELATION]
    
    copresence = []
    stimulus = []
    surrogate = []
    
    for v in range(len(video_names)):
        V=v+1
        group_1_v = group_1_R[V]
        group_2_v = group_2_R[V]
        
        copresence_v, stimulus_v, _, _, surrogate_v = compute_distances_golland(group_1_v, group_2_v, DISTANCE, LAG, False, NORMALIZE)
        copresence.append(copresence_v)
        stimulus.append(stimulus_v)
        surrogate.append(surrogate_v)

    copresence_pd = pd.DataFrame(np.array(copresence), columns=selected_dyads, index = video_names)
    stimulus_pd = pd.DataFrame(np.array(stimulus), index = video_names)
    surrogate_pd = pd.DataFrame(np.array(surrogate), index = video_names)
    
    copresence_pd.to_csv(f'{OUTDIR}/{RELATION}/copresence.csv')
    stimulus_pd.to_csv(f'{OUTDIR}/{RELATION}/stimulus.csv')
    surrogate_pd.to_csv(f'{OUTDIR}/{RELATION}/surrogate.csv')