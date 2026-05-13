from config import PATH

import os
import pyphysio as ph
import numpy as np
import pandas as pd

lp_filter = ph.IIRFilter(fp = 0.04, fs= 0.1)

#%%
IBIDIR = f'{PATH}/data/pkl/ibi'
TRGDIR = f'{PATH}/data/pkl/trg'
VIDEOS = ['HS', 'TIT', 'WD', 'NEUT', 'NH', 'PEN']

F_RESAMP = 2

#%%
def load_video(dyad, gender, video):
    ibi = ph.from_pickle(f'{IBIDIR}/{dyad}_{gender}.pkl')
    trg = ph.from_pickle(f'{TRGDIR}/{dyad}.pkl')
    t_trg = trg.get_times()
    
    t_video = t_trg[np.where(trg==video)[0]]
    t_start = t_video[0]+10
    t_stop = t_video[-1]
    ibi = ibi.segment_time(t_start, t_stop)
    return(ibi)

#%%
dyads = np.unique([x.split('_')[0] for x in os.listdir(IBIDIR)])
relation = np.array([D[0] for D in dyads])
exclude = ['L04']

#%%
data_ =  []
for i_v, VIDEO in enumerate(VIDEOS):
    
    for i_r, REL in enumerate(['U', 'F', 'L'] ):
        #%
        selected_dyads = dyads[relation==REL]
        
        for D in selected_dyads:
            if D not in exclude:
                ibi_m = np.mean(load_video(D, 'M', i_v+1))
                ibi_f = np.mean(load_video(D, 'F', i_v+1))
                data_.append([VIDEO, REL, D, 'M', ibi_m])
                data_.append([VIDEO, REL, D, 'F', ibi_f])

#%%
data_pd = pd.DataFrame(data_)
data_pd.columns = ['video', 'group', 'dyad', 'gender', 'ibi_mean']
data_pd.to_csv(f'{PATH}/data/mean_ibi.csv')
