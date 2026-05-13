from config import PATH

import pandas as pd
import scipy.stats as stat
import numpy as np


def p2sig(p):
    if p<0.001:
        return('***')
    elif p<0.01:
        return('**')
    elif p<0.05:
        return('*')
    else:
        return('n.s.')

def noNan(x):
    return(x[~np.isnan(x)])
    
def isNormal(x, alpha=0.01):
    W,p = stat.shapiro(x)
    return(p>0.05)
    
def test(x,y):
    T,p = stat.mannwhitneyu(x, y, alternative = 'less')
    sig = p2sig(p)
    return(sig, p, T, 'U-test')

#%%
video_names = ['HS', 'TIT', 'WD', 'RELAX', 'NH', 'PEN']
video_emotions = ['EMBRS', 'SAD', 'FEAR', 'CALMNESS', 'LOVE', 'PRIDE']

DIST = 'cc_IBI_complete'
DIST_DIR = f'{PATH}/data/synchrony/{DIST}'

#%%
for i_r, RELATION in enumerate(['U', 'F', 'L']):
    copresence = pd.read_csv(f'{DIST_DIR}/{RELATION}/copresence.csv', index_col=0)
    stimulus = pd.read_csv(f'{DIST_DIR}/{RELATION}/stimulus.csv', index_col=0)
    surrogate = pd.read_csv(f'{DIST_DIR}/{RELATION}/surrogate.csv', index_col=0)

    print(RELATION)
    for i_v, VIDEO in enumerate(video_emotions):
        copresence_v = noNan(copresence.iloc[i_v,:])
        mean_copr, sd_copr = np.mean(copresence_v), np.std(copresence_v)
        
        stimulus_v = stimulus.iloc[i_v,:]
        mean_stim, sd_stim = np.mean(stimulus_v), np.std(stimulus_v)
        
        surrogate_v = surrogate.iloc[i_v,:]
        mean_surr, sd_surr = np.mean(surrogate_v), np.std(surrogate_v)
       
        sig_surr_stim, p_surr_stim, _, _ = test(surrogate_v, stimulus_v)
        sig_stim_copr, p_stim_copr, _, _ = test(stimulus_v, copresence_v)
        
#        print('{:5s},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(VIDEO, mean_surr, sd_surr, p_surr_stim, mean_stim, sd_stim, p_stim_copr, mean_copr, sd_copr))
        print('{:5s},{},{}'.format(VIDEO, sig_surr_stim, sig_stim_copr))