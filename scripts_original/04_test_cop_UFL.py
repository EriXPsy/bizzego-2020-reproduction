from config import PATH

import pandas as pd
import scipy.stats as stat

def isNormal(x, alpha=0.05):
    W,p = stat.shapiro(x)
    return(p>0.05)

#%%
video_names    = ['HS',    'TIT', 'WD',   'RELAX', 'NH',   'PEN']
video_emotions = ['EMBRS', 'SAD', 'FEAR', 'CALMNESS', 'LOVE', 'PRIDE']

DIST = 'cc_IBI_complete'

#%%
copresence_groups = []
for i_r, RELATION in enumerate(['U', 'F', 'L']):
    #%
    DIST_DIR = f'{PATH}/data/synchrony/{DIST}/{RELATION}'
    copresence = pd.read_csv(f'{DIST_DIR}/copresence.csv', index_col=0)
    
    copresence_videos = []
    for i_v, VIDEO in enumerate(video_emotions):
        copresence_v = copresence.iloc[i_v,:]
        copresence_videos.append(copresence_v)
    copresence_groups.append(copresence_videos)
    
#%%
for V in range(6):
    rUF,pUF = stat.kruskal(copresence_groups[0][V], copresence_groups[1][V])
    rFL,pFL = stat.kruskal(copresence_groups[1][V], copresence_groups[2][V])
    rUL,pUL = stat.kruskal(copresence_groups[0][V], copresence_groups[2][V])
    
    print(f'U-F rho: {rUF} - p:  {pUF}')
    print(f'F-L rho: {rFL} - p:  {pFL}')
    print(f'U-L rho: {rUL} - p:  {pUL}')