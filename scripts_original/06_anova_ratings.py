from config import PATH

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
 
def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

#%%
datafile = f'{PATH}/data/emotion_ratings_embedding.csv'
data = pd.read_csv(datafile, index_col=0)

videos = np.unique(data['stimulus'])

for VIDEO in videos:
    data_video = data.query('stimulus == @VIDEO')
    model = ols('embedding1 ~ C(subject) + C(relation) + C(relation):C(subject)', data=data_video).fit()

    table = sm.stats.anova_lm(model, typ=2)
    table = omega_squared(table)
    print('==============')
    print(VIDEO)
    print(table)
#    print(model.summary())
    print('==============') 

#%%

model = ols('embedding1 ~ C(stimulus) + C(relation) + C(subject) + C(stimulus):C(relation) + C(relation):C(subject) + C(stimulus):C(subject)', data=data).fit()
#model = ols('embedding1 ~ C(video)', data=data).fit()

table = sm.stats.anova_lm(model, typ=2)
table = omega_squared(table)
print('==============')
print(table)
#print(model.summary())
print('==============') 
