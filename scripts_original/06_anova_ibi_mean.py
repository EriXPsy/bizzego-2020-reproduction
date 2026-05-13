from config import PATH

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
data = pd.read_csv(f'{PATH}/data/mean_ibi.csv', index_col=0)

videos = ['HS', 'TIT', 'WD', 'NEUT', 'NH', 'PEN']

for VIDEO in videos:
    data_video = data.query('video == @VIDEO')
    model = ols('ibi_mean ~ C(gender) + C(group) + C(group):C(gender)', data=data_video).fit()

    table = sm.stats.anova_lm(model, typ=2)
    table = omega_squared(table)
    print('==============')
    print(VIDEO)
    print(table)
    print('==============')
    
#%%
    
model = ols('ibi_mean ~ C(group) + C(gender) + C(group):C(gender)', data=data).fit()
table = sm.stats.anova_lm(model, typ=2)

res = model.resid
fig = sm.qqplot(res, line='s')