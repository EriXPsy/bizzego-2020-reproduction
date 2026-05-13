from config import PATH

import pandas as pd
from sklearn.decomposition import PCA as PCA

#%%
datafile = f'{PATH}/data/emotion_ratings.csv'
data = pd.read_csv(datafile, index_col=0)

#%%
pca = PCA(n_components=2)

embedding2  = pca.fit_transform((data.iloc[:, 4:].values - 1)/6)

data['embedding2_1'] = embedding2[:,0]
data['embedding2_2'] = embedding2[:,1]

#%%
pca = PCA(n_components=1)

embedding1  = pca.fit_transform((data.iloc[:, 4:].values - 1)/6)

data['embedding1'] = embedding1[:,0]

data.to_csv(f'{PATH}/data/emotion_ratings_embedding.csv')
