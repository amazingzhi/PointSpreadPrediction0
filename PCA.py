import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#upload dataset
df = pd.read_csv('NewOriginalNBAData.csv')
Y = df.iloc[:, 4]
df = df.iloc[:,5:]
sc = StandardScaler()

# PCA transformation with creating columns names for PCA data.
columns = []
for j in range(1,41):
    columns.append(f'PCA{j}')
std = sc.fit_transform(df)
pca = PCA(n_components=40)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
FinalDF = pd.concat([Y, tempDF], axis=1)

# save PCA data
resultCSVPath = r'data/PCA.csv'
FinalDF.to_csv(resultCSVPath,index = False,na_rep = 0)
