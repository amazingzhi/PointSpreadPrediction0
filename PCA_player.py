import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#upload dataset
df = pd.read_csv('../data/processing/non_time.csv')
rest = df.iloc[:,:35]
oppo_avg = df.loc[:,'avg_oppo_82_MIN':'avg_oppo_82_PLUS_MINUS']
lag1 = df.loc[:,'lag1_MIN':'lag1_season_Reg']
lag2 = df.loc[:,'lag2_MIN':'lag2_season_Reg']
lag3 = df.loc[:,'lag3_MIN':'lag3_season_Reg']
lag4 = df.loc[:,'lag4_MIN':'lag4_season_Reg']
lag5 = df.loc[:,'lag5_MIN':'lag5_season_Reg']
lag6 = df.loc[:,'lag6_MIN':'lag6_season_Reg']
lag7 = df.loc[:,'lag7_MIN':'lag7_season_Reg']
lag8 = df.loc[:,'lag8_MIN':'lag8_season_Reg']
lag9 = df.loc[:,'lag9_MIN':'lag9_season_Reg']
lag10 = df.loc[:,'lag10_MIN':'lag10_season_Reg']
lag11 = df.loc[:,'lag11_MIN':'lag11_season_Reg']
lag12 = df.loc[:,'lag12_MIN':'lag12_season_Reg']
lag13 = df.loc[:,'lag13_MIN':'lag13_season_Reg']
data = [oppo_avg,lag1,lag2,lag3,lag4,lag5,lag6,lag7,lag8,lag9,lag10,lag11,lag12,lag13]
sc = StandardScaler()

# for i in data:
#     sc.fit(i)
#     std = sc.transform(i)
#     pca = PCA().fit(std)
#     plt.plot(np.cumsum(pca.explained_variance_ratio_), label = f'{i.columns[0]}')
#     plt.legend()
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
final_df = []

columns = []
for j in range(1,6):
    columns.append(f'oppo_avg_{j}')
std = sc.fit_transform(oppo_avg)
pca = PCA(n_components=5)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([rest,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_one_{j}')
std = sc.fit_transform(lag1)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_two_{j}')
std = sc.fit_transform(lag2)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_three_{j}')
std = sc.fit_transform(lag3)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_four_{j}')
std = sc.fit_transform(lag4)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_five_{j}')
std = sc.fit_transform(lag5)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_six_{j}')
std = sc.fit_transform(lag6)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_seven_{j}')
std = sc.fit_transform(lag7)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_eight_{j}')
std = sc.fit_transform(lag8)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_nine_{j}')
std = sc.fit_transform(lag9)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_ten_{j}')
std = sc.fit_transform(lag10)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_eleven_{j}')
std = sc.fit_transform(lag11)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_twelve_{j}')
std = sc.fit_transform(lag12)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'lag_thirteen_{j}')
std = sc.fit_transform(lag13)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

resultCSVPath = '../data/processing/non_time_pca.csv'
final_df.to_csv(resultCSVPath,index = False,na_rep = 0)