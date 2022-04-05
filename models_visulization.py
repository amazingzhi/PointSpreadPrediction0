import joblib
import matplotlib.pyplot as plt
from sklearn import tree

linear = joblib.load('models/PCA_mean_absolute_error_LR_model.joblib')
DT = joblib.load('models/feature_selectionneg_mean_absolute_error_DTR_StandardScaler_model.joblib')
DT = DT.best_estimator_
tree.plot_tree(DT)
print('a')