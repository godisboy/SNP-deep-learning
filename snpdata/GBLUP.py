import math
import numpy as np
import pandas as pd
import csv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr
import scipy.io as sio  

# Load the diabetes dataset
#diabetes = datasets.load_diabetes()

mageno = pd.read_csv('Magenodata.csv') 
mapheno = pd.read_csv('Maphedata.csv')
np.save('genodata.npy',mageno)
np.save('phenodata.npy',mapheno)
geno = np.load('genodata.npy')
pheno = np.load('phenodata.npy')

X_tr = geno[:1000,1:]   #slicing geno 
#X_va = geno[201:250,:]
X_te = geno[1001:,1:]
Y_tr = pheno[:1000,1:]   #slicing pheno
#Y_va = pheno[201:250,:]
Y_te = pheno[1001:,1:]

diabetes_X_train = X_tr
diabetes_X_test = X_te
diabetes_y_train = Y_tr
diabetes_y_test = Y_te

reg = MLPRegressor(hidden_layer_sizes=(1, ),algorithm='l-bfgs')
reg.fit(X_tr,Y_tr)

scores = cross_val_score(reg,geno[:,1:],pheno[:,1:],cv=10)

#Result_Y = np.zeros((249,1), dtype='float64')
Result_Y = reg.predict(X_te)
#Yte = np.array(Y_te, dtype=np.float64) 
r_row,p_score = pearsonr(Result_Y,Y_te)

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((reg.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(diabetes_X_test, diabetes_y_test))
print(Result_Y)
print(scores)
print(Result_Y.shape)
print(r_row)
print(p_score)

