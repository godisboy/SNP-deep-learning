#!/usr/bin/env python


from sklearn import preprocessing
import scipy.io as sio  
import numpy as np  

def scaled(X,X_mean,max_y):
    return X-X_mean/max_y

def writeToTxt(list_name,file_path):
    try:
        fp = open(file_path,"w+")
        for item in list_name:
            fp.write(str(item)+"\n")#list中一项占一行
        fp.close()
    except IOError:
        print("fail to open file")


#matlab文件名   
genomat='/home/if/Downloads/snpdata/genodata.mat'
phenomat='/home/if/Downloads/snpdata/phenodata.mat'

data_1=sio.loadmat(genomat)
data_2=sio.loadmat(phenomat)
#print type(data_1)
#num = len(data_1)
gen = data_1.keys()
phe = data_2.keys()
geno = data_1['genotest']
pheno = data_2['pheno']



###### feacture scaling ######
#caculate the mean of 
pheno_mean = np.mean(pheno)
pheno_max = np.max(pheno)
#pheno_std = preprocessing.scale(pheno)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
pheno_scaled = min_max_scaler.fit_transform(pheno)
#np.savetxt("geno.txt",geno)
#writeToTxt(geno,'/home/if/Downloads/snpdata/geno.txt')

X_tr = geno[:200,:]   #slicing geno 
X_va = geno[201:250,:]
X_te = geno[251:301,:]
Y_tr = pheno_scaled[:200,:]   #slicing pheno 
Y_va = pheno_scaled[201:250,:]
Y_te = pheno_scaled[251:301,:]

np.save('geno_X_tr.npy', X_tr)
np.save('geno_X_va.npy', X_va)
np.save('geno_X_te.npy', X_te)
np.save('pheno_Y_tr.npy', Y_tr)
np.save('pheno_Y_va.npy', Y_va)
np.save('pheno_Y_te.npy', Y_te)

#X_tr = np.load('geno_X_tr.npy')
#Y_tr = np.load('pheno_Y_tr.npy')

#print (num)
#print (data_1)
#print (geno)
#print (data_2)
#print (pheno)
#print (pheno_mean)
#print (pheno_max)
#print (pheno_minmax)
#print (pheno_scaled)
#print(X_tr)
#print(Y_tr)

