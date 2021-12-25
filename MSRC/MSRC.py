import scipy.io
import numpy as np
import Models
import pandas as pd
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import collections

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

mat = scipy.io.loadmat('MSRC.mat')
X = mat['fts']
Y = mat['labels'].reshape(1269)
std = StandardScaler()
X = std.fit_transform(X)

result_table = pd.DataFrame(columns=['round','#train_sample','#test_sample','classifier','Epoches','landa','Correct','Precision','Recall','F-1','ACC'])
c = 0
for land in [10,1,0,0.1,0.01]:
    for ep in [500,1000,1500]:
        kf = KFold(n_splits=5, shuffle=True)
        for train_index, test_index in kf.split(X):
            train_x , test_x = X[train_index] ,X[test_index]
            train_y , test_y = Y[train_index] ,Y[test_index]
            c+=1
            print(c)
            clf = Models.Generative_NB()
            clf.fit(train_x,train_y)
            predict = clf.predict(test_x)
            # prob = clf.predict_proba(test_x)
            row = {
                'round' :c,
                '#train_sample' : train_x.shape[0],
                '#test_sample': test_x.shape[0],
                'classifier': 'LogesticReg',
                'Epoches' : ep,
                'Landa' : land,
                'Correct': np.sum(predict == test_y),
                'Precision': round(metrics.precision_score(test_y, predict,average='macro'), 4),
                'Recall': round(metrics.recall_score(test_y, predict,average='macro'), 4),
                'F-1': round(metrics.f1_score(test_y, predict,average='macro'), 4),
                'ACC' : round(metrics.accuracy_score(test_y,predict),4),
                # 'ROC' : round(metrics.roc_auc_score(test_y, prob,average='macro',multi_class='ovo'),4)
            }
            print(row)
            result_table = result_table.append(row,ignore_index=True)
result_table.to_excel('Result.xlsx',sheet_name='LogesticReg')
# with pd.ExcelWriter('Result.xlsx',mode='a') as wr:
#     result_table.to_excel(wr,sheet_name='BayesianLogesticReg')
