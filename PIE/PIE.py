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

P1= scipy.io.loadmat('P1.mat')
P2= scipy.io.loadmat('P2.mat')
P3= scipy.io.loadmat('P3.mat')
P4= scipy.io.loadmat('P4.mat')
P5= scipy.io.loadmat('P5.mat')
std =StandardScaler()
P1['fts'] = std.fit_transform(P1['fts'])
P2['fts'] = std.fit_transform(P2['fts'])
P3['fts'] = std.fit_transform(P3['fts'])
P4['fts'] = std.fit_transform(P4['fts'])
P5['fts'] = std.fit_transform(P5['fts'])
x_test = P1['fts']
y_test = P1['labels']
x_train = np.vstack((P2['fts'],P3['fts'],P4['fts'],P5['fts']))
y_train = np.vstack((P2['labels'],P3['labels'],P4['labels'],P5['labels'])).reshape(x_train.shape[0])
result_table = pd.DataFrame(columns=['round','#train_sample','#test_sample','Test','classifier','land','epoches','Correct','Precision','Recall','F-1','ACC'])
c = 0
# for lan in [10,1,0,0.1,0.01]:
# for ep in [500,1000,1500]:
# c+=1
clf = Models.Generative_NB()
clf.fit(x_train, y_train)
predict = clf.predict(x_test)
row = {
    'round': c,
    '#train_sample': x_train.shape[0],
    '#test_sample': x_test.shape[0],
    'Test': 'P1',
    'classifier': 'LogesticReg',
    # 'land':lan,
    # 'epoches':ep,
    'Correct': np.sum(predict == y_test),
    'Precision': round(metrics.precision_score(y_test, predict, average='micro', pos_label='positive'), 4),
    'Recall': round(metrics.recall_score(y_test, predict, average='micro', pos_label='positive'), 4),
    'F-1': round(metrics.f1_score(y_test, predict, average='micro', pos_label='positive'), 4),
    'ACC': round(metrics.accuracy_score(y_test, predict), 4)
}
print(row)
result_table = result_table.append(row, ignore_index=True)
result_table.to_excel('Result.xlsx',sheet_name='LogesticReg(P1)')
