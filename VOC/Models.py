import numpy as np
import math
from scipy.special import expit
import collections

def sigmoid(x):
    return 1 / (1+np.exp(-x))
def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim ==1:
        return e / np.sum(e,axis=0)
    else:
        return e / np.array([np.sum(e,axis=1)]).T

class LogesticReg(object):

    def __init__(self,land=1e-4,epochs=100):
        self.land = land
        self.Epoch = epochs
        self.Thetas =[]
        self.Grads =[]

    def fit(self,x,y):
        n_sample , n_feature = x.shape
        for i in np.unique(y):
            theta = np.zeros(n_feature)
            gradint = np.ones(n_feature)
            Y = np.zeros(n_sample)
            for index , j in enumerate(y):
                if j == i:
                    Y[index] = 1
            print('class',i)
            ep = 1
            while ep <= self.Epoch:
                z = np.dot(theta,x.T)
                h = sigmoid(z)
                gradint = (self.land * theta) - np.dot(x.T,(Y-h))
                r = np.dot(sigmoid(z),1-sigmoid(z))
                H = (np.eye(theta.shape[0]) * self.land) + np.dot(np.dot(x.T,r),x)
                R = np.dot(np.linalg.inv(H), gradint)
                theta = theta -R
                ep = ep + 1
            self.Thetas.append(theta)
            self.Grads.append(gradint)

    def predict(self,x):
        results = []
        for t in x:
            clsses =[]
            for i in self.Thetas:
                z = np.dot(t,i)
                h = sigmoid(z)
                h = (h >0.5)
                clsses.append(h)
            result = np.argmax(clsses) + 1
            results.append(result)
        return results

class W_LogesticReg(object):
    def __init__(self,land=1e-4,epochs=250,tau=1):
        self.land = land
        self.Epoch = epochs
        self.Tau = tau
        self.Thetas =[]
        self.Grads =[]

    def W(self,x,t):
        n_sample, n_feature = x.shape
        w =np.zeros(n_sample)
        print(w.shape)
        print(n_feature)
        for i in range(n_sample):
            point = x[i]
            for j in range(n_sample):
                diff =1
                for k in range(n_feature):
                    diff = diff * (point[k] - x[j][k])
                w[i] = np.exp((diff**2)/(-2.0*(t**2)))
        w = w * np.eye(n_sample)
        return w

    def fit(self,x,y):
        n_sample , n_feature = x.shape
        for i in np.unique(y):
            theta = np.zeros(n_feature)
            gradint = np.ones(n_feature)
            Y = np.zeros(n_sample)
            for index , j in enumerate(y):
                if j == i:
                    Y[index] = 1
            ep =1
            while ep <= self.Epoch:
                print(ep)
                z = np.dot(theta,x.T)
                h = sigmoid(z)
                gradint = (self.land * theta) - np.dot(np.dot(x.T,self.W(x,self.Tau)),(Y-h))
                print(gradint)
                r = np.dot(sigmoid(z),1-sigmoid(z))
                H = (np.eye(theta.shape[0]) * self.land) + np.dot(np.dot(np.dot(x.T,self.W(x,self.Tau)),r),x)
                R = np.dot(np.linalg.inv(H), gradint)
                theta = theta - R
                ep = ep+1
            self.Thetas.append(theta)
            self.Grads.append(gradint)

    def predict(self,x):
        results = []
        for t in x:
            clsses =[]
            for i in self.Thetas:
                z = np.dot(t,i)
                h = sigmoid(z)
                h = (h >0.5)
                clsses.append(h)
            result = np.argmax(clsses) + 1
            results.append(result)
        return results

class BayesianLogesticReg(object):
    def __init__(self,epochs=250,mu=0,sd=10):
        self.Epoch = epochs
        self.mu0 = mu
        self.S0 = sd
        self.Thetas =[]
        self.Sn =[]
        self.Grads =[]
    def fit(self,x,y):
        n_sample, n_feature = x.shape
        for i in np.unique(y):
            theta = np.zeros(n_feature)
            gradint = np.ones(n_feature)
            H = np.zeros(n_feature)
            Y = np.zeros(n_sample)
            for index, j in enumerate(y):
                if j == i:
                    Y[index] = 1
            ep = 1
            while ep <= self.Epoch:
                z = np.dot(theta, x.T)
                h = expit(z)
                t = np.dot(np.linalg.inv(np.eye(theta.shape[0]) * self.S0),(theta-self.mu0))
                gradint = t - np.dot(x.T, (Y - h))
                r = np.dot(expit(z), 1 - expit(z))
                H = (np.linalg.inv(np.eye(theta.shape[0]) * self.S0)) + np.dot(np.dot(x.T, r), x)
                R = np.dot(np.linalg.inv(H), gradint)
                theta = theta - R
                ep = ep +1
            self.Sn.append(np.linalg.inv(H))
            self.Thetas.append(theta)
            self.Grads.append(gradint)

    def predict(self,x):
        results = []
        for t in x:
            clsses = []
            for i,j in zip(self.Thetas,self.Sn):
                mua = np.dot(t,i)
                simga2 = np.dot(np.dot(t.T, j), t)
                k = ((1+math.pi*simga2)/8)**(0.5)
                h = expit(k*mua)
                h = (h > 0.5)
                clsses.append(h)
            result = np.argmax(clsses) + 1
            results.append(result)
        return results

class Generative_NB(object):
    def __init__(self):
        self.Classes = 0
        self.pres = []
        self.mus = []
        self.sigmas = []
        self.Like =[]
    def pre(self,x):
        n = x.shape[0]
        x = collections.Counter(x)
        for i in x:
            self.pres.append(x[i]/n)
    def mean(self,x,y):
        t = []
        for cl in np.unique(y):
            ti =[]
            for index,i in enumerate(x):
                if y[index] == cl:
                    ti.append(np.array(i))
            t.append(np.array(ti))
        for i in t:
            self.mus.append(np.array(i.mean(axis=0)))
    def var(self,x,y):
        t = []
        for cl in np.unique(y):
            ti = []
            for index, i in enumerate(x):
                if y[index] == cl:
                    ti.append(np.array(i))
            t.append(np.array(ti))
        for i in t:
            self.sigmas.append(np.array(i.var(axis=0)))
    def fit(self,x,y):
        self.Classes = np.unique(y)
        self.mean(x,y)
        self.var(x,y)
        self.pre(y)
    def likelihood(self,x):
        t =[]
        for cl in self.Classes:
            prod = 1
            for index,i in enumerate(x):
                prod = prod * (1/math.sqrt(2*math.pi*self.sigmas[cl - 1][index])) * math.exp(((-0.5) *(i - self.mus[cl - 1][index])**2)/self.sigmas[cl -1][index])
            t.append(prod)
        self.Like.append(t)

    def predict(self,x):
        results =[]
        for t in x:
            self.likelihood(t)
        for like in self.Like:
            res = []
            # total= 0
            # for index,i in enumerate(like):
            #     total = total + np.prod(i*self.pres[index])
            for index,i in enumerate(like):
                res.append((np.prod(i*self.pres[index])))
            results.append(np.argmax(res) + 1)
        return results

class LDA(object):
    def __init__(self):
        self.Classes = 0
        self.mean = []
        self.Scatter =[]
        self.B = None
        self.S = None
        self.eig_Val = None
        self.eig_Vec = None
        self.W = None
        self.transform =None
        self.Leads = []

    def fit(self,x,y):
        self.Classes = np.unique(y)
        for cl in self.Classes:
            self.mean.append(np.mean(x[y == cl],axis=0))
        data_mean = np.mean(x,axis=0).reshape((1,x.shape[1]))
        self.B = np.zeros((x.shape[1],x.shape[1]))
        for cl,mean in enumerate(self.mean):
            n = x[y == cl].shape[0]
            mean = mean.reshape((1,x.shape[1]))
            temp = mean - data_mean
            self.B += n * np.dot(temp.T,temp)
        for cl,mean in enumerate(self.mean):
            si = np.zeros((x.shape[1],x.shape[1]))
            for row in x[y == cl]:
                t = (row - mean).reshape((1,x.shape[1]))
                si += np.dot(t.T,t)
            self.Scatter.append(si)
        self.S = np.zeros((x.shape[1],x.shape[1]))
        for si in self.Scatter:
            self.S += si
        S_inv = np.linalg.inv(self.S)
        S_inv_B = S_inv.dot(self.B)
        self.eig_Val , self.eig_Vec = np.linalg.eig(S_inv_B)
        idx = self.eig_Val.argsort()[::-1]
        self.eig_Val = self.eig_Val[idx]
        self.eig_Vec = self.eig_Vec[:,idx]
        self.W = self.eig_Vec[:,:2]
        self.transform = x.dot(self.W)
        for i in self.Classes:
            self.Leads.append(np.mean(self.transform[y == i],axis=0))


    def predict(self,x):
        results =[]
        for t in x:
            h = t.dot(self.W)
            temp =[]
            for lead in self.Leads:
                temp.append(np.linalg.norm(h-lead))
            results.append(np.argmin(temp) +1 )
        return results








