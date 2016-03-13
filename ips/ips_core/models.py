# -*- coding:utf-8 -*-
from mysql_raw import *
from numpy import *
import time
from collections import defaultdict
import random
import cPickle as pickle
from sklearn import neighbors
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pylab as plt
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(threshold=np.nan)

class AnalysisModel(object):
    def __init__(self):
        self.ave=None
    def erro(self):
        self.ave=(abs(self.pred_y-self.test_y)).sum(0)/self.test_y.shape[0]
        print 'k=',self.k,' ave=',self.ave
    #def predict(self):
    #    self.pred_y = self.clf.predict(self.test_x)

class TrainingData(AnalysisModel):
    def __init__(self):
        self.data=None
        self.training_x = None
        self.training_y = None

        self.test_x = None
        self.test_y = None

        #self.total_x = np.zeros((0,0),dtype=np.float)
        #self.total_y = np.zeros((0,0),dtype=np.float)
        self.test_size=0.02
        self.random_state=3
    def load_data(self):
        #load training data
        #raw_data = MeasRawV1.load_all()
        #raw_time=MeasRawV1.get_createtime()
        tra_bssid=['00:e1:40:20:00:6e','00:e1:40:20:00:d7','40:a5:ef:84:81:8d','40:a5:ef:84:7a:79','01']
        raw_data=ReqRawData.load_all(tra_bssid)
        raw_time=ReqRawData.get_createtime()
        data=zeros((len(raw_time),(len(tra_bssid)+1)))#[四个bssid ,x,y]
        for i in raw_data:
            data[raw_time.index(i['createtime'])][-2],data[raw_time.index(i['createtime'])][-1]=int(float(i["x"])*250),int(float(i["y"])*120)
            if(data[raw_time.index(i['createtime'])][tra_bssid.index(i['bssid'])]!=0.0
                and int(data[raw_time.index(i['createtime'])][tra_bssid.index(i['bssid'])])>int(i['rssi'])):
                continue
            else:
                data[raw_time.index(i['createtime'])][tra_bssid.index(i['bssid'])]=int(i['rssi'])
        #删除[0,0,0,0,0,0]
        data1=list(data)
        j=0
        for i in data:
            if((i==[0]*data.shape[1]).all()):
                del data1[j]
            else:
                j+=1
        self.data=array(data1)
    def sample_training_set(self, test_size=0.2,random_state=3):
        self.training_x, self.test_x, self.training_y, self.test_y = \
            cross_validation.train_test_split(self.data[:,0:-2], self.data[:,-2:], test_size = test_size, random_state = random_state)
        #print self.test_x
class TrainingModel(TrainingData,AnalysisModel):
    def __init__(self):
            TrainingData.__init__(self)#super(TrainingModel, self).__init__()
            self.k = 3
            self.clf=None
            self.pred_y=None
    def knn_model(self):
        start_time = time.time()
        self.clf = KNeighborsClassifier(n_neighbors =self.k)
        self.clf.fit(self.training_x,self.training_y)
        print("time spent:", time.time() - start_time)
    def predict(self):
        self.pred_y = self.clf.predict(self.test_x)

