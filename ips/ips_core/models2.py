from meas_raw import *
import numpy as np
from collections import defaultdict
import random
import cPickle as pickle
from sklearn import neighbors
import matplotlib
matplotlib.use('Cairo')
import matplotlib.pylab as plt
np.set_printoptions(threshold=np.nan)

class TrainingModel(object):

    def __init__(self):
        self.estimator = None
        self.training_x = np.zeros((0,0),dtype=np.float)
        self.training_y = np.zeros((0,0),dtype=np.float)

        self.test_x = np.zeros((0,0),dtype=np.float)
        self.test_y = np.zeros((0,0),dtype=np.float)

        self.total_x = np.zeros((0,0),dtype=np.float)
        self.total_y = np.zeros((0,0),dtype=np.float)


    def train(self):
        pass

    def get_position(object):
        pass


    @property
    def rows(self):
        return self.x.shape(0)

    def load_data(self):
        #load training data

        raw_data = MeasRawV1.load_all()
        map_key_data = defaultdict(list)

        for data in raw_data:
            map_key_data[data.key].append(data)


        #calcuate the
        APMacSet.init()

        # get numer of cols
        cols = len(APMacSet.map_mac_idx)

        # get number of rows
        rows = len(map_key_data)

        self.total_x = np.zeros((rows, cols), dtype=np.float32)
        self.total_y = np.zeros((rows, 2), dtype=np.float32)

        self.total_x -= 255
        i=0
        row_key_idx = dict()
        for key in map_key_data.keys():
            row_key_idx[key] = i
            i+=1
        for k,v in map_key_data.iteritems():
            for data in v:
                assert isinstance(data, MeasRawV1)
                row_idx = row_key_idx[k]
                col_idx = APMacSet.get_idx(data.ap_mac)
                self.total_x[row_idx, col_idx] = data.rssi


                self.total_y[row_idx, 0] = data.x
                self.total_y[row_idx, 1] = data.y
        #self.normalize_rssi()
    def normalize_rssi(self):
        pass


    def predict(self, data):
        pass

    def train(self):
        pass

    def errors(self, predicted_y):
        errors = np.abs(predicted_y - self.test_y)
        return errors
    def sample_training_set(self, sample_rate=0.8):
        rows = self.total_y.shape[0]
        idx = range(rows)
        s = int(rows*sample_rate)
        random.shuffle(idx)

        self.training_x = self.total_x[idx[0:s],:]
        self.training_y = self.total_y[idx[0:s],:]

        self.test_x = self.total_x[idx[s:-1],:]
        self.test_y = self.total_y[idx[s:-1],:]

    def save(self, filename):
        pickle.dump(self, open(filename,'w'))

    @classmethod
    def load(cls, filename):
        return pickle.load(open(filename,"r"))

    def plot(self, predicted_y, real_y,floor_plan_file, output_file):
        im = plt.imread(floor_plan_file)
        implot = plt.imshow(im)
        im_rows = im.shape[0]
        im_cols = im.shape[1]
        p = np.copy(predicted_y)
        r = np.copy(real_y)
        p[:,0] *= im_cols
        r[:,0] *= im_cols


        #flip y
        p[:,1] = 1.0- p[:,1]
        r[:,1] = 1.0- r[:,1]

        p[:,1] *= im_rows
        r[:,1] *= im_rows


        scale = 15
        #plt.scatter(p[:,0], p[:,1], c='r',s=scale)
        #plt.scatter(r[:,0], r[:,1], c='b',s=scale)
        rows = predicted_y.shape[0]

        labels = ['{0}'.format(i) for i in range(rows)]

        size=2
        X =[]
        Y = []
        for label, x, y,s,t in zip(labels, p[:, 0], p[:, 1], r[:,0], r[:,1]):
                #if any([x, y, s,t]) is np.nan:
                #print np.isnan([x,y,s,t])
            if any(np.isnan([x,y,s,t])) is True:
                continue
            plt.annotate(label,xy = (x, y), ha = 'center', va = 'center',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),size=size)
            plt.annotate(label,xy = (s, t), ha = 'center', va = 'center',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.5),size=size)
            X.append(x)
            X.append(s)
            X.append(None)
            Y.append(y)
            Y.append(t)
            Y.append(None)


        plt.plot(X,Y)

        plt.savefig(output_file, dpi=600)



    def predict(self, data):
        return self.estimator.predict(data)




class KnnModel(TrainingModel):


    def __init__(self):
        super(TrainingModel, self).__init__()
        self.k = 3
        #self.weights ='distance'
        self.weights ='uniform'

    def train(self):
        self.estimator = neighbors.KNeighborsRegressor(self.k, weights=self.weights)
        self.estimator.fit(self.training_x, self.training_y)


    def normalize_rssi(self):
        pass
        #self.total_x = np.power(self.total_x, 10)
        #self.total_x = np.power(10, self.total_x)
        #for i in range(self.total_x.shape[0]):
        #    self.total_x[i,:] -= np.max(self.total_x[i,])


#knn = KnnModel()
#knn.load_data()
#knn.sample_training_set(sample_rate = 0.99)
##print knn.test_x, knn.test_y
##
#knn.save("knn.dat")


#
#knn.load("knn.dat")


#knn = KnnModel.load("knn.dat")
#knn.sample_training_set(0.98)
#knn.train()
#predicted_y = knn.predict(knn.test_x)
#print predicted_y
#knn.plot(predicted_y, knn.test_y, "./gb_f1.jpg", "gb.png")


#e= knn.errors(predicted_y)

#v = np.hstack((knn.test_x, e))
#e = e[~np.isnan(e).any(1)]
#print v
#print np.sum(e)
#print knn.training_x


