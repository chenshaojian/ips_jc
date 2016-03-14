from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
import json
from ips_core.mysql_raw import *
from ips_core.models import *
from django.views.decorators.csrf import csrf_exempt
# Create your views here.


#knn = TrainingModel()
#knn.sample_training_set(0.98,3)
#knn.knn_model()
@csrf_exempt
def ips_knn(req):
    knn = TrainingModel()
    knn.load_data()
    knn.sample_training_set()
    knn.knn_model()
    knn.predict()
    knn.dis_erro()
    knn.error_plot()
    #knn.plot()
    #data={'a':(knn.pred_y).tolist(),'b':(knn.test_y).tolist()}
    #data=a.get_createtime()
    #data=ReqRawData.bssid_list()
    #data='123'
    data=(knn.dis_erro).tolist()
    return HttpResponse(json.dumps(data))
