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
    knn.erro()
    #a=TrainingData()
    #a.load_data()
    #data=a.data
    #a=MeasRawV1()
    #data=a.load_all()
    #data=a.get_createtime()
    print type(knn.data[1][-1])
    #data='123'
    return HttpResponse(knn.ave)
