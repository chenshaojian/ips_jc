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
    #data=a.get_createtime()
    #data=ReqRawData.bssid_list()
    return HttpResponse(json.dumps('213')
