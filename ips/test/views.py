#import sys
#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cumulatedaily.settings')
#reload(sys)
from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
import json
from ips_core.meas_raw import *
from ips_core.models import *
from django.views.decorators.csrf import csrf_exempt
# Create your views here.

@csrf_exempt
def ips_result(req):
    rate = req.POST.get('rate')
    rate = float(rate)

    knn = KnnModel()
    knn.load_data()
    knn.sample_training_set(sample_rate = rate)
    knn.train()
    pre_pos = knn.predict(knn.test_x)

    ret = {'predict_pos':pre_pos.tolist(), 'real_pos':knn.test_y.tolist()}

    return HttpResponse(json.dumps(ret), content_type='application/json')
