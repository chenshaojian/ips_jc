__author__ = 'chen'
import MySQLdb as mdb
import MySQLdb.cursors
import datetime
import numpy as np

conn = mdb.connect('localhost', 'root', 'zkkj20141101db', 'gb_ips')
MIN_RSSI = -100

class MysqlAccesser(object):
    @classmethod
    def load(cls,cols=None,where=None,table_name=None,offset=None,limit=None,order_by=None):
        cur=conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
        sql_str='select '
        if(cols is not None):
            sql_str +=cols+' '
        else:
            sql_str +='* '
        sql_str +='from '+table_name+' '
        if(where is not None):
            sql_str +='where '+where+' '
        if order_by is not None:
            sql_str +='order by '+order_by+' '
        if limit is not None:
            sql_str +='limit '+limit

        cur.execute(sql_str)

        return cur.fetchall()
class MeasRawV1(MysqlAccesser):
    @classmethod
    def load_all(cls):
        rows = cls.load(cols='x,y,ssid,bssid,device_id,rssi,createtime',\
                        where="ssid like 'GuangBai%' and market_id='005' and floor_id='01' and\
                        bssid='00:e1:40:20:00:6e' or bssid='00:e1:40:20:00:d7' or bssid='40:a5:ef:84:81:8d' or bssid='40:a5:ef:84:7a:79'",\
                        table_name='training_raw_data',limit='1000')
        return rows
    @classmethod
    def get_createtime(cls):
        createtime = cls.load(cols='createtime',where="ssid like 'GuangBai%' and market_id='005' and floor_id='01'",table_name='training_raw_data')
        createtime1=[]
        for i in createtime:
           createtime1.append(i['createtime'])
        createtime2=list(set(createtime1))
        createtime2.sort(key=createtime1.index)
        return createtime1


class ReqRawData():
    def bssid(request):
        ###require brssi
        bssid={'bssid':json.dumps('True')}#data must be json
        bssid=datapost1(bssid)
        a=datapost1({'ssid':json.dumps('True')})
        print a
        return HttpResponse(json.dumps({'bssid':a}))


