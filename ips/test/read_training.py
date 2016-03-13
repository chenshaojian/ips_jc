#! /usr/bin/python
# -*- coding:utf-8 -*-

import MySQLdb
import json
import requests

conn = MySQLdb.connect('9320','120.25.86.215', 'root', 'zkkj20141101db', 'gb_ips')
cur = conn.cursor()
sql = 'select * from training_raw_data where floor_id="01"'
cur.execute(sql)
result = cur.fetchall()
print result

