import requests
import json
import time

### This is sample script for debug.
### ex.)
###   python reset_sample.py

JUDGESERVER_REQUEST_URL="http://127.0.0.1:5000/judgeserver/request"
JUDGESERVER_UPDATEDATA_URL="http://127.0.0.1:5000/judgeserver/updateData"
JUDGESERVER_GETSTATE_URL="http://127.0.0.1:5000/judgeserver/getState"

def httpPostReqToURL(url, data):
    res = requests.post(url,
                        json.dumps(data),
                        headers={'Content-Type': 'application/json'}
    )
    return res

# init button
def Init():
    url = JUDGESERVER_REQUEST_URL
    req_data = {"change_state": "init"}
    res = httpPostReqToURL(url, req_data)
    return res

# start button
def Start():
    url = JUDGESERVER_REQUEST_URL
    req_data = {"change_state": "start"}
    res = httpPostReqToURL(url, req_data)
    return res

# stop button
def Stop():
    url = JUDGESERVER_REQUEST_URL
    req_data = {"change_state": "stop"}
    res = httpPostReqToURL(url, req_data)
    return res

# ManualRecovery button
def ManualRecovery():
    url = JUDGESERVER_UPDATEDATA_URL
    req_data = {
        # "courseout_count": 1,
        "is_courseout": 1
    }
    res = httpPostReqToURL(url, req_data)
    return res


if __name__ == '__main__':
    print("Init()")
    Init()
    time.sleep(3)
    
    print("Start()")
    Start()
    time.sleep(3)
    
    print("Stop()")
    Stop()
    time.sleep(3)
    
    print("ManualRecovery()")
    ManualRecovery()
