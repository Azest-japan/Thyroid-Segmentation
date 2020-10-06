# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:37:53 2020

@author: 81807
"""

import numpy as np
import cv2
import pandas as pd
import os
import sys
import pymongo
import time
if 'c:\\users\\81807\\documents\\RD\\yolo-frcnn' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\yolo-frcnn')
    
if 'c:\\users\\81807\\documents\\RD\\deep_sort' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\deep_sort')
    
from yolo_3 import load,givebox 
import imutils
from datetime import datetime
from analysis import reshapeimg


def dbconnect(url = "mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false",dbname=None,delete=False):
    client = pymongo.MongoClient(url)
    #print(client.list_database_names())
    if dbname != None:
        db = client[dbname]
        collnames = db.list_collection_names()
        if delete==True:
            for i in collnames:
                db.drop_collection(i)
        db = client[dbname]  
    return db


basepath = 'C:\\Users\\81807\\Documents\\RD\\realdata\\'

# 1)
def savevid(link="rtsp://admin:0000@192.168.99.3:554/trackID=1",basepath='C:\\Users\\81807\\Documents\\RD\\realdata\\',url = "mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false"): 
    
    db = dbconnect(dbname='videos',delete=True)
    loccol = db['loc']
    frame_index = 0
    vindex = 1
    cap = cv2.VideoCapture(link)
    cap.set(3,960)
    cap.set(4,720)

    try:
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame = reshapeimg(frame)
            if frame_index == 0:
                fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
                out = cv2.VideoWriter(basepath+str(vindex)+'.mp4',  fourcc, 30, frame.shape[::-1][1:])
                vid = {'name':str(vindex),'date':str(datetime.now()),'path': str(vindex)+'.mp4'
                    }
                vindex+=1
            
            if frame_index<1500:  # saves 50 second video in 10 seconds
                frame_index += 1
                if frame_index%5==0:
                    out.write(frame)
            
            else:
                loccol.insert_one(vid)
                frame_index = 0
                out.release()

    except KeyboardInterrupt:
        cap.release()
        out.release()

# 2)
def savebox(): 
    db = dbconnect(dbname='videos')
    loccol = db['loc']
    csvcol = db['csv']
    
    count = 0
    waitc = 0
    
    path,net = load()
    time.sleep(9)
    
    try:
        while True:
            if loccol.estimated_document_count() > count:
                count += 1 
                vs = cv2.VideoCapture(basepath+str(count)+'.mp4')
                df = pd.DataFrame()
                frame_index = 0
                while(True):
                    grabbed,frame = vs.read()
                    if grabbed != True:
                        break
                    df = df.append(pd.DataFrame(givebox(path,net,frame,frame_index,doplot = False,docv = False)),ignore_index=True)
                        
                    frame_index += 1
                df.to_csv(basepath+str(count)+'.csv',header=None,index=False)
                csvcol.insert_one({'name':str(count)+'.csv','date':str(datetime.now())})
            
            elif waitc==0:
                waitc += 1
                time.sleep(10)
                
            else:
                break
    
    except KeyboardInterrupt:
        pass

# 3)
def outvid():
    count = 0
    waitc = 0
    db = dbconnect(dbname='videos')
    csvcol = db['csv']
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    out = cv2.VideoWriter(basepath+'out+id.mp4',  fourcc, 30, (1000,600))
    #time.sleep(100)
    try:
        while True:
            if csvcol.estimated_document_count()>count:
                count += 1
                try:
                    df = pd.read_csv(basepath+str(count)+'+id.csv',delimiter=',',names=['f','id','x','y','w','h'])
                    vs = cv2.VideoCapture(basepath+str(count)+'.mp4')
                except FileNotFoundError:
                    continue
                
                try:
                    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                        else cv2.CAP_PROP_FRAME_COUNT
                    total = int(vs.get(prop))
                    print("[INFO] {} total frames in video".format(total))
                except:
                    print("[INFO] could not determine # of frames in video")
                    continue
                
                frame_index = 0
                while frame_index<total:
                    
                    grabbed,frame = vs.read()
                    if grabbed != True:
                        continue
                    
                    if frame_index == 0:
                        fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
                        out = cv2.VideoWriter(basepath+'out+id.mp4',  fourcc, 30, frame.shape[::-1][1:])
                    
                    df2 = df[df['f']==frame_index]
                    for _,row in df2.iterrows():
                        _,pid,tlx,tly,brx,bry = row
                        pid,tlx,tly,brx,bry = int(pid),int(tlx),int(tly),int(brx),int(bry)
                        
                        color = (120,60,180)
                        #print(frame.shape,x,y,w,h,color)
                        cv2.rectangle(frame, (tlx, tly), (brx, bry), color, 1)
                        
                        cv2.putText(frame, str(pid), (int(tlx), int(tly) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (60,180,90), 2)
                    frame_index += 1
                    #out.write(frame)
                    out.write(cv2.resize(frame,(1000,600),interpolation = cv2.INTER_LINEAR))
            
            elif waitc==0:
                waitc += 1
                time.sleep(100)
                
            else:
                out.release()
                break
    
    except KeyboardInterrupt:
        vs.release()
        out.release()
        cv2.destroyAllWindows()





