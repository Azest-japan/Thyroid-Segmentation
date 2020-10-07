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
import ftplib
from ftplib import FTP
import shutil
import urllib.request as request
from dateutil import parser
from contextlib import closing
import imutils

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

def ftpconnect(day='RecordFolder20201005',link='192.168.99.101',baseloc = '/ftp_share/VideoVolume1/'):
    
    fullp = baseloc+day+'/'
    def savedb(link):
        with closing(request.urlopen('ftp://admin:admin@'+link+fullp+'nvr.db')) as r:
            with open('C:\\Users\\81807\\Documents\\RD\realdata\\a.db', 'wb') as f:
                shutil.copyfileobj(r, f)
    
    vdb = dbconnect(dbname='videos')
    loccol = vdb[day]
    waitc = 0
    
    try:
        ftp = FTP('192.168.99.101','admin','admin')
        ftp.cwd(fullp)
        vlist = ftp.nlst()
        vtlist = [(i,parser.parse(ftp.voidcmd("MDTM "+fullp+i)[4:].strip())) for i in vlist if i[0] == '2']
        vtlist.sort(key=lambda x:x[1])
        tmax = vtlist[-1][1]
        for vname, t in vtlist:
            if loccol.find_one({'name':vname}) == None:
                loccol.insert_one({'name':vname,'time':t,'status':0})
            
        while True:
            
            ftp = FTP('192.168.99.101','admin','admin')
            ftp.cwd(fullp)
            vlist = ftp.nlst()
            vtlist = [(i,parser.parse(ftp.voidcmd("MDTM "+fullp+i)[4:].strip())) for i in vlist if i[0] == '2']
            vtmax = max(vtlist,key=lambda x:x[1])
            
            if vtmax[1] > tmax:
                tmax = vtmax[1]
                loccol.insert_one({'name':vtmax[0],'time':vtmax[1]})
            
            elif waitc <2:
                waitc += 1
                print('waiting ',waitc)
                time.sleep(3)
            else:
                break

    except ftplib.all_errors:
        return 


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
def savebox(day='RecordFolder20201005',link='192.168.99.101',baseloc='/ftp_share/VideoVolume1/'): 
    path,net = load()
    basepath = 'C:\\Users\\81807\\Documents\\RD\\realdata\\'
    fullp = baseloc+day+'/'
    boxdb = dbconnect(dbname='boxcsv')
    boxcol = boxdb[day]
    
    vdb = dbconnect(dbname='videos')
    loccol = vdb[day]
    
    if day not in [i for i in os.listdir(basepath) if os.path.isdir(basepath+i)]:
        os.mkdir(basepath+day)
    
    waitc = 0
    frame_index = 0
    start = 0
    
    boxlist = [i for i in boxcol.find({'status':1})]
    if len(boxlist)!=0:
        boxmax = max(boxlist,key=lambda x:x['date'])
        start = boxmax['start']
    
    try:
        while True:
            vinfo = loccol.find_one({'status':0})
            if vinfo != None:
                vs = cv2.VideoCapture('ftp://admin:admin@'+link+fullp+vinfo['name'])
                df = pd.DataFrame()
                frame_index = 0
                
                try:
                    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                        else cv2.CAP_PROP_FRAME_COUNT
                    total = int(vs.get(prop))
                    print("[INFO] {} total frames in video".format(total),vinfo['name'])
                except:
                    print("[INFO] could not determine # of frames in video")
                    total = -1
                    
                while frame_index+start < total:
                    
                    grabbed,frame = vs.read()
                    frame = reshapeimg(frame)
                    
                    if (frame_index+start)%5 != 0:
                        frame_index += 1
                        continue
                    
                    if grabbed != True:
                        break
                    
                    df = df.append(pd.DataFrame(givebox(path,net,frame,frame_index,doplot = False,docv = False)),ignore_index=True)
                        
                    frame_index += 1
                    
                start = (frame_index+start)%5
                
                df.to_csv('C:\\Users\\81807\\Documents\\RD\\realdata\\'+day+'\\'+vinfo['name'][:-4]+'.csv',header=None,index=False)
                boxcol.insert_one({'name':vinfo['name'][:-4]+'.csv','date':str(datetime.now()),'status':0,'start':start})
                loccol.update_one({'_id':vinfo['_id']},{'$set' :{'status':1}})
                
            elif waitc==0:
                waitc += 1
                print('waiting ',waitc)
                time.sleep(3)
                
            else:
                break
    
    except KeyboardInterrupt:
        pass



# 3)
def outvid(basepath='C:\\Users\\81807\\Documents\\RD\\realdata\\',link='192.168.99.101',day='RecordFolder20201005',size=(1000,600)):
    count = 0
    waitc = 0
    deepdb = dbconnect(dbname='deepcsv')
    deepcol = deepdb[day]
    
    deeplist = [i for i in deepcol.find()]
    
    
    fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
    out = cv2.VideoWriter(basepath+day+'\\'+'out+id.mp4',  fourcc, 30, size)
    #time.sleep(100)
    index = 0
    folderp = basepath + day + '\\'
    baseloc = '/ftp_share/VideoVolume1/'
    fullp = baseloc + day + '/'
    frame_index = 0
    try:
        while True:
            deeplist = [i for i in deepcol.find()]
            if len(deeplist)>index:
                print(index,deeplist[index]['name'])
                try:
                    df = pd.read_csv(folderp+deeplist[index]['name'],delimiter=',',names=['f','id','x','y','w','h'])
                    vs = cv2.VideoCapture('ftp://admin:admin@'+link+fullp+deeplist[index]['name'][:-7]+'.avi')
                    
                except FileNotFoundError:
                    waitc += 1
                    time.sleep(3)
                    continue
                
                try:
                    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                        else cv2.CAP_PROP_FRAME_COUNT
                    total = int(vs.get(prop))
                    print("[INFO] {} total frames in video".format(total))
                except:
                    print("[INFO] could not determine # of frames in video")
                    continue
                
                loc_index = 0
                icount = 0
                while icount < len(df):
                    
                    grabbed,frame = vs.read()
                    frame = reshapeimg(frame)
                    if grabbed != True:
                        continue
                    
                    if frame_index == 0:
                        frame_index += 1
                        fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
                        out = cv2.VideoWriter(basepath+day+'\\'+'out+id.mp4',  fourcc, 30, size)
                    
                    if loc_index != df.iloc[icount]['f']:
                        loc_index += 1
                        continue
                    
                    df2 = df[df['f']==loc_index]
                    for _,row in df2.iterrows():
                        _,pid,tlx,tly,brx,bry = row
                        pid,tlx,tly,brx,bry = int(pid),int(tlx),int(tly),int(brx),int(bry)
                        
                        color = (120,60,180)
                        #print(frame.shape,x,y,w,h,color)
                        cv2.rectangle(frame, (tlx, tly), (brx, bry), color, 1)
                        
                        cv2.putText(frame, str(pid), (int(tlx), int(tly) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (60,180,90), 2)

                    #out.write(frame)
                    out.write(cv2.resize(frame,size,interpolation = cv2.INTER_LINEAR))
                    icount += 1
                    loc_index += 1
                index += 1
                    
            elif waitc==0:
                waitc += 1
                print('waiting ',waitc)
                time.sleep(3)
                
            else:
                out.release()
                break
    
    except KeyboardInterrupt:
        vs.release()
        out.release()
        cv2.destroyAllWindows()





