# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:18:28 2020

@author: 81807
"""

import numpy as np
import cv2
import os
import sys
import time
from datetime import datetime
import datetime as dtlib
import pymongo
from analysis import reshapeimg
import imutils
import pandas as pd
from yolo_3 import load,givebox 
from demo import process

class Dal:
    
    def __init__(self,dburl=None,delete=False):
        
        self.defaults = {'dburl' : "mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false"}
        
        self.dburl = dburl
        self.videodb = self.mongoconnect('videoDB',delete)
        
        self.isoweekdict = {1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday'}
        
        if dburl==None:
            self.dburl = self.defaults['dburl']
       
    
    def mongoconnect(self,dbname,delete=False):
        client = pymongo.MongoClient(self.dburl)
        
        db = client[dbname]
        collnames = db.list_collection_names()
        if delete==False:
            return db
        
        for i in collnames:
            db.drop_collection(i)
        db = client[dbname]
        return db
    
    
    def delete_db(self,dbname):
        self.mongoconnect(dbname,delete=True)
    
    def insert_day(self,day):
        
        daycol = self.videodb['day']
        actday = day[day.find('2'):]
        year = int(actday[:4])
        month = int(actday[4:6])
        dy = int(actday[6:])
        dt = datetime(year=year,month=month,day=dy)
        
        daydoc = daycol.find_one({'dayname':day})
        
        if daydoc == None:
            daycol.insert_one({'dayname':day,'year':year,'month':month,'day':dy,'weekday':self.isoweekdict[dt.isoweekday()],'vidcount':0})
            return 0
        
        else:
            return daydoc['vidcount']
        
    
    def insert_video(self,vname,day,stime,etime,camname=None,check=True):
    
        vidcol = self.videodb['video']
        daycol = self.videodb['day']
        
        vidcount = self.insert_day(day)
        self.insert_camera(camname)
        
        if check == True:
            if vidcol.find_one({'vidname':vname}) == None:
                vidcol.insert_one({'vidname':vname,'dayname':day,'start_time':stime,'end_time':etime,'camname':camname,'status':0,'total':0})
                daycol.update_one({'dayname':day},{'$set' :{'vidcount':vidcount+1}})
            return 
        
        vidcol.insert_one({'vidname':vname,'dayname':day,'start_time':stime,'end_time':etime,'camname':camname,'status':0,'total':0})
        daycol.update_one({'dayname':day},{'$set' :{'vidcount':vidcount+1}})
        
        
    def delete_video(self,vname,day):
        
        vidcol = self.videodb['video']
        vidcol.delete_one({'vidname':vname})
        
    
    def insert_camera(self,camname,fps=None,model=None,resolution=None,matrix=None,ipop=None,roomname=None,check=True):
        
        if camname == None:
            return
        
        camcol = self.videodb['camera']
        camIP = camname.split('_')[0]
        if check == True:
            if camcol.find_one({'camname':camname}) == None:
                camcol.insert_one({'camname':camname,'camIP':camIP,'fps':fps,'model':model,'resolution':resolution,'matrix':matrix,'ipop':ipop,'roomname':roomname})
            return
        
        camcol.insert_one({'camname':camname,'camIP':camIP,'fps':fps,'model':model,'resolution':resolution,'matrix':matrix,'ipop':ipop,'roomname':roomname})
    
    def delete_camera(self,camname):
       
       camcol = self.videodb['camera']
       vidcol = self.videodb['video']
       camcol.delete_one({'camname':camname})
       vidcol.update_many({'camname':camname}, {'$set':{'camname':None}})
       
        
    def insert_room(self,roomname,buldingname=None,check=True):
        
        if roomname == None:
            return
        
        roomcol = self.videodb['room']
        if check == True:
            if roomcol.find_one({'roomname':roomname}) == None:
                roomcol.insert_one({'roomname':roomname})
            return
        
        roomcol.insert_one({'roomname':roomname})
    
    
    def insert_box(self,ftplink,day=None,vname=None):
        
        vidcol = self.videodb['video']
        boxcol = self.videodb['box_file']
        boxdatacol = self.videodb['box_data']
        
        waitc = 0
        frame_index = 0
        reminder = 0
        start = 0
        vinfo = None
        path, net = load()
        
        restart_frame = 0
        unfinished_box = boxcol.find_one({'enddate':{'$exists':False}})
        if unfinished_box != None:
            try:
                restart_frame = boxdatacol.find_one({'vidname':unfinished_box['vidname']},sort=[('f',pymongo.DESCENDING)])['f']
            except Exception:
                restart_frame = 0
                boxcol.delete_one({'vidname':unfinished_box['vidname']})
                
            vinfo = vidcol.find_one({'status':0,'vidname':unfinished_box['vidname']})
            
        try:
            while True:
                
                if vname == None:
                    vinfo = vidcol.find_one({'status':0,'dayname':day})
                elif vinfo == None:
                    vinfo = vidcol.find_one({'status':0,'vidname':vname})
                
                vprev = vidcol.find_one({'status':1,'camname':vinfo['camname'],'end_time':{'$gte':vinfo['start_time']-dtlib.timedelta(seconds=1),'$lte':vinfo['start_time']}},sort=[('end_time',pymongo.DESCENDING)])
                if vprev != None:
                    reminder = boxcol.find_one({'vidname':vprev['vidname']})['reminder']
                    start = 1

                if vinfo != None:
                    
                    frame_index = 0
                    vs = cv2.VideoCapture(ftplink+vinfo['vidname'])
                    
                    try:
                        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                            else cv2.CAP_PROP_FRAME_COUNT
                        total = int(vs.get(prop))
                        print("[INFO] {} total frames in video".format(total),vinfo['vidname'])
                        if restart_frame == 0:
                            boxcol.insert_one({'vidname':vinfo['vidname'],'startdate':str(datetime.now()),'status':0, 'start':start})
                        
                    except:
                        print("[INFO] could not determine # of frames in video")
                        total = -1
                        
                    while frame_index < total:
                        
                        grabbed,frame = vs.read()
                        frame = reshapeimg(frame)

                        if (frame_index-reminder)%5 != 0 or restart_frame >= frame_index:
                            frame_index += 1
                            continue
                        
                        if grabbed != True:
                            continue
                        
                        pos = givebox(vinfo['vidname'],path,net,frame,frame_index,doplot = False,docv = False)
                        if len(pos) > 1:
                            boxdatacol.insert_many(pos)
                        
                        elif len(pos) == 1:
                            boxdatacol.insert_one(pos[0])
                            
                        frame_index += 1
                    
                    reminder = 4 - (total-1-reminder)%5
                    
                    boxcol.update_one({'vidname':vinfo['vidname']},{'$set':{'enddate':str(datetime.now()),'reminder':reminder}})
                    vidcol.update_one({'_id':vinfo['_id']},{'$set' :{'total':total,'status':1}})
                    
                    if vname != None:
                        break
                    vinfo = None
                    restart_frame = 0
                    start = 0
                    vprev = None
                    reminder = 0
                    
                elif waitc==0:
                    waitc += 1
                    print('waiting ',waitc)
                    time.sleep(3)
                    
                else:
                    break
        
        except KeyboardInterrupt as e:
            raise e
        
    
    def calc_id(self,ftplink,day,jpath=None):
        
        try:
            vidcol = self.videodb['video']
            boxcol = self.videodb['box_file']
            boxdatacol = self.videodb['box_data']
            deepcol = self.videodb['deep_file']
            deepdatacol = self.videodb['deep_data']
            deepstate = self.videodb['deep_state']
            process(ftplink,day,vidcol,boxcol,boxdatacol,deepcol,deepdatacol,deepstate,jpath)
            
        except KeyboardInterrupt as e:
            raise e
                
        
    def outvid(self,basepath='C:\\Users\\81807\\Documents\\RD\\realdata\\',link='192.168.99.101',day='RecordFolder20201005',size=(1000,600)):
        
        waitc = 0
        videodb = self.mongoconnect(dbname='videoDB')
        deepcol = videodb['deep_data']
        
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
    
        
        

