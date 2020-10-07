#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import warnings
import sys
import cv2
import time
import numpy as np
from datetime import datetime
#from PIL import Image
import imutils

import pandas as pd

if 'c:\\users\\81807\\documents\\RD\\deep_sort' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\deep_sort')

import tensorflow as tf
from deep_sort import preprocessing,nn_matching                                                                        
from deep_sort.tracker import Tracker
from deep_sort.track import Track
from tools import generate_detections as gdet
from deep_sort.detection import Detection 
from analysis import *
from stream import dbconnect


#import gc

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

pos = []
trinf = []
frame_index = -1
savetr = None
savemt = None

def loadtr(jpath,tracker,metric):
    
    jsontr = read_json(jpath)
    trs = []    
    
    for trid in jsontr['tracks'].keys():
        #Track(mean, covariance, self._next_id, self.n_init, self.max_age, self.check_spot(detection.to_xyah(),shape,fno),
        #    detection.feature)
        tr = jsontr['tracks'][trid]

        mean = np.array(tr['mean'],dtype=np.float)
        cov = np.array(tr['cov'],dtype=np.float)
        #tr['hits'] = int(tr['hits'])
        #tr['age'] = int(tr['age'])
        #tr['tsu'] = int(tr['tsu'])
        pastpos = [np.array(i) for i in tr['pastpos']]
        iosb = tuple(tr['iosb'])
        #tr['n_init'] = int(tr['n_init'])
        #tr['maxage'] = int(tr['maxage'])
        metric.samples[int(trid)] = [np.array(tr['fts'])]
        tr['fts'] = [np.array(tr['fts'])]
        
        track = Track(mean, cov, int(trid), tr['n_init'], tr['max_age'], tr['checkspot'],
                 feature=tr['fts'])
        track.reinit(tr['hits'],tr['age'],tr['tsu'],pastpos,tr['dh'],tr['dhd'],iosb,tr['state'])
        
        trs.append(track)
    
    middle_check = {}
    for ntrid,time in jsontr['middle_check']:
        middle_check[int(ntrid)] = time
        
    critical_tracks = {}
    for ti_id,val in jsontr['critical_tracks']:
        critical_tracks[int(ti_id)] = [int(val[0]),float(val[1])]
    
    tracker['tracks'] = trs
    tracker['middle_check'] = middle_check
    tracker['critical_tracks'] = critical_tracks
    tracker['_next_id'] = int(jsontr['next_id'])
    
    return tracker,metric


def freeze(tracker,metric,vname,loc_no):
    jsontr = {}
    trs = {}
    for tr in tracker.tracks:    
        fts = None
        pastpos = [list(i) for i in savetr.tracks[0].pastpos]
        if tr.track_id in metric.samples.keys():
            fts = metric.samples[tr.track_id][0].tolist()
            
        trs[tr.track_id] = {'mean':tr.mean.tolist(),
                            'cov':tr.covariance.tolist(),
                            'hits':tr.hits,
                            'age':tr.age,
                            'tsu':tr.time_since_update,
                            'pastpos':pastpos,
                            'dh':tr.dh,
                            'dhd':tr.dhd,
                            'iosb':tr.iosb,
                            'theta':tr.theta,
                            'checkspot':tr.checkspot,
                            'fts':fts,
                            'state':tr.state,
                            'n_init':tr._n_init,
                            'maxage':tr._max_age}

    jsontr['tracks'] = trs
    jsontr['middle_check'] = tracker.middle_check
    jsontr['critical_tracks'] = tracker.critical_tracks
    jsontr['next_id'] = tracker._next_id
    
    write_json(basepath+'dsfreeze.json',jsontr)


def dsort(folderp,sname,df,total,frame_index,pos,metric,tracker,encoder,vs=None,files=None):
    
    loc_index = 0
    icount = 0
    pindex = frame_index
    while icount < len(df):
        
        if vs != None:
            ret, frame = vs.read()  # frame shape 562*1000*3
            frame = reshapeimg(frame)
            if loc_index != df.iloc[icount]['f']:
                loc_index += 1
                frame_index += 1
                continue
            
            if ret != True:
                frame_index += 1
                continue

        elif '.jpg' in files[frame_index] or '.png' in files[frame_index]:
            frame = cv2.imread(fullp+files[frame_index])
            frame = reshapeimg(frame)
            
        else:
            frame_index+=1
            continue
        
        #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = df[df['f']==loc_index][['x','y','w','h']].to_numpy()  # tlwh format
        
        features = encoder(frame,boxs)
        
        # score to 1.0 here.
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # predict and update
        tracker.predict(frame.shape[:2])
        _ = tracker.update(detections,frame,frame_index,encoder)
        #trinfo = [(tr.track_id,tr.time_since_update) for tr in tracker.tracks]
        
        for track in tracker.tracks:
            bbox = track.to_tlbr()
            pos.append((loc_index,track.track_id,bbox[0],bbox[1],bbox[2],bbox[3]))
       
        if icount%10 == 0 or icount == len(df)-1:
            print(icount)
            df2 = pd.DataFrame(pos)
            df2.to_csv(folderp+sname+'+id.csv',header=None,index=False,mode='a')
            pos = []
            #print(frame_index,len(tracker.tracks))
            
        frame_index += 1 
        loc_index += 1
        icount += 1
        
    return frame_index,pos,tracker


def main_dsort(link,day,basepath,boxcol,deepcol,jpath=None):
    global savetr, savemt
   # Definition of the parameters
   
    fullp = '/ftp_share/VideoVolume1/'+day+'/'
    folderp = basepath+day+'\\'
    max_cosine_distance = 0.3
    nn_budget = None

   # deep_sort 
    model_filename = 'C:\\Users\\81807\\Documents\\RD\\deep_sort\\model_data\\mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    total = -1
    count = 0
    frame_index = 0
    waitc = 0
    pos = []

    
    if jpath:
        tracker,metric = loadtr(jpath,tracker,metric)

    while True:
        boxinfo = boxcol.find_one({'status':0})
        
        if boxinfo != None:
            try:
                df = pd.read_csv(basepath+day+'\\'+boxinfo['name'],delimiter=',',names=['f','x','y','w','h'])
                vs = cv2.VideoCapture('ftp://admin:admin@'+link+fullp+boxinfo['name'][:-4]+'.avi')
            
            except FileNotFoundError:
                print('File not found')
                waitc += 1
                time.sleep(3)
                continue
            
            # try to determine the total number of frames in the video file
            try:
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                    else cv2.CAP_PROP_FRAME_COUNT
                total = int(vs.get(prop))
                print("[INFO] {} total frames in video".format(total),boxinfo['name'])
            except:
                print("[INFO] could not determine # of frames in video")
                total = -1
            
            frame_index,pos,tracker = dsort(folderp,boxinfo['name'][:-4],df,total,frame_index,pos,metric,tracker,encoder,vs)
            savetr = tracker
            savemt = metric
            boxcol.update_one({'_id':boxinfo['_id']},{'$set' :{'status':1}})
            deepcol.insert_one({'name':boxinfo['name'][:-4]+'+id.csv','date':str(datetime.now()),'status':0})
            print(len(tracker.tracks))
            print('ok!')

        elif waitc==0:
            waitc += 1
            print('waiting ',waitc)
            time.sleep(3)
        
        else:
            break
            vs.release()


def process(day='RecordFolder20201005'):
    tf.compat.v1.enable_eager_execution() 
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
    basepath = 'C:\\Users\\81807\\Documents\\RD\\realdata\\'
    link = '192.168.99.101'
    
    
    boxdb = dbconnect(dbname='boxcsv')
    boxcol = boxdb[day]
    deepdb = dbconnect(dbname='deepcsv')
    deepcol = deepdb[day]
    
    main_dsort(link,day,basepath,boxcol,deepcol)





