#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import warnings
import sys
import cv2
import time
import numpy as np
from datetime import datetime
import datetime as dtlib
#from PIL import Image
import imutils

import pandas as pd
import pymongo
if 'c:\\users\\81807\\documents\\RD\\deep_sort' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\deep_sort')

import tensorflow as tf
from deep_sort import preprocessing,nn_matching                                                                        
from deep_sort.tracker import Tracker
from deep_sort.track import Track
from tools import generate_detections as gdet
from deep_sort.detection import Detection 
from analysis import *

#import gc

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

pos = []
trinf = []
frame_index = -1
savetr = None
savemt = None

def loadtr(jsontr,tracker,metric):
    
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
        #metric.samples[int(trid)] = [np.array(tr['fts'])]
        tr['fts'] = np.array(tr['fts'])
        track = Track(mean, cov, int(trid), tr['n_init'], tr['maxage'], tr['checkspot'],
                 feature=tr['fts'])
        track.reinit(tr['hits'],tr['age'],tr['tsu'],pastpos,tr['dh'],tr['dhd'],iosb,tr['state'])
        trs.append(track)
    
    middle_check = {}
    for ntrid,t in jsontr['middle_check']:
        middle_check[int(ntrid)] = t
        
    critical_tracks = {}
    for ti_id,val in jsontr['critical_tracks']:
        critical_tracks[int(ti_id)] = [int(val[0]),float(val[1])]
    
    tracker.tracks = trs
    tracker.middle_check = middle_check
    tracker.critical_tracks = critical_tracks
    tracker._next_id = int(jsontr['next_id'])
    
    active_targets = [t.track_id for t in tracker.tracks if t.is_confirmed()]
    #print('active targets ',active_targets)
    
    features, targets = [], []
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        features += track.features
        targets += [track.track_id for _ in track.features]
        track.features = []
    #print('test ',active_targets,targets,np.array(features).shape) 
    metric.partial_fit(
        np.asarray(features), np.asarray(targets), active_targets)
    
    return tracker,metric


def freeze(deepstatecol,vidname,camname,tracker,metric,basepath='c:\\users\\81807\\documents\\RD\\realdata\\'):
    jsontr = {}
    trs = {}
    for tr in tracker.tracks:    
        fts = None
        pastpos = [list(i) for i in savetr.tracks[0].pastpos]
        if tr.track_id in metric.samples.keys():
            fts = metric.samples[tr.track_id][0].tolist()
        if fts == None or len(fts)==0:
            fts = tr.features
        trs[str(tr.track_id)] = {'mean':tr.mean.tolist(),
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
    
    jsontr['vidname'] = vidname
    jsontr['camname'] = camname
    jsontr['tracks'] = trs
    jsontr['middle_check'] = tracker.middle_check
    jsontr['critical_tracks'] = tracker.critical_tracks
    jsontr['next_id'] = tracker._next_id
    
    if deepstatecol.find_one({'camname':camname}):
        deepstatecol.update_one({'camname':camname},{'$set':{'vidname':vidname,'modified_time':datetime.now(),'tracks':trs,'middle_check':tracker.middle_check,'critical_tracks':tracker.critical_tracks,'next_id':tracker._next_id}})
    else:
        deepstatecol.insert_one({'camname':camname,'vidname':vidname,'modified_time':datetime.now(),'tracks':trs,'middle_check':tracker.middle_check,'critical_tracks':tracker.critical_tracks,'next_id':tracker._next_id})
    write_json(basepath+camname+'.json',jsontr)


def dsort(deepdatacol,vname,df,total,frame_index,metric,tracker,encoder,vs=None,files=None):
    
    loc_index = 0
    icount = 0
    pos = []

    while icount < len(df):
        
        if vs != None:
            ret, frame = vs.read()  # frame shape 562*1000*3
            
            if ret != True:
                icount += 1
                frame_index += 1
                continue
            
            frame = reshapeimg(frame)
            if loc_index != df.iloc[icount]['f']:
                loc_index += 1
                frame_index += 1
                continue
            
        else:
            icount += 1
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
            pos.append({'vidname':vname,'f':loc_index,'tid':track.track_id,'tx':bbox[0],'ty':bbox[1],'bx':bbox[2],'by':bbox[3]})
        
        
        if len(pos) > 1:
            deepdatacol.insert_many(pos)
            
        elif len(pos) == 1:
            deepdatacol.insert_one(pos[0])
        
        frame_index += 1 
        loc_index += 1
        icount += 1
        pos = []
        
    return frame_index,tracker


def main_dsort(ftplink,day,vidcol,boxcol,boxdatacol,deepcol,deepdatacol,deepstatecol,jpath):
    global savetr, savemt
   # Definition of the parameters
   
    max_cosine_distance = 0.3
    nn_budget = None

   # deep_sort 
    model_filename = 'C:\\Users\\81807\\Documents\\RD\\deep_sort\\model_data\\mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    total = -1
    frame_index = 0
    waitc = 0
    boxinfo = None
    
    unfinished = deepcol.find_one({'enddate':{'$exists':False}})
    if unfinished != None:
        deepdatacol.delete_many({'vidname':unfinished['vidname']})
        boxinfo = boxcol.find_one({'vidname':unfinished['vidname']}) 
        deepcol.delete_one({'vidname':unfinished['vidname']})
        
    while True:

        if boxinfo == None:
            boxinfo = boxcol.find_one({'status':0,'dayname':day})
        
        vinfo = vidcol.find_one({'status':1,'vidname':boxinfo['vidname']})
        
        if boxinfo['start']==0:
            _next_id = tracker._next_id
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric,_next_id=_next_id)
        
        elif vinfo == None:
            if waitc > 1:
                break
            waitc += 1
            boxinfo = None
            time.sleep(3)

        else:
            print('inside jpath loading...')
            print(vinfo['vidname'],vinfo['camname'])
            vprev = vidcol.find_one({'status':2,'camname':vinfo['camname'],'end_time':{'$gte':vinfo['start_time']-dtlib.timedelta(seconds=1),'$lte':vinfo['start_time']}},sort=[('end_time',pymongo.DESCENDING)])
            #fulljpath = jpath + vinfo['camname']+'.json'
            
            jsontr = deepstatecol.find_one({'camname':vinfo['camname']})
            print('jsontr ',jsontr['vidname'])
            print(vprev['vidname'])
            
            if vprev != None and jsontr['vidname']==vprev['vidname']:
                tracker,metric = loadtr(jsontr,tracker,metric)
            else:
                print('insufficient information ')
                break
        
        if boxinfo != None:
            print('deepsort ',boxinfo['vidname'],vinfo['camname'])
            try:
                df = pd.DataFrame([[r['f'],r['x'],r['y'],r['w'],r['h']] for r in boxdatacol.find({'vidname':boxinfo['vidname']})],columns=['f','x','y','w','h'])
                vs = cv2.VideoCapture(ftplink+boxinfo['vidname'])
            
            except Exception:
                print('File not found')
                waitc += 1
                time.sleep(3)
                continue
            
            
            # try to determine the total number of frames in the video file
            try:
                prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                    else cv2.CAP_PROP_FRAME_COUNT
                total = int(vs.get(prop))
                print("[INFO] {} total frames in video".format(total),boxinfo['vidname'])
                deepcol.insert_one({'vidname':boxinfo['vidname'],'startdate':str(datetime.now()),'status':0})
                
            except:
                print("[INFO] could not determine # of frames in video")
                total = -1
            
            frame_index,tracker = dsort(deepdatacol,boxinfo['vidname'],df,total,frame_index,metric,tracker,encoder,vs)
            savetr = tracker
            savemt = metric
            freeze(deepstatecol,vinfo['vidname'],vinfo['camname'],tracker,metric)
            
            boxcol.update_one({'_id':boxinfo['_id']},{'$set' :{'status':1}})
            vidcol.update_one({'vidname':boxinfo['vidname']},{'$set' :{'status':2}})
            deepcol.update_one({'vidname':boxinfo['vidname']},{'$set':{'enddate':str(datetime.now())}})
            boxinfo = None
            
            print(len(tracker.tracks))
            print('ok!')

        elif waitc==0:
            waitc += 1
            print('waiting ',waitc)
            time.sleep(3)
        
        else:
            break
            vs.release()


def process(ftplink,day,vidcol,boxcol,boxdatacol,deepcol,deepdatacol,deepstatecol,jpath = 'c:\\users\\81807\\documents\\RD\\realdata\\'):
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
            
    
    main_dsort(ftplink,day,vidcol,boxcol,boxdatacol,deepcol,deepdatacol,deepstatecol,jpath='c:\\users\\81807\\documents\\RD\\realdata\\')





