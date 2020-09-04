#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
import imutils
if '/test/Ito/Code' not in sys.path:
    sys.path.append('/test/Ito/Code')
from dataprep import imgplot,displt
import matplotlib.pyplot as plt
import pandas as pd

if '/test/RD/deep_sort' not in sys.path:
    sys.path.append('/test/RD/deep_sort')
    sys.path.append('/test/RD/deep_sort/deep_sort')

import tensorflow as tf
from tensorflow import keras
from deep_sort import preprocessing,nn_matching                                                                        
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection 

if '/test/RD/yolo-frcnn' not in sys.path:
    sys.path.append('/test/RD/yolo-frcnn')
from fcnn import mprocess, bcenter_xywh, bcenter_tlbr, load_csv
import mxnet as mx
import gluoncv
from gluoncv import model_zoo, data, utils
import gc
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

pos = []
frame_index = -1

def main(df,fpath,opath=None,video=False):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   
    
   # deep_sort 
    model_filename = '/test/RD/deep_sort/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    total = -1
    writeVideo_flag = False
    
    if video == True:
        fps = 0.0
        video_capture = cv2.VideoCapture(fpath)
        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(video_capture.get(prop))
            print("[INFO] {} total frames in video".format(total))
        except:
            print("[INFO] could not determine # of frames in video")
            total = -1
    
    else:
        files = np.sort(os.listdir(fpath))
        total = len(files)
        
    if writeVideo_flag:
    # Define the codec and create VideoWriter object

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(opath, fourcc, 5, (1000, 562))
        list_file = open('/test/RD/detection.txt', 'w')
    
    frame_index = -1 
    
    while frame_index+1 < 30:
            
        if video:
            ret, frame = video_capture.read()  # frame shape 562*1000*3
            if ret != True:
                break
        elif '.jpg' in files[frame_index+1] or '.png' in files[frame_index+1]:
            frame = cv2.imread(fpath+files[frame_index+1])
            h,w =frame.shape[:2]
            if w/h>10/6:
                h = int(1000*h/w+0.5)
                w = 1000
            else:
                w = int(600*w/h+0.5)
                h = 600
            frame = cv2.resize(frame,(w,h),interpolation = cv2.INTER_AREA)
        else:
            frame_index+=1
            continue
            
        t1 = time.time()   
        #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = df[df['f']==frame_index][['x','y','w','h']].to_numpy()
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        #indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        #detections = [detections[i] for i in indices]
        
        print('--------\n')
        print('Detections length',len(detections))
        
        # Call the tracker
        tracker.predict()
        m = tracker.update(detections,frame,frame_index+1,encoder)
        
        cv2.putText(frame, str(frame_index),(70, 70),cv2.FONT_HERSHEY_SIMPLEX,0.5, (80,120,240), 2)
        print('tid\n')
        
        for track in tracker.tracks:
            color = (255,255,255)
            bbox = track.to_tlbr()
            if not track.is_confirmed() or track.time_since_update > 1:
                color = (210,30,240)
                print(track.is_confirmed(),track.time_since_update,track.track_id)
            else:
                print('confirmed     ',track.track_id)
            
            if track.is_confirmed() and track.time_since_update==0:
                pos.append((frame_index,track.track_id,bbox,m[track.track_id],detections[m[track.track_id]].to_tlbr()))
            else:
                pos.append((frame_index,track.track_id,bbox))
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5, (20,240,60), 2)
            
        for det_no,det in enumerate(detections):
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            cv2.putText(frame, str(det_no),(int(bbox[0]), int(bbox[1]+10)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (20,40,240), 2)
            
        #cv2.imwrite('/test/RD/Images/'+str(frame_index)+'.jpg',frame)
        print(files[frame_index+1])
        frame_index = frame_index + 1
        imgplot(frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
    if video:
        video_capture.release()
        
    if writeVideo_flag:
        out.release()
        list_file.close()
        
        


def boxprocess(pos):
    df = pd.DataFrame(pos)
    df = df.fillna(0)
    df[5] = [bcenter_tlbr(i) if type(i)!=type(0) else 0 for i in df[2]]
    
    idty = 1
    l1 = []
    t1c = np.array([list(bcenter_tlbr(i)) if type(i)!=type(0) else 0 for i in df[df[1]==1][2]])
    d1c = np.array([list(bcenter_tlbr(i)) if type(i)!=type(0) else 0 for i in df[df[1]==1][4]])
    
    t1 = [i for i in df[df[1]==1][2]]
    t1 = [[index-1]+list(i) for index,i in enumerate(t1) if type(i)!=type(0)]
    t1 = np.array(t1)
    
    d1 = [i for i in df[df[1]==1][4]]
    d1 = [[index-1]+list(i) for index,i in enumerate(d1) if type(i)!=type(0)]
    d1 = np.array(d1)
    
    d3 = [i for i in df[df[1]==3][4]]
    d3 = [[index-1]+list(i) for index,i in enumerate(d3) if type(i)!=type(0)]
    d3 = np.array(d3)
    
    t3 = [i for i in df[df[1]==3][2]]
    t3 = [[index-1]+list(i) for index,i in enumerate(t3) if type(i)!=type(0)]
    t3 = np.array(t3)
    
    # iou plot
    plt.plot(np.arange(30),[bb_iou(t1[i][1:], t3[i][1:]) for i in range(30)])
    for i in range(30):
        plt.text(i,bb_iou(t1[i][1:], t3[i][1:]),str(i-1))
    
    
    index = 0
    for frame in range(-1,28):
        bb = df[df[0]==frame][2].to_numpy()
        ids = df[df[0]==frame][1].to_numpy()
        
        m = bb.shape[0]
        miou = [[] for i in range(m)]
        for i in range(m):
            for j in range(i+1,m):
                if bb_iosb(bb[i],bb[j])>0.2:
                    miou[i].append(ids[j])
                    miou[j].append(ids[i])
        df[6][index:index+m] = miou
        index = index+m

    
if __name__ == '__main__':
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
    pos = []
    opath = '/test/RD/output2.avi'
    vpath = '/test/RD/videos/walk.mp4'
    ipaths = ['/test/RD/MOT16/train/MOT16-04/img1/',
              '/test/RD/test/']
    frame_index = -1 
    
    df = pd.read_csv('/test/RD/det.csv',delimiter=',',names=['f','x','y','w','h']).drop([0],axis=0).astype(float)
    main(df,ipaths[0],opath,video=False)




