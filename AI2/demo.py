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
from analysis import *
import gc

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

pos = []
frame_index = -1

def main(df,base_path,fpath,opath=None,video=False,savepos=True):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    #nms_max_overlap = 1.0

   # deep_sort 
    model_filename = base_path + 'deep_sort\\model_data\\mars-small128.pb'
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
        list_file = open(base_path + 'detection.txt', 'w')
    
    ctrl = 1
    ctrl2 = 1
    frame_index = -1 
    
    while frame_index+1 < 95:
        
        if video:
            ret, frame = video_capture.read()  # frame shape 562*1000*3
            if ret != True:
                break
        
        if ctrl2 % min(5,ctrl) != 0:
            ctrl2 += 1
            frame_index += 1
            continue
        
        if frame_index%2==0 and ctrl<6:
            ctrl+=1
        ctrl2=1
    
        if '.jpg' in files[frame_index+1] or '.png' in files[frame_index+1]:
            frame = cv2.imread(fpath+files[frame_index+1])
            h,w = frame.shape[:2]
            if w/h>10/6:
                h = int(1000*h/w+0.5)
                w = 1000
            else:
                w = int(600*w/h+0.5)
                h = 600
            frame = cv2.resize(frame,(w,h),interpolation = cv2.INTER_AREA)
            print(frame.shape,files[frame_index+1])
        else:
            frame_index+=1
            continue
            
        t1 = time.time()   
        #image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = df[df['f']==frame_index][['x','y','w','h']].to_numpy()  # tlwh format
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
                #print(track.is_confirmed(),track.time_since_update,track.track_id)
            #else:
                #print('confirmed     ',track.track_id)
                
            if savepos ==True:
                if track.is_confirmed() and track.time_since_update==0:
                    pos.append((frame_index,track.track_id,bbox,m[track.track_id],detections[m[track.track_id]].to_tlbr()))
                else:
                    pos.append((frame_index,track.track_id,bbox))
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color, 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX,0.5, (20,240,60), 2)
            
        for det_no,det in enumerate(detections):
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            cv2.putText(frame, str(det_no),(int(bbox[0]), int(bbox[1]+10)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (60,250,250), 2)
            
        #cv2.imwrite('/test/RD/Images/'+str(frame_index)+'.jpg',frame)
        print(frame_index)
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
            
    if video:
        video_capture.release()
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
    if writeVideo_flag:
        out.release()
        list_file.close()


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
    base_path = 'C:\\Users\\81807\\Documents\\RD\\'
    opath = base_path + 'output2.avi'
    vpath = base_path + 'videos\\walk.mp4'
    ipaths = [base_path + 'MOT16\\train\\MOT16-04\\img1\\',
              base_path + 'test\\']
    frame_index = -1 
    
    df = pd.read_csv(base_path+'deep_sort\\det.csv',delimiter=',',names=['f','x','y','w','h']).drop([0],axis=0).astype(float)
    main(df,base_path,ipaths[0],opath,video=False,savepos=False)







