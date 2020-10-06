import numpy as np
import argparse
#import imutils
import cv2
import os
import sys
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imutils
from analysis import imgplot,save_csv,load_csv

import mxnet as mx
import gluoncv
from gluoncv import model_zoo, data, utils
import gc
#from analysis import *


    
def load_fcnn():
    GPU_COUNT = 2 # increase if you have more
    ctx = [mx.gpu(i) for i in range(GPU_COUNT)]

    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True,ctx = ctx)
    return net,ctx

def mprocess(im,net,ctx,confidence = 0.65,threshold = 0.3):
    
    i2,img = data.transforms.presets.rcnn.transform_test(im)
    i2 = i2.as_in_context(ctx[1])
    bid, scores, box = net(i2)
    #boxids, sc, bb = bid, scores, box

    bid = np.int32(bid[0].asnumpy())
    scores = scores[0].asnumpy()
    box = box[0].asnumpy()

    box = box[np.where(scores>0.1)[0]]
    bid = bid[np.where(scores>0.1)[0]]
    scores = scores[np.where(scores>0.1)[0]].reshape(-1)
    box[:,2] = box[:,2] - box[:,0]
    box[:,3] = box[:,3] - box[:,1]

    idxs = cv2.dnn.NMSBoxes(box.tolist(), scores.tolist(), confidence,threshold)
    idxs = [i[0] for i in idxs if bid[i]==0]
    box = np.array([list(box[i]) for i in idxs])
    scores = np.array([scores[i] for i in idxs])
    gc.collect()

    return box,scores,img

def pvideo(fpath,net,ctx):
    
    video_capture = cv2.VideoCapture(fpath)
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(video_capture.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        total = -1
        
    bbox = [[],[],[],[]]
    
    for frame_index in range(-1,total-1):
        print(frame_index)
        ret, frame = video_capture.read()  # frame shape 562*1000*3
        if ret != True:
            break
        b,s,_ = mprocess(mx.ndarray.array(frame),net,ctx)
        for i in range(b.shape[0]):
            
            bbox[0].append(frame_index)
            bbox[1].append(i)
            bbox[2].append(b[i])
            bbox[3].append(s[i])
    
    return pd.DataFrame(np.array(bbox).T,columns=['f','d','bb','c'])

def pimage(fpath,net,ctx):
    data = []
    files = np.sort(os.listdir(fpath))
    print(files)
    for i,f in enumerate(files):
        if not '.jpg' in f and not '.png' in f:
            continue
        im = cv2.imread(fpath+f)
        im = mx.ndarray.array(im)
        boxes,_,_ = mprocess(im,net,ctx)
        for box in boxes:
            data.append([i-1,box[0],box[1],box[2],box[3]])
    return data

# display bbox for an image
def bdisp(box,img,doplot=False):
    color = (50,100,255)
    img2 = img.copy()
    
    box = np.array(box).astype(int)
    #img2 = gluoncv.utils.viz.cv_plot_bbox(img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    for i in range(box.shape[0]):
        #print(bid[i],scores[i],box[i])

        x, y = box[i][0], box[i][1]
        w, h = box[i][2], box[i][3]
        cv2.rectangle(img2, (x, y), (x+w, y+h), color, 1)
        #mark ID
        text = str(i)
        cv2.putText(img2, text, (int(x+w/2)-8, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (100,220,80), 2)
        # mark center
        cv2.putText(img2, 'x', (int(x+w/2)-5, int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,100,65), 2)  
        
    if doplot == True:
        imgplot(img2)
        
    return img2

def bcenter_xywh(a):
    return np.array((a[0]+a[2]/2,a[1]+a[3]/2,a[3]/a[2]))

def bcenter_tlbr(a):
    x = a[0]
    y = a[1]
    x2 = a[2]
    y2 = a[3]
    return ((x+x2)/2,(y+y2)/2,x2-x,y2-y)



'''
im = cv2.imread('/test/RD/MOT16/train/MOT16-04/img1/000001.jpg')
im = mx.ndarray.array(im)
net,ctx = load_model()
box,scores,img = mprocess(im,net,ctx)
img2 = bdisp(box,img)
imgplot(img2)

pos = np.array([list(center(i)[:2]) for i in box])
temp = np.sum(np.square(pos),axis=1).reshape(-1,1)
dmat = np.round(np.sqrt((temp+temp.T) - 2*np.dot(pos,pos.T)))  # Distance matrix of all people



m,n = pos.shape
diff = np.array([pos - pos[i] for i in range(m)])

nn = [list(np.argsort(dmat[i])[1:5]) for i in range(m)]   # Nearest neighbours

for i in range(m):
    for j in nn[i].copy():
        if i not in nn[j]:
            nn[i].remove(j)

nnd = [[dmat[i][j] for j in nn[i]] for i in range(m)]     # Distance of nearest neighbours

for i in range(m):
    mn = np.min(nnd[i])
    nnc = nn[i].copy()
    for index,j in enumerate(nnd[i].copy()):
        if j>=142 or j>=(mn+10)*2:
            nnd[i].remove(j)
            nn[i].remove(nnc[index])


#cv2.imwrite('/test/RD/p.jpg',cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))

'''

