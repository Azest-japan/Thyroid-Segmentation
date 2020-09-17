# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:41:55 2020

@author: 81807
"""

import numpy as np
import sys
if 'c:\\users\\81807\\documents\\RD\\deep_sort' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\deep_sort')
    
from camera import loadjson,findh,findxy,td3d,xyzpos,perror
from analysis import read_json,imgplot,mouse
import cv2

import os
import imutils
import pandas as pd

basepath = 'C:\\Users\\81807\\Documents\\RD\\'
cp = loadjson(basepath+'deep_sort\\camera_parameters.txt')
df2 = pd.read_csv(basepath+'fdet.csv',delimiter=',',names=['f','id','tx','ty','bx','by'])
ids = np.unique(df2['id'])
idsc = ids.tolist()
for i in ids:
    if df2[df2['id']==i].shape[0]<3:
        idsc.remove(i)
ids = idsc


shape = (563,1000)

df2['botx'] = (df2['tx']+df2['bx'])/2



def corrpos(cp,x,y,w,h,needz=False):
    mtx,dist,rv,tv = cp['mtx'],cp['dist'],cp['rv'],cp['tv']
    
    (px,py),_ = td3d((x+w/2, y+h),mtx,dist,rv,tv)
    #px = px - 15  # account for the foot size (30/2)
    if needz:
        for z in range(150,200):
            u1,v1 = cv2.projectPoints(np.float32([px,py,z]), rv, tv, mtx, dist)[0].reshape(-1)
            if v1<y or u1<x:
                print(u1,v1)
                break

        return (px,py,z)
    return px,py



a = np.float32([[  0,   0,   0],    #2
        [800, 1300, 0],     #4
        [2450,  0,   0],    #8
        [2350, 1270, 0],    #5
        [910, 650,   0],    #9
        [2450, 1260, 0],    #6
        [2450, 435,   0],   #12
        [0, 435, 0],       #15
        [900,  35,   0],  #1
        [2450, 450,   0],   #13
        [0,   1070,   0],   #3
        [920, 275,   0]])    #7
        
        #[2250, 180,   0],   #10
        #[930, 870, 0],      #11
        #[2450, 1050,   0]])  #14
        
        
        
#a = np.vstack((a[:,1],a[:,0],a[:,2])).T
b = np.float32([[[176, 28]],    #2
        [[784,107]],    #4
        [[81, 408]],    #8
        [[924,346]],    #5
        [[486, 145]],   #9
        [[930,375]],    #6
        [[401, 411]],   #12
        [[376, 31]],    #15
        [[164, 150]],  #1
        [[411, 410]],   #13
        [[620, 31]],    #3
        [[278, 152]]])   #7
        
        #[[222, 330]],   #10
        #[[562, 156]],   #11
        #[[785, 384]]])   #14

thmse = 100
optk = None
for i in range(100):
    k = np.random.permutation(a.shape[0])
    a1 = a[k]
    b1 = b[k]
    op = [a1]
    ip = [b1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(op, ip, shape[::-1],None,None)      
    rv,tv,mse = perror(ip,op,mtx,dist)
    if mse<thmse:
        thmse = mse
        opt = ret, mtx, dist, rvecs, tvecs

ret, mtx, dist, rvecs, tvecs = opt

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(basepath+'project.avi',  0, 4, (1920,692))

indices = np.unique(df2['f'])
for frame_index in indices:
    df3 = df2[df2['f']==frame_index][['id','botx','by']]
    c_img = np.zeros((shape[0],shape[0],3)).astype(np.uint8)
    cv2.line(c_img,(63,63),(562,63),(0,120,240),2)
    cv2.line(c_img,(63,63),(63,562),(0,120,240),2)
    cv2.putText(c_img, '(0,0)',(55, 50),cv2.FONT_HERSHEY_SIMPLEX,0.75, (60,250,250), 2)
    img = cv2.imread(basepath+'motout2\\'+str(frame_index)+'.jpg')
    for i in range(df3.shape[0]):
        
        px = tuple(df3.iloc[i])
        if not px[0] in ids:
            continue
        rc,_ = td3d(px[1:3], mtx, dist, rv, tv)
        
        pcx = int(rc[0]*5/30 + shape[0] - 500)
        pcy = int(rc[1]*5/25 + shape[0] - 500)
            
        cv2.circle(c_img,(pcy,pcx),5,(255,255,255),-1)
        cv2.putText(c_img, str(px[0]),(pcy, pcx),cv2.FONT_HERSHEY_SIMPLEX,0.5, (60,250,250), 2)
    
    fimg = np.hstack((img,c_img))
    fimg = cv2.resize(fimg,(1920, 692),interpolation = cv2.INTER_AREA)
    out.write(img)
    #imgplot(fimg)
    #cv2.imwrite(basepath+'motout_c\\'+str(frame_index)+'.jpg',fimg)



def img2vid(fpath = basepath+'motout_c',outpath=basepath+'project.avi'):
    img_array = []
    out = cv2.VideoWriter(outpath,  0, 4, (1920,692))
    
    for path,subdir,files in os.walk(fpath):
        files = [i.split('.')[0] for i in files]
        files = np.sort(np.array(files).astype(int))
        for file in files:
            
            img = cv2.imread(basepath+'motout_c\\'+str(file)+'.jpg')
            img_array.append(img)
    
    for i in range(len(img_array)):
        img  = img_array[i]
        img = cv2.resize(img,(1920, 692),interpolation = cv2.INTER_AREA)
        out.write(img)
        
    out.release()








                                        
