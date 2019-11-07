# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:28:45 2019

@author: AZEST-2019-07
"""


import skimage.io as io
import sys
import os
import cv2
import json
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import tqdm

base = 'D:\\Ito data\\'

dicom_path = base + 'AI'
jpg_path = base + 'AI2'
annotated_path = base + 'annotated'
overlap_path = base + 'overlap'

sample = None
imgdata = None
#loads data from folder and saves it in dict, key is patientID, value is images in list
def load_dicom(fpath):    
    global imgdata,sample
    imgd = {}
    for path,subdir,files in os.walk(fpath):
        name = os.path.basename(path)
        imglist = []
        for file in files:
            full_path = path+ '\\' + file
            if int(file.replace('Image',''))%2 != 0:
                imgdata = pydicom.read_file(full_path)
                if sample == None:
                    sample = imgdata
                img = imgdata.pixel_array
                imglist.append(img[40:-40,:,:])
        if len(imglist) != 0:
            imgd[name] = imglist
    return imgd
imgd = load_dicom(dicom_path)
clr = np.random.rand(8,3)*255
maskcolor = dict((i+1,clr[i]) for i in range(8))


def load_jpg(fpath):
    imgj = {}
    for path,subdir,files in os.walk(fpath):
        name = os.path.basename(path)
        imglist = []
        for file in files:
            full_path = path+ '\\' + file
            
            if '.jpg' in full_path and 'red' in full_path:
                img = cv2.imread(full_path)
                imglist.append(img[40:-40,:,:])
            
        if len(imglist) != 0:
            imgj[name] = imglist
    return imgj
            

imgj = load_jpg(jpg_path)

# resizes image while maintaining aspect ratio
def img_resize(img,final_shape):
    
    m,n = img.shape
    mx = max(m,n)
    ratio = final_shape/mx
    #print(ratio)
    rimg = cv2.resize(img,dsize=None,fx=ratio,fy=ratio)
    ax = 0
    i1 = np.uint8(np.zeros((final_shape-min(rimg.shape),final_shape)))
    #print(i1.shape)
    if m>n:
        i1 = i1.T
        ax = 1
    rimg = np.uint8(np.concatenate([i1,rimg],axis=ax))
    
    assert rimg.shape == (final_shape,final_shape)
    return rimg
    #print(rimg.shape)

#segments the image based on the input points
def segment(img):
    points = []
    def mousepoint(event,x,y,flags,param): 
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y,img[y,x],'  --------')
            cv2.circle(img,(x,y),1,(255,255,255),-1)
            points.append((x,y))
            
    
    editimg = None
    def fill(img):
        global points,maskcolor,editimg
        editimg = np.uint8(np.zeros((img.shape[0],img.shape[1])))
        while(1):
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',mousepoint)
            points = []
            while(1):
                cv2.imshow('image',img)
                if cv2.waitKey(20) & 0xFF == 27:
                    #cv2.destroyAllWindows()
                    break
            cv2.destroyAllWindows()
            if points != []:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1,1,2))
                
                mask = input('Enter the value ')
                clr = maskcolor[int(mask)]
                x = int(clr[0])
                y = int(clr[1])
                z = int(clr[2])
                
                cv2.polylines(img,[pts],True,color = (x,y,z))
                cv2.polylines(editimg,[pts],True,color = int(mask))
                cv2.fillConvexPoly(img, points=np.array(points), color=(x,y,z))
                cv2.fillConvexPoly(editimg, points=np.array(points), color=int(mask))
                
                
                print('Area ',cv2.contourArea(pts,oriented = False))
                print('Length ',cv2.arcLength(pts,closed=True))
                cv2.imshow('image',img)
                if cv2.waitKey(20) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    break
            else:
                cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\demo.png',img)
                cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\demo2.png',editimg)
                cv2.destroyAllWindows()
                break
    fill(img)



#cv2.imshow('image2',editimg*30)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
    

def cut_alt(img):
    e1 = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    e1 = np.uint8(cv2.cvtColor(e1,cv2.COLOR_BGR2GRAY))
#    e1[e1>=100] = 255
#    e1[e1<100] = 0
    
    e = np.where(e1.sum(axis=1)/(255*e1.shape[1])>0.3)[0]
    top = e[0]
    tot = 0
    print(e)
    for i in range(e[0],e[0]+10):
        if i in e:
            tot += 1
        else:
            tot = 0
        if tot >= 7:
            top = i
    #print(e)
    ec = e1.sum(axis=0)/(255*e1.shape[0])
    print(np.round(ec*100))
    i = 0
    left = False
    right = False
    li = int(e1.shape[1]/2)
    ri = int(e1.shape[1]/2)
    for i in range(int(e1.shape[1]/2)):
        if left == False:
            li = int(e1.shape[1]/2-i)
            if ec[li]<0.1:
                left = True
        if right == False:
            ri = int(e1.shape[1]/2+i)
            if ec[ri]<0.1:
                right = True
    print(li,ri)
    cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\demo.png',e1)
    
    cv2.imshow('img',img[top:,li:ri])
    cv2.imshow('img2',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def cut(img):

    mono_img = np.sum(img, axis=2)
    #bin_img = np.sign(np.where((mono_img>140)&(mono_img<170), 0, mono_img))
    
    row_activate = np.zeros(mono_img.shape[0])
    col_activate = np.zeros(mono_img.shape[1])
    
    for row in range(mono_img.shape[0]):
        row_activate[row] = len(np.unique(mono_img[row]))
    for col in range(mono_img.shape[1]):
        col_activate[col] = len(np.unique(mono_img[:,col]))
    
    judge_len = 30
    judge_len_2 = 20
    min_unique_1 = 5
    min_unique_2 = 20
    
    top = 0
    bottom = mono_img.shape[0]-1
    for t in range(mono_img.shape[0]-judge_len):
        if all(row_activate[t:t+judge_len] >= min_unique_1):
            top = t
            for b in range(top, mono_img.shape[0]-judge_len_2):
                if all(row_activate[b:b+judge_len_2] < min_unique_2):
                    break
            bottom = b
            break
    
    judge_len = 30
    min_unique = 30
    left = 0
    right = mono_img.shape[1]-1
    for l in range(mono_img.shape[1]-judge_len):
        if all(col_activate[l:l+judge_len] >= min_unique):
            left = l
            for r in range(left, mono_img.shape[1]):
                if col_activate[r] < min_unique:
                    break
            right = r
            break
#    cut_img = img[top:bottom, left:right]
    
#    cv2.imshow('image',cut_img)
#    cv2.imshow('org',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return top, bottom, left, right



def scale(img):
    
    top,bottom,left,right = cut(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mx = np.max(img[:,:left])
    
    print(top,left,mx,img.shape)
    
    j = top - 10
    s = j
    length = 0
    repetition = 0
    col = 0
    row = 0
    length_prev = 0
    for i in range(left):
        j = np.max([0,top - 10])
        s = j
        length = 0
        repetition = 0
        length_prev = 0
        #print(i,' *')
        while j>=0 and j<(bottom -10):
            j += 1
            minmax = np.min([200,mx])
            
            if img[j,i]> minmax:
                #print(' ## ',i,j)
                
                #print('-- ',i,j,img[j,i],minmax,j,s,l,n,col)
                if s==0 and s!=j:
                    s = j
                    j += 15
                
                elif length==0 and s!=j:
                    
                    length = (j-s)
                    s = j
                    j += 15
                    
                elif (j-s)>0.9*length and  (j-s)<1.1*length:
                    
                    length_prev = length
                    length = j-s
                    s = j
                    repetition += 1
                    j += 15
                    if repetition >= 3:
                        row = j
                        col = i
                        break
                elif (j-s)<=0.9*length or (j-s)>=1.1*length:
                    
                    length = j-s
                    s = j
                    j += 15
        if col != 0:
            break
        
        
    print(col,length,length_prev)
    if col!=0:
        cv2.imshow('scale',cv2.resize(img[0:row+15,col-32:col+32], dsize=(128,2*(row+15))))
        cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\scale.png',cv2.resize(img[0:row+15,col-32:col+32], dsize=(128,2*(row+15))))
    cv2.imshow('org',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_image(imgd,img,color,k,index,c=5):
    top,bottom,left,right = cut(imgd[k][min(int(index/2),1,c)])          
    img = img[top:bottom,left:right]
    img2 = np.zeros(img.shape)
    r,c,d = img.shape
    #print(r,c,k,index,index/2)
    for i in range(r):
        for j in range(c):
            if (img[i,j]>[0,30,100]).sum()>=3 and (img[i,j]<[80,160,256]).sum()>=3:
                img2[i,j] = np.array(color)
    cv2.imwrite('D:\\Ito data\\annotated\\'+k+str(index)+'.png',img2)
#    cv2.imshow('img',img2)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    return img2
    
def create_map(imgd,imgj):
    for k,v in imgj.items():
        for i in range(len(v)):
            if i%2==0:
                color = [0,255,0]
            else:
                color = [0,255,255]
            create_image(imgd,v[i],color,k,i)


#create_map(imgd,imgj)
#create_image(imgd,imgj['05'][2],[0,255,255],'05',2,0)
#create_image(imgd,imgj['05'][3],[0,255,0],'05',3)
#create_image(imgd,imgj['05'][4],[0,255,255],'05',4)

def load_annotation(apath):
    imgA = {}
    m = '01'
    imglist = []
    for path,subdir,files in os.walk(apath):
        for file in files:
            full_path = path+'\\'+file
            
            if m != file[:2]:
                imgA[m] = imglist
                m = file[:2]
                imglist = []
                
            imglist.append(cv2.imread(full_path))
        imgA['07'] = imglist
        return imgA


def add(i1,i2,k,index):
    r0,c0,d0 = i1.shape
    r1,c1,d1 = i2.shape
    
    r,c = min(r0,r1),min(c0,c1)
    #print(r,c)
    
    i1 = i1[0:r,0:c,:]
    i2 = i2[0:r,0:c,:]
    
    i3 = cv2.add(i1,i2)
    cv2.imwrite('D:\\Ito data\\overlap\\'+k+str(index)+'.png',i3)
#    cv2.imshow('i1',i1)
#    cv2.imshow('i2',i2)
#    cv2.imshow('i3',i3)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    
def overlap(imgA):
    for k,v in imgA.items():
        for i in range(2):
            i1 = v[i*2]
            i2 = v[i*2+1]
            add(i1,i2,k,i)


#imgA = load_annotation(annotated_path)
#overlap(imgA)
#add(imgA['05'][3],imgA['05'][4],'05',1)


def one_hot(overlap_path,fd=6):   # 1-thyroid, 2-papillary, 3-benign, 4-cyst, 5-solid lesions 0-other
    imglist = []
    #thyroid green 
    #outline yellow
    #papillary red 
    #benign blue 252,3,22
    #cyst violet 239,27,218
    #solid-lesion pale white 180,207,216
    
    for path, subdir, files in os.walk(overlap_path):
        fimg_list = []
        for file in files:
            full_path = path + '\\' + file
            img = cv2.imread(full_path)     
            #reshape(img)
            r,c,d = img.shape
            
            fimg = np.uint8(np.zeros([r,c]))
            
            for i in range(r):
                for j in range(c):
                    b,g,r = img[i,j]
                    
                    if g>2*r and g > 2*b:            # green thyroid 1
                        fimg[i,j] = 1
                        
                    elif r>2*g and r>2*b:            # red papillary 2
                        fimg[i,j] = 2 
                        
                    elif b<60 and g<60 and r<60:     # black other 0
                        fimg[i,j] = 0
                        
                    elif b>200 and b>3*g and b>3*r:  # blue benign 3 
                        fimg[i,j] = 3
                        
                    elif b>180 and r>180 and g<80:   # violet cyst 4
                        fimg[i,j] = 4
                        
                    elif r>180 and g>180 and b>150:  # pale white solid-lesion 5
                        fimg[i,j] = 5
                        
                    elif r>200 and g>200 and b<60:   # yellow outline 1
                        fimg[i,j] = 1
                        
            fimg = to_categorical(fimg,fd)
            fimg_list.append(fimg)
            return fimg_list
final_dim = 6
fhot = one_hot(overlap_path,final_dim)

    
