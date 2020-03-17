# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:32 2019

@author: AZEST-2019-07
"""

import numpy as np
import os 
import sys
import cv2

class Patient:
    
    def __init__(self,details):
        self.orgimage = []    #list of 2 images
        self.details = details      # dict containing patient details
        self.top = None        
        self.bottom = None        
        self.left = None
        self.right = None
        self.scimage = []
        
    def addimage(self,img):
        self.orgimage.append(img)
        
    def cut(self,dims):
        top,bottom,left,right = dims
        self.top = top              
        self.bottom = bottom        
        self.left = left
        self.right = right
        
    def dist(self,scol,scale):
        self.scale = scale          #list of 2 values
        self.scol = scol
        
    def reborder(self,fimg):
        self.fimg = fimg
    
    def annotated(self,fannotated):
        self.annotated = fannotated

    def display(img):
        cv2.imshow('img',img)
        cv2.waitKey(1800)
        cv2.destroyAllWindows()
        

def cut(img):
    # Sum the color modes, B+G+R, resulting dimension (length,breadth,1) 
    mono_img = np.sum(img, axis=2)

    row_activate = np.zeros(mono_img.shape[0])
    col_activate = np.zeros(mono_img.shape[1])
    
    for row in range(mono_img.shape[0]):
        #Count the number of unique values in each row
        row_activate[row] = len(np.unique(mono_img[row]))
    for col in range(mono_img.shape[1]):
        #Count the number of unique values in each column
        col_activate[col] = len(np.unique(mono_img[:,col]))
    
    judge_len = 30
    judge_len_2 = 20
    min_unique_1 = 5
    min_unique_2 = 30
    
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

    return top, bottom, left, right

def cut_man(img):
    points = []
    def mousepoint(event,x,y,flags,param): 
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y,img[y,x],'  --------')
            cv2.circle(img,(x,y),1,(255,255,255),-1)
            points.append((x,y))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mousepoint)
    
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            #cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    # top, bottom, left, right
    return points[0][1],points[1][1],points[0][0],points[1][0]


# Extract the column corresponding to the scale 
def extscale(img,top,bottom,left):
    
    #img = pobj.orgimage[index].copy()
    #top,bottom,left = pobj.top, pobj.bottom, pobj.left
    i2 = img.copy()
    
    
    # convert BGR image to gray scale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(img,160,255)
    img[mask==0]=0  # All pixels below 160 becomes 0
    mx = np.max(img[:,:left])  # max pixel value 
    minmax = np.min([200,mx])  
    print(top,bottom,left,mx,img.shape)
    
    crp = top - 10  # current row position
    prp = crp        # previous row position
    length = 0
    plength = 0
    repetition = 0
    col = []
    flist = []  # stores the pixel length between subsequent points in a column

    for c in range(left):
        crp = np.max([0,top - 10])
        prp = 0
        length = 0
        repetition = 0
        plength = 0
        
        while crp>=0 and crp<(bottom -10):
            crp += 1
         
            if img[crp,c]> minmax:
                
                if prp==0 and prp!=crp:
                    
                    prp = crp
                    crp += 15
                
                elif length==0 and prp!=crp  and np.average(img[crp-prp:crp,c])>10 and np.average(img[crp-prp:crp,c])<100:
                   
                    length = (crp-prp)
                    prp = crp
                    crp += 15
                    
                elif length!=0 and (crp-prp)>0.9*length and (crp-prp)<1.10*length and length>15 and np.average(img[crp-prp:crp,c])>=10 and np.average(img[crp-prp:crp,c])<100:
                   
                    plength = length
                    length = crp-prp
                    
                    repetition += 1                    
                    if repetition >= 3:
                        flist.append((crp,length,plength))
                        print('* ',c,crp,prp,prp-plength)
                        col.append(c)
                        repetition = 0
                        break
                    prp = crp
                    crp += 15
                    
                elif length!= 0 and ((crp-prp)<=0.9*length or (crp-prp)>=1.1*length):
                    length = crp-prp
                    prp = crp
                    crp += 15
                    
        if len(col)!=0:
            break
        
    print(col,flist)
    
    if len(col)!=0:
        cv2.imshow('scale',np.concatenate((np.concatenate((np.zeros([flist[0][0]+30-max(top-10,0),100]),img[max(top-10,0):flist[0][0]+30,col[-1]:col[-1]+1]),axis=1),np.zeros([flist[0][0]+30-max(top-10,0),100])),axis=1))
        #cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\scale.png',cv2.resize(img[0:row+15,col-32:col+32], dsize=(128,2*(row+15))))
    
    i2[:,col] = [0,100,255]
    cv2.imshow('org',i2[max(top-10,0):flist[0][0]+30,:left])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return col[0],flist[0]

def fillextra(img):
    # img is already border removed and cut image
    m = list(map(lambda x: 255 if (x[1]>50 or x[2]>50) and (1.8*x[0]<x[1] and 1.8*x[0]<x[2]) else 0, img.reshape(-1,3))) # select the pixels and form a 1D mask
    m = np.uint8(m).reshape((img.shape[0],img.shape[1])) # reshape it to a 2D mask
    dst = cv2.inpaint(img,m,3,cv2.INPAINT_TELEA) # fill the mask area
    return dst

#p1.display(p1.orgimage)

