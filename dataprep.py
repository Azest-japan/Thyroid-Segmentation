# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:28:45 2019

@author: AZEST-2019-07
"""



import sys
import os
import cv2
import json
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import model_from_json,load_model
import tqdm

base = 'D:\\Ito data\\'

dicom_path = base + 'AI'
jpg_path = base + 'AI2'
annotated_path = base + 'annotated'
overlap_path = base + 'overlap'
final_dim = 6

sample = None
imgdata = None

#loads data from folder and saves it in dict, key is patientID, value is images in list
def loadimg(fpath,ftype):
    
    global sample
    imgdict = {}       # Dictionary - key = Patient Number(String) Value = Patient Images(List)
    
    #Load Dicom Images
    if ftype == 'dicom':
        for path,subdir,files in os.walk(fpath):
            name = os.path.basename(path)  # Patient ID as Key
            imglist = []                   # List of Patient Images
            for file in files:
                full_path = path+ '\\' + file
                if int(file.replace('Image',''))%2 != 0:
                    imgdata = pydicom.read_file(full_path)
                    if sample == None:
                        sample = imgdata
                    img = imgdata.pixel_array
                    imglist.append(img[40:-40,:,:]) # cut the top and bottom borders (40 px)
                    
            if len(imglist) != 0:
                imgdict[name] = imglist   # Dict[key] = List
    
    #Load raw Jpeg Images
    elif ftype == 'jpg':
        for path,subdir,files in os.walk(fpath):
            name = os.path.basename(path)  # Patient ID as Key
            imglist = []                   # List of Patient Images
            for file in files:
                full_path = path+ '\\' + file
                if '.jpg' in full_path and 'red' in full_path:
                    img = cv2.imread(full_path)
                    imglist.append(img[40:-40,:,:]) # cut the top and bottom borders (40 px)
                
            if len(imglist) != 0:
                imgdict[name] = imglist   # Dict[key] = List
                
    #Load Annotated Images
    elif ftype == 'annotation':
        m = '01'                          # Patient ID as Key
        for path,subdir,files in os.walk(fpath):
            for file in files:
                full_path = path+'\\'+file
                
                if m != file[:2]:
                    imgdict[m] = imglist
                    m = file[:2]
                    imglist = []
                    
                imglist.append(cv2.imread(full_path))
            imgdict['07'] = imglist
            
    return imgdict

#imgd = loadimg(dicom_path,'dicom')

clr = np.random.rand(8,3)*255
maskcolor = dict((i+1,clr[i]) for i in range(8))

def load_model2():
    model = load_model('/home/barath/Downloads/home/barath/Downloads/mymodel.h5')
    model.load_weights('/home/barath/Downloads/home/barath/Downloads/best_weights.hdf5')
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
#imgj = loadimg(jpg_path,'jpg')



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
    
# Cut out the relevant portion of image
def cut(img):  # image has dimension (length,breadth,3) 3 = BGR
    
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
#    
#    cv2.imshow('image',cut_img)
#    cv2.imshow('org',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return top, bottom, left, right


# Extract the column corresponding to the scale 
def scale(img,top,bottom,left):
    i2 = img.copy()
    
    # convert BGR image to gray scale
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(img,160,255)
    img[mask==0]=0  # All pixels below 160 becomes 0
    mx = np.max(img[:,:left])  # max pixel value 
    minmax = np.min([200,mx])  
    #print(top,bottom,left,mx,img.shape)
    
    
    j = top - 10
    s = j  
    length = 0
    repetition = 0
    col = []
    ll = []  # stores the pixel length between subsequent points in a column
    row = 0
    length_prev = 0
    for i in range(left):
        j = np.max([0,top - 10])
        s = 0
        length = 0
        repetition = 0
        length_prev = 0
        while j>=0 and j<(bottom -10):
            j += 1
            
            if img[j,i]> minmax:
                #print(' ## ',i,j)
                
                #print('-- ',i,j,img[j,i],minmax,j,s,l,n,col)
                if s==0 and s!=j:
                    #print('a__0',i,j)
                    s = j
                    j += 15
                
                elif length==0 and s!=j  and np.average(img[j-s:j,i])>10 and np.average(img[j-s:j,i])<100:
                    #print('a__1',np.average(img[j-s:j,i]),i,j,s,length)
                    length = (j-s)
                    s = j
                    j += 15
                    
                elif length!=0 and (j-s)>0.9*length and  (j-s)<1.10*length and length>15 and np.average(img[j-s:j,i])>=10 and np.average(img[j-s:j,i])<100:
                    #print('a__2',np.average(img[j-s:j,i]),i,j,s,length)
                    length_prev = length
                    length = j-s
                    
                    
                    repetition += 1                    
                    if repetition >= 3:
                        ll.append((j,length,length_prev))
                        print('* ',i,j,s,s-length_prev)
                        col.append(i)
                        repetition = 0
                        print('')
                        break
                    s = j
                    j += 15
                    
                elif length!= 0 and ((j-s)<=0.9*length or (j-s)>=1.1*length):
                    #print('a__3',i,j,s,length)
                    length = j-s
                    s = j
                    j += 15
                    
        if len(col)!=0:
            break
        
    print(col,ll)
    
    if len(col)!=0:
        cv2.imshow('scale',np.concatenate((np.concatenate((np.zeros([ll[0][0]+30-max(top-10,0),100]),img[max(top-10,0):ll[0][0]+30,col[-1]:col[-1]+1]),axis=1),np.zeros([ll[0][0]+30-max(top-10,0),100])),axis=1))
        #cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\scale.png',cv2.resize(img[0:row+15,col-32:col+32], dsize=(128,2*(row+15))))
    
    i2[:,col] = [0,100,255]
    cv2.imshow('org',i2[max(top-10,0):ll[0][0]+30,:left])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return col[0],ll[0]


# Extract the pixels of the scale and measure the distance if there is a number beside it
def extract(img,col):
   
    i1 = img[max(top-10,0):min(ll[0]+30,bottom),col:col+1]
    mask = cv2.inRange(img,100,255)
    img[mask == 0] = 0
    img = img[max(top-10,0):min(ll[0]+30,bottom),:]
    
    row,column = i1.shape
    peak = []
    m = 0
    s = 0
    for r in range(row):    
        if i1[r,0] >= 140:
            m = r
            s += 1
        elif m!=0:
            peak.append(m-round(s/2))
            s = 0
            m = 0
    
    #print(peak,i1.shape)
    d = 0
    m = ''    # left side(l) or right side(r)
    dist = []
    l0 = 0
    r0 = 0
    for pos in peak:
        lside = img[pos-14:pos+14,col-29:col-1]
        rside = img[pos-14:pos+14,col+1:col+29]
        if lside.shape == (28,28) and rside.shape == (28,28):
            l = num_dict[np.argmax(model.predict(lside.reshape(1,28,28,1)/255))]
        
            r = num_dict[np.argmax(model.predict(rside.reshape(1,28,28,1)/255))]
           
            if l == 0 and m=='':
                m = 'l'
                d = pos
            elif r == 0 and m=='':
                m = 'r'
                d = pos
            elif m=='l' and (l == 1 or l == 2 or l == 0.5):
                #print((pos-d)/(l-l0),m,l,l0,pos)
                dist.append((pos-d)/(l-l0))
                d = pos
                l0 = l
                
            elif m=='r' and (r == 1 or r == 2 or r == 0.5):
                #print((pos-d)/(r-r0),m,r,r0,pos)
                dist.append((pos-d)/(r-r0))
                d = pos
                r0 = r
            
#        cv2.imshow('l',lside)
#        cv2.imshow('r',rside)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    return peak,dist


# resizes image based on distance while maintaining aspect ratio
def img_resize(img,d_avg,final_shape):
    
    m,n = img.shape
    m2,n2 = np.int32(np.round([m*120/d_avg,n*120/d_avg]))
    img = cv2.resize(img,(n2,m2))
    img = cv2.copyMakeBorder(img, int((final_shape-m2)/2), int((final_shape-m2+1)/2), int((final_shape-n2)/2), int((final_shape-n2+1)/2), cv2.BORDER_CONSTANT) 
    
    assert img.shape == (final_shape,final_shape)
    
    return img

# augment the images with their mirror-image 
def flip(fpath):
    for path,subdir,files in os.walk(fpath):
        for file in files:
            full_path = path+ '\\' + file
            flipimg = cv2.flip(cv2.imread(full_path), 1)
            cv2.imwrite('.jpg',flipimg)
    
# cnn model trained on decimal mnist
model = load_model2()

# maps model output to numbers
num_dict = {}
for i in range(10):
    num_dict[i] = i
    num_dict[10+i] = i+0.5

num_dict[20] = -1

path = '/home/barath/Downloads/Thyroid/AI2/01/Image003.jpg'
img = cv2.imread(path)[40:-40,:]


top,bottom,left,right = cut(img)
print(bottom-top,right-left)
#print(img.shape,top,bottom,left)


col,ll = scale(img,top,bottom,left)
# Convert image to gray scale
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
peak,dist = extract(img.copy(),col)
d_avg = np.average(dist)
print(d_avg)

img = img_resize(img[top:bottom,left:right],d_avg,512)
cv2.imshow('i3',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# extract the outline of the annotated image
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

# extract the outline of all images by iteration
def create_map(imgd,imgj):
    for k,v in imgj.items():
        for i in range(len(v)):
            if i%2==0:
                color = [0,255,0]
            else:
                color = [0,255,255]
            create_image(imgd,v[i],color,k,i)

# add the thyroid and nodule images of a single patient
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
    
# add the thyroid and nodule images of all patients by iteration
def overlap(imgA):
    for k,v in imgA.items():
        for i in range(2):
            i1 = v[i*2]
            i2 = v[i*2+1]
            add(i1,i2,k,i)

#create_map(imgd,imgj)
#create_image(imgd,imgj['05'][2],[0,255,255],'05',2,0)
#create_image(imgd,imgj['05'][3],[0,255,0],'05',3)
#create_image(imgd,imgj['05'][4],[0,255,255],'05',4)

#imgA = loadimg(annotated_path,'annotation')
#
#overlap(imgA)
#add(imgA['05'][3],imgA['05'][4],'05',1)

# Convert color -> classes and classes -> one-hot vector
def one_hot(overlap_path,fd=6):   # 1-thyroid, 2-papillary, 3-benign, 4-cyst, 5-solid lesions 0-other
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
            # color -> classes
            for i in range(r):
                for j in range(c):
                    b,g,r = img[i,j]
                        
                    if b<60 and g<60 and r<60:       # black other 0
                        fimg[i,j] = 0
                    
                    elif g>2*r and g > 2*b:          # green thyroid 1
                        fimg[i,j] = 1
                        
                    elif r>2*g and r>2*b:            # red papillary 2
                        fimg[i,j] = 2 
                        
                    elif b>200 and b>3*g and b>3*r:  # blue benign 3 
                        fimg[i,j] = 3
                        
                    elif b>180 and r>180 and g<80:   # violet cyst 4
                        fimg[i,j] = 4
                        
                    elif r>180 and g>180 and b>150:  # pale white solid-lesion 5
                        fimg[i,j] = 5
                        
                    elif r>200 and g>200 and b<60:   # yellow outline 1
                        fimg[i,j] = 1
            
            #color -> one-hot vector
            fimg = to_categorical(fimg,fd)
            fimg_list.append(fimg)
            return fimg_list

#fhot = one_hot(overlap_path,final_dim)


