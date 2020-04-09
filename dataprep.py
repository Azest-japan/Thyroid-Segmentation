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
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import model_from_json, load_model
import tqdm
import csv
import xml.etree.ElementTree as et

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
    imgdict = {}
    if ftype == 'dicom':
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
                imgdict[name] = imglist
                
    elif ftype == 'jpg':
        for path,subdir,files in os.walk(fpath):
            name = os.path.basename(path)
            imglist = []
            for file in files:
                full_path = path+ '\\' + file
                
                if '.jpg' in full_path and 'red' in full_path:
                    img = cv2.imread(full_path)
                    imglist.append(img[40:-40,:,:])
                
            if len(imglist) != 0:
                imgdict[name] = imglist
                
    elif ftype == 'annotation':
        print('annotation')
        m = '01'
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

imgd = loadimg(dicom_path,'dicom')

clr = np.random.rand(8,3)*255
maskcolor = dict((i+1,clr[i]) for i in range(8))

def load_model2():
    model = load_model('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

#imgj = loadimg(jpg_path,'jpg')

def disp(img,imgl=None):
    
    cv2.imshow('img',img)
    if not imgl is None:
        n = len(imgl)
        for i in range(n):
            if not imgl[i] is None:
                cv2.imshow('img'+str(i),imgl[i])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#cv2.imshow('image2',editimg*30)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def gray(img):
    return np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

def orinfo(img0,name):
    img = np.uint8(cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY))

    c = cv2.Canny(img,750,800)
    kernel = np.ones((50,50),np.float32)/500
    dst = cv2.filter2D(c,-1,kernel,borderType = cv2.BORDER_WRAP)
    
    ct = np.int32(np.mean(np.where(dst==np.max(dst)),axis=-1))
    #print(ct,ct[0]/img.shape[0],ct[1]/img.shape[1])
    if ct[0]/img.shape[0]<0.75 or ct[1]/img.shape[1]<0.75: ######
        #print(name,0)
        return 1
    
    dst = np.uint8(255/np.max(dst)*dst)
    #disp(img,[dst])
    t,b,l,r = 0,0,0,0
    s = 0
    while(s<4):
        s = 0
        if  dst[ct[0]-t,ct[1]] >= 120:
            t += 1
        else:
            s += 1
        if  dst.shape[0]>ct[0]+b and dst[ct[0]+b,ct[1]] >= 120:
            b += 1
        else:
            s += 1
        if  dst[ct[0],ct[1]-l] >= 100:
            l += 1
        else:
            s += 1
        if  dst.shape[1]>ct[1]+r and dst[ct[0],ct[1]+r] >= 100:
            r += 1
        else:
            s += 1
    
    #print(t,b,l,r)
    #print(np.min([t,b,l,r]),np.sum([t,b,l,r]),t,b,l,r)
    if np.min([t,b,l,r])<=12 or (np.min([t,b,l,r])<20 and (np.max([t,b,l,r])>45 or np.sum([t,b,l,r])<=115)):
        return 1
    
    sc = img0[ct[0]-t:ct[0]+b,ct[1]-l:ct[1]+r].copy()
    m,n,d = sc.shape
    sh = np.uint8(np.zeros((sc.shape[0],sc.shape[1])))
    
    for i in range(m):
        for j in range(n):
            bl,gr,rd = sc[i,j]
            if (not (rd<100 or gr<100)) and ((2*bl < gr and 2*bl < rd) or (rd<200 and rd<bl-30 and rd<gr-10)):
                sh[i,j] = 255
    #gc.collect()
    
    #disp(sc)
    
    if np.max(sh.sum(axis=0)) > np.max(sh.sum(axis=1)):
        # cd = np.argmax(sh.sum(axis=0))   
        # sc0 = np.uint8(cv2.cvtColor(sc,cv2.COLOR_BGR2GRAY))
        # sc0[sh>0] = 0
        # dmin = 100
        # cmax = int(sc.shape[1]/2)-3
        
        # for i in range(int(sc.shape[1]/2)-4,int(sc.shape[1]/2)+4):
        #     col = len(sc0[:,i])
            
        #     p = 0
        #     n = 0
        #     j = 10
        #     while j<col:
        #         #print(i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin)
                
        #         if (sc0[j,i] > 120) and (sc0[j-2,i] < 36) :
        #             n += 1
        #             if n == 2:
        #                 p = j
        #             elif n==3:
        #                 if j-p < dmin:
        #                     dmin = j-p
        #                     cmax = i
        #                 #print('break ',i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin,' ',j-p)
        #                 break
        #             j += 3
        #         j += 1
                
        #print(name,' Vertical',cd,cmax,abs(cd-cmax))
        #sc[:,cd] = [50,200,50]
        #sc[:,cmax] = [200,100,50]
        #disp(sc)
        return 0 #,int(cd),int(cmax)
    
    else:
        cd = np.argmax(sh.sum(axis=1))
        if cd==0:
            print(name,t,b,l,r, ct[0]/img.shape[0], ct[1]/img.shape[1],np.sum([t,b,l,r]),np.min([t,b,l,r]),ct)
            #disp(img,[dst,sc])
        #sc[cd,:] = [50,200,50]
        #disp(sc)
        #print(name,' Horizontal')
        return 1 #,int(cd),-1

def write_csv(wpath,data):
    with open(wpath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def write_json(wpath,data):
    with open(wpath, 'w') as outfile:
        json.dump(data, outfile)
    
    
def read_json(rpath):
    with open(rpath) as json_file:
        data = json.load(json_file)
        return data

def append_json(rpath,name,val,wpath=None):
    if wpath == None:
        wpath = rpath
    data = read_json(rpath)
    data[name] = val
    write_json(wpath,data)
    
    return data

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

    mono_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # np.sum(img, axis=2)
    #bin_img = np.sign(np.where((mono_img>140)&(mono_img<170), 0, mono_img))
    
    row_activate = np.zeros(mono_img.shape[0])
    col_activate = np.zeros(mono_img.shape[1])
    
    for row in range(mono_img.shape[0]):
        row_activate[row] = len(np.unique(mono_img[row]))
    for col in range(mono_img.shape[1]):
        col_activate[col] = len(np.unique(mono_img[:,col]))
    
    judge_len = 30
    judge_len_2 = 20
    min_unique_1 = 30
    min_unique_2 = 35
    
    top = 0
    bottom = mono_img.shape[0]-1
    for t in range(mono_img.shape[0]-judge_len):
        if all(row_activate[t:t+judge_len] >= min_unique_1):
            top = t
            for b in range(top+100, mono_img.shape[0]-judge_len_2):
                if all(row_activate[b:b+judge_len_2] < min_unique_2):
                    bottom = b
                    if b < top + 0.75*(mono_img.shape[0] - top):
                        bottom = int(top+ 0.75*(mono_img.shape[0] - top))
                        
                    break
            break

    judge_len = 30                             
    min_unique = 30
    left = 0
    right = mono_img.shape[1]-1
    for l in range(mono_img.shape[1]-judge_len):
        if all(col_activate[l:l+judge_len] >= min_unique):
            left = l
            for r in range(left + int(0.6*(mono_img.shape[1] - left)), mono_img.shape[1]):
                
                if all(col_activate[r:r+10] < min_unique) or len(np.where(mono_img[top:bottom,r]<6)[0])>bottom-top-60:
                    #print(r-10,r+10,col_activate[r-10:r+10])
                    #disp(img[:,r-10:r+10])
                    break
            right = r
            break


    return top, bottom, left, right

def cutpr(img):
    t,b,l,r = cut(img)
    if b > t+100 and r > l+100:
        img = img[t:b,l:r]
    return gray(img)


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
    
    i2[:,col] = [100,50,255]
    cv2.imshow('org',i2[max(top-10,0):ll[0][0]+30,:left])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return col[0],ll[0]


# Extract the pixels of the scale and measure the distance if there is a number beside it
def extract(img,col,top,bottom,ll):
   # cnn model trained on decimal mnist
    model = load_model2()

     # maps model output to numbers
    num_dict = {}
    for i in range(10):
        num_dict[i] = i
        num_dict[10+i] = i+0.5

    num_dict[20] = -1
    
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

def readxml(fxml,imgdict):
    colordict = {'Thyroid':(100,200,0),'Trachea':(100,200,200),'Nodule':(30,60,160),'Benign':(30,60,160),'Papillary':(200,30,220), 'Malignant':(200,30,220)}
    labeldict = {'Thyroid':0,'Nodule':1,'Benign':2,'Malignant':3}
    tree = et.parse(fxml)
    root = tree.getroot()
    d = {}
    
    for ann in root.iter('image'):
        d = {}
        #print(ann.attrib['name'])
        for el in ann.findall('polygon'):
            lab = el.attrib['label']
            points = el.attrib['points'].split(';')
            p = [(float(i.split(',')[0]),float(i.split(',')[1])) for i in points]
            if not lab in d.keys():
                d[lab] = []
            d[lab].append(p)
            
        i3 = imgdict[ann.attrib['name']]
        i2 = np.zeros(i3.shape)
        for k in ['Thyroid','Nodule','Benign','Malignant','Papillary']:
            if k in d.keys():
                for v in d[k]:
                    pts = np.array(v, np.int32)
                    mask = colordict[k]
                    
                    #cv2.polylines(imgdict[ann.attrib['name']],[pts],True,color = mask)
                    cv2.fillPoly(i2, [pts], color=mask)
        
        
        i3 = np.hstack((i3,cv2.add(7*np.uint8(i3/12),5*np.uint8(i2/12))))
        #disp(imgdict[ann.attrib['name']])
        cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\Ito\\Comp\\Papillary\\a_'+ann.attrib['name'], i3)
        


# resizes image based on distance while maintaining aspect ratio
def img_resize(img,m0=320,n0=512):

    m,n = img.shape
    #print(m,n)
    
    if m<m0-32 and n>n0+32:
        i0 = cv2.copyMakeBorder(img, int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        i1 = i0[:,0:n0]
        i3 = i0[:,n-n0:n]
        i4 = cv2.resize(img,(n0,int(m*n0/n)))
        m = int(m*n0/n)
        i4 = cv2.copyMakeBorder(i4,int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        return [i1,i3,i4]
    
    elif m<m0-32 and n>=n0:
        i0 = cv2.copyMakeBorder(img, int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        i2 = i0[:,int((n-n0)/2):n0+int((n-n0)/2)]
        i4 = cv2.resize(img,(n0,int(m*n0/n)))
        m = int(m*n0/n)
        i4 = cv2.copyMakeBorder(i4,int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        return [i2,i4]
    
    elif m>=m0-32 and m<=m0 and n>=n0 and n<n0+32:
        
        i4 = cv2.resize(img,(n0,int(m*n0/n)))
        m = int(m*n0/n)
        i4 = cv2.copyMakeBorder(i4,int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        return [i4]
    
    elif m>=m0-32 and m<=m0 and n>=n0:
        
        i0 = cv2.copyMakeBorder(img, int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        i2 = i0[:,int((n-n0)/2):n0+int((n-n0)/2)]
        i4 = cv2.resize(img,(n0,int(m*n0/n)))
        m = int(m*n0/n)
        i4 = cv2.copyMakeBorder(i4,int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        return [i2,i4]
        
    elif m>=m0 and n>=n0:
        if m*n0<n*m0:
            i4 = cv2.resize(img,(n0,int(m*n0/n)))
            m = int(m*n0/n)
            i4 = cv2.copyMakeBorder(i4,int((m0-m)/2), int((m0-m+1)/2),0,0,cv2.BORDER_CONSTANT)
        else:
            i4 = cv2.resize(img,(int(m0*n/m),m0))
            n = int(m0*n/m)
            i4 = cv2.copyMakeBorder(i4,0,0,int((n0-n)/2), int((n0-n+1)/2),cv2.BORDER_CONSTANT)
        return [i4]
    
    elif m<=m0 and n<=n0:
        i4 = cv2.copyMakeBorder(img,int((m0-m)/2), int((m0-m+1)/2), int((n0-n)/2), int((n0-n+1)/2), cv2.BORDER_CONSTANT)
        return [i4]
    
    elif m>m0 and n<n0:
        i0 = cv2.copyMakeBorder(img,0,0, int((n0-n)/2), int((n0-n+1)/2),cv2.BORDER_CONSTANT)
        i1 = i0[0:m0,:]
        #i2 = i0[int((m-m0)/2):m0+int((m-m0)/2),:]
        i3 = i0[m-m0:m,:]
        i4 = cv2.resize(img,(int(n*m0/m),m0))
        n = int(n*m0/m)
        i4 = cv2.copyMakeBorder(i4,0,0,int((n0-n)/2), int((n0-n+1)/2),cv2.BORDER_CONSTANT)
        return [i1,i3,i4]
    
    else: 
        return 0

# augment the images with their mirror-image 
def flip(img):
    return cv2.flip(img, 1)
    

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
def iadd(i1,i2,k,index):
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
            iadd(i1,i2,k,i)

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




    
