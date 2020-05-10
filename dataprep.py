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
from tensorflow.keras.models import model_from_json, load_model
import tqdm
import csv
import xml.etree.ElementTree as et
import matplotlib.pyplot as plt
import segmentation_models as sm


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

    
def displt(img,imgl=None):
    print('img')
    fig = plt.figure()
    if np.sum(img.shape)>1000:
        fig = plt.gcf()
        fig.set_size_inches(18,10)
        print(1)
    else:
        fig.set_size_inches(6,4)
        print(0)
    ax1 = fig.add_subplot(111)
    ax1.imshow(img)
    plt.show()
    if not imgl is None:
        n = len(imgl)
        for i in range(n):
            if not imgl[i] is None:
                print('img'+str(i))
                plt.imshow(imgl[i])
                plt.show()

#cv2.imshow('image2',editimg*30)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

def gray(img):
    return np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

# find quality of color image
def quality(img):
    #img = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    ddepth = cv2.CV_8U
    laplacian = cv2.Laplacian(img, ddepth, ksize=3) 
    #disp(laplacian)
    return laplacian.var()

# enhance the quality of cut image using CLAHE method
def enhanceQ(img,q):
    
    cl = 1
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8,8))
    img = clahe.apply(img)
    #img[:,:,1] = clahe.apply(img[:,:,1])
    #img[:,:,2] = clahe.apply(img[:,:,2])
    
    return img


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
       
        return 0 
    
    else:
        cd = np.argmax(sh.sum(axis=1))
        if cd==0:
            print(name,t,b,l,r, ct[0]/img.shape[0], ct[1]/img.shape[1],np.sum([t,b,l,r]),np.min([t,b,l,r]),ct)
            #disp(img,[dst,sc])
       
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
            for r in reversed(range(left + int(0.6*(mono_img.shape[1] - left)), mono_img.shape[1])):
                #print(r-judge_len,r,col_activate[r-judge_len:r])
                if all(col_activate[r-judge_len:r] >= min_unique):
                    
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


def readxml(fxml):
    imgd = {}
    colordict = {'Thyroid':80,'Trachea':0,'Nodule':150,'Benign':150,'Papillary':220, 'Malignant':220}
    labeldict = {'Thyroid':1,'Nodule':2,'Benign':2,'Malignant':7,'Papillary':7}
    hotdict = {'Thyroid':1,'Nodule':2,'Benign':2,'Malignant':3,'Papillary':3}
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
            if not lab.lower() in d.keys():
                d[lab.lower()] = []
            d[lab.lower()].append(p)
        
        no = int(ann.attrib['name'].split('_')[0])
        
        if no<201:
            i3 = cv2.imread('/test/Ito/Selected1/' + ann.attrib['name'])
        else:
            i3 = cv2.imread('/test/Ito/SelectedP/' + ann.attrib['name'])
        if type(i3) == type(None):
            i3 = cv2.imread('/test/Ito/test/' + ann.attrib['name'])
        
        t,b,l,r = cut(i3)
        if b < t+100 or r < l+100:
            print('DaMe')
      
        
        i2c = np.uint8(np.zeros(i3.shape[0:2]))
        #temp = [0,0]
        
        for k in ['Thyroid','Nodule','Benign','Malignant','Papillary']:
            if k.lower() in d.keys():
                for v in d[k.lower()]:
                    '''
                    mask = hotdict[k] - 2 
                    if mask>=0:
                        temp[mask] = 1
                    '''
                    
                    i2 = np.uint8(np.zeros(i3.shape[0:2]))
                    pts = np.int32(np.round(v))
                    mask = colordict[k]
                    #cv2.polylines(i2,[pts],True,color = mask)
                    cv2.fillPoly(i2, [pts], color=mask)
                    i2c[i2!=0] += labeldict[k]
                    

        i2c = i2c[t:b,l:r]
        
        imgd[ann.attrib['name']] = i2c 
    return imgd


def onehot(img):
    i2 = np.uint8(np.zeros((320,512)+(4,)))
    i2[img==1] = [0,1,0,0]
    
    i2[(img<7)*(img>=2)] = [0,1,1,0]
    i2[img==7] = [0,0,0,1]
    i2[img>=8] = [0,1,0,1]
    i2[img==0] = [1,0,0,0]
    
    return i2

def decode(himg):
    himg = himg.copy()
    himg[himg<0.45] = 0
    himg[:,:,1][(himg[:,:,0]<0.5)*(himg[:,:,1]>0.45)] = 1
    himg[:,:,1][(himg[:,:,0]>0.5)*(himg[:,:,1]>0.75)] = 1
    himg[:,:,1][(himg[:,:,0]>0.75)*(himg[:,:,1]<0.75)] = 0
    himg[:,:,2][himg[:,:,2]<0.5] = 0
    himg[:,:,3][himg[:,:,3]<0.5] = 0
    
    yimg = np.uint8(np.ones((320,512,3)))
    #print(himg.shape)
    i=1
    yimg[himg[:,:,i-1]!=0] = himg[:,:,i-1][himg[:,:,i-1]!=0].reshape(-1,1)*np.array([8,8,8]).reshape(1,-1)
    yimg[himg[:,:,i]!=0] = himg[:,:,i][himg[:,:,i]!=0].reshape(-1,1)*np.array([50,200,50]).reshape(1,-1) + (1-himg[:,:,i])[himg[:,:,i]!=0].reshape(-1,1)*np.array([50,100,50]).reshape(1,-1)
    yimg[himg[:,:,i+1]!=0] = himg[:,:,i+1][himg[:,:,i+1]!=0].reshape(-1,1)*np.array([50,50,200]).reshape(1,-1) + (1-himg[:,:,i+1])[himg[:,:,i+1]!=0].reshape(-1,1)*np.array([50,200,50]).reshape(1,-1)
    yimg[himg[:,:,i+2]!=0] = himg[:,:,i+2][himg[:,:,i+2]!=0].reshape(-1,1)*np.array([200,50,50]).reshape(1,-1) + (1-himg[:,:,i+2])[himg[:,:,i+2]!=0].reshape(-1,1)*np.array([50,200,50]).reshape(1,-1)
    
    return yimg


def disp_decode(x,y,lb,i):
    
    print(lb[i])
    displt(x[i])
    displt(decode(y[i]))

# mix the annotated output and input image
def fusion(img,img2):
    img3 = cv2.add(7*np.uint8(img/12),5*np.uint8(img2/12))
    img2[img2<10] = 0
    img3[img2==0] = img[img2==0]
    return img3
    
def save_res(x,y,yt,lb):
    n = x.shape[0]
    
    for i in range(n):
        ximg = np.uint8(np.zeros((320,512,3)))
        ximg[:,:,0] = np.uint8(x[i].reshape((320,512))*255)
        ximg[:,:,1] = ximg[:,:,0]
        ximg[:,:,2] = ximg[:,:,0]
        
        i3 = np.hstack((ximg,fusion(ximg,decode(y[i])),fusion(ximg,decode(yt[i]))))
        cv2.imwrite('/test/Ito/result/'+str(i)+'.jpg',i3)
        iou_m = sm.metrics.IOUScore()(yt[i],y[i])
        iou_0 = sm.metrics.IOUScore()(yt[i,:,:,0],y[i,:,:,0])
        iou_1 = sm.metrics.IOUScore()(yt[i,:,:,1],y[i,:,:,1])
        iou_2 = sm.metrics.IOUScore()(yt[i,:,:,2],y[i,:,:,2])
        iou_3 = sm.metrics.IOUScore()(yt[i,:,:,3],y[i,:,:,3])
        th = findthresh(yt[i],y[i])
        print(i,lb[-500+i],th)
        
def img_test(ximg,yp,yt):
    
    None
        
# resizes image based on distance while maintaining aspect ratio 
def img_resize(img,m0=320,n0=512):

    m,n = img.shape[0:2]
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
    
    # np.flip(img,len(img.shape)-2)
    return cv2.flip(img, 1)

    

def findthresh(yt,yp):
    
    m = 0
    iou = []
    th = np.zeros(yt.shape[-1])
    for i in range(yt.shape[-1]):
        iou = []
        for k in np.arange(0.2,0.92,0.02):
            y = yp[:,:,:,i].copy()
            y[y<k] = 0
            y[y>=k] = 1
            intersection = np.sum(yt[:,:,:,i]*y)
            union = np.sum(yt[:,:,:,i])+np.sum(y)-intersection
            #print(k,i/u,yt,y)
            jl = intersection/union
            iou.append(jl)
            if jl > m:
                m = jl
                th[i] = k
    return th
        #print(th)
        #x = np.arange(0.2,0.92,0.02)
        #plt.plot(x,np.array(iou))
        #plt.show()
