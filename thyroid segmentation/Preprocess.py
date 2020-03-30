# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:14:37 2019

@author: AZEST-2019-07
"""

import cv2
import numpy as np
import sys
import os
import cv2
import json
import pydicom
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import model_from_json, load_model
import tqdm
import xml.etree.ElementTree as et

base = 'D:\\Ito data\\'

dicom_path = base + 'AI'
jpg_path = base + 'AI2'
annotated_path = base + 'annotated'
overlap_path = base + 'overlap'
final_dim = 6

sample = None
imgdata = None

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

def readxml(fxml,imgdict):
    colordict = {'Thyroid':(100,200,0),'Trachea':(100,200,200),'Nodule':(30,60,160),'Artery':(200,100,0)}
    tree = et.parse(fxml)
    root = tree.getroot()
    d = {}
    
    for ann in root.iter('image'):
        d = {}
        print(ann.attrib['name'])
        for el in ann.findall('polygon'):
            lab = el.attrib['label']
            points = el.attrib['points'].split(';')
            p = [(float(i.split(',')[0]),float(i.split(',')[1])) for i in points]
            if not lab in d.keys():
                d[lab] = []
            d[lab].append(p)

        for k in ['Thyroid','Nodule','Artery','Trachea']:
            if k in d.keys():
                for v in d[k]:
                    pts = np.array(v, np.int32)
                    mask = colordict[k]
                    
                    #cv2.polylines(imgdict[ann.attrib['name']],[pts],True,color = mask)
                    cv2.fillPoly(imgdict[ann.attrib['name']], [pts], color=mask)
        #disp(imgdict[ann.attrib['name']])
        #cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\Ito\\Patient 1\\annotated'+ann.attrib['name'], imgdict[ann.attrib['name']])
    return imgdict      


def load_model2():
    model = load_model('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\mymodel.h5')
    model.load_weights('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles\\best_weights.hdf5')
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

# display any image
def disp(img,imgl=None):
    
    cv2.imshow('img',img)
    if not imgl is None:
        n = len(imgl)
        for i in range(n):
            if not imgl[i] is None:
                cv2.imshow('img'+str(i),imgl[i])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# returns gray scale of the image
def gray(img):
    return np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    
# calculate distance between points
def calcdist(p1,p2):
    return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))

def mouse(img):
    points = []
    def mousepoint(event,x,y,flags,param): 
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y,img[y,x],'  --------')
            #cv2.circle(img,(x,y),1,(255,255,255),-1)
            #points.append((x,y))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',mousepoint)
    
    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

# Extract the pixels of the scale and measure the distance if there is a number beside it
def getdist(img,col,top,bottom,flist):
    
    img = img.copy()
   # cnn model trained on decimal mnist
    model = load_model2()

     # maps model output to numbers
    num_dict = {}
    for i in range(10):
        num_dict[i] = i
        num_dict[10+i] = i+0.5

    num_dict[20] = -1
    
    colslice = img[max(top-10,0):min(flist[0]+30,bottom),col:col+1]
    mask = cv2.inRange(img,100,255)
    img[mask == 0] = 0
    img = img[max(top-10,0):min(flist[0]+30,bottom),:]
    print(colslice.shape)
    row,column = colslice.shape
    peak = []
    pr = 0  #row which has the point
    corr = 0
    for r in range(row):    
        if colslice[r,0] >= 140:
            pr = r
            corr += 1
        elif pr!=0:
            peak.append(pr-round(corr/2))
            corr = 0
            pr = 0
    
    #print(peak,i1.shape)
    d = 0
    lor = ''    # left side(l) or right side(r)
    dist = []
    l0 = 0
    r0 = 0
    
    for pos in peak:
        lside = img[pos-14:pos+14,col-29:col-1]
        rside = img[pos-14:pos+14,col+1:col+29]
        if lside.shape == (28,28) and rside.shape == (28,28):
            l = num_dict[np.argmax(model.predict(lside.reshape(1,28,28,1)/255))]
        
            r = num_dict[np.argmax(model.predict(rside.reshape(1,28,28,1)/255))]
           
            if l == 0 and lor=='':
                lor = 'l'
                d = pos
            elif r == 0 and lor=='':
                lor = 'r'
                d = pos
            elif lor=='l' and (l == 1 or l == 2 or l == 0.5):
                #print((pos-d)/(l-l0),m,l,l0,pos)
                dist.append((pos-d)/(l-l0))
                d = pos
                l0 = l
            elif lor=='r' and (r == 1 or r == 2 or r == 0.5):
                #print((pos-d)/(r-r0),m,r,r0,pos)
                dist.append((pos-d)/(r-r0))
                d = pos
                r0 = r
       
#        cv2.imshow('l',lside)
#        cv2.imshow('r',rside)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    return peak,np.average(dist)


# resizes image based on distance while maintaining aspect ratio
def img_resize(img,top,bottom,left,right,d_avg):
#    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = img[top:bottom,left:right]
    m,n = img.shape
    
    m2,n2 = np.int32(np.round([m*120/d_avg,n*120/d_avg]))
    print(m,n,d_avg,final_shape,m2,n2)
    img = cv2.resize(img,(n2,m2))
    img = cv2.copyMakeBorder(img, int((448-m2)/2), int((448-m2+1)/2), int((512-n2)/2), int((512-n2+1)/2), cv2.BORDER_CONSTANT) 

    assert img.shape == (448,512)
    
    return img

# find quality of color image
def quality(img):
    img = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    ddepth = cv2.CV_8U
    laplacian = cv2.Laplacian(img, ddepth, ksize=3) 
    #disp(laplacian)
    return laplacian.var()

# enhance the quality of cut image using CLAHE method
def enhanceQ(img):
    q = quality(img)
    cl = 5000/max(2000,q) - 0.9
    clahe = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8,8))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img[:,:,1] = clahe.apply(img[:,:,1])
    img[:,:,2] = clahe.apply(img[:,:,2])
    return img

# plot histogram
def histo(img):
    img = np.uint8(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #plt.hist(img.ravel(),256,[0,256])
    #plt.show()
    return hist

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


def orinfo(img0):
    img = np.uint8(cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY))

    c = cv2.Canny(img,750,800)
    
    kernel = np.ones((50,50),np.float32)/500
    dst = cv2.filter2D(c,-1,kernel,borderType = cv2.BORDER_WRAP)
    
    ct = np.int32(np.mean(np.where(dst==np.max(dst)),axis=-1))
    #print(ct,ct[0]/img.shape[0],ct[1]/img.shape[1])
    if ct[0]/img.shape[0]<0.75 or ct[1]/img.shape[1]<0.75: ######
        return -1,-1,-1
    
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
        return -1,-1,-1
    
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
        cd = np.argmax(sh.sum(axis=0))   
        sc0 = np.uint8(cv2.cvtColor(sc,cv2.COLOR_BGR2GRAY))
        sc0[sh>0] = 0
        dmin = 100
        cmax = int(sc.shape[1]/2)-3
        
        for i in range(int(sc.shape[1]/2)-4,int(sc.shape[1]/2)+4):
            col = len(sc0[:,i])
            
            p = 0
            n = 0
            j = 10
            while j<col:
                #print(i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin)
                
                if (sc0[j,i] > 120) and (sc0[j-2,i] < 36) :
                    n += 1
                    if n == 2:
                        p = j
                    elif n==3:
                        if j-p < dmin:
                            dmin = j-p
                            cmax = i
                        #print('break ',i,' ',j,' ',sc0[j,i],' ',sc0[j-2,i],' ',p,' ',n,' ',cmax,' ',dmin,' ',j-p)
                        break
                    j += 3
                j += 1
                
        #print('vertical',cd,cmax,abs(cd-cmax))
        sc[:,cd] = [50,200,50]
        sc[:,cmax] = [200,100,50]
        #disp(sc)
        return 0,int(cd),int(cmax)
    
    else:
        cd = np.argmax(sh.sum(axis=1))
        if cd==0:
            print(t,b,l,r, ct[0]/img.shape[0], ct[1]/img.shape[1],np.sum([t,b,l,r]),np.min([t,b,l,r]),ct)
            disp(img,[dst,sc])
        sc[cd,:] = [50,200,50]
        #disp(sc)
        #print('Horizontal')
        return 1,int(cd),-1
    
    

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
    
    i1 = i1[0:r,0:c,:]
    i2 = i2[0:r,0:c,:]
    
    i3 = cv2.add(i1,i2)
    cv2.imwrite('D:\\Ito data\\overlap\\'+k+str(index)+'.png',i3)

def find_contours(img):
    img = cv2.imread('C:\\Users\\AZEST-2019-07\\Desktop\\Ito\\Patient 1\\annotatedImage009.jpg')
    i2 = np.zeros((img.shape[0],img.shape[1]))
    m = cv2.inRange(img,np.array([20,50,150]),np.array([40,70,170]),img)
    edged = cv2.Canny(m, 30, 200) 
    
    
    contours, hierarchy = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    n = len(contours)
    print("Number of Contours found = " + str(n)) 
    disp(edged)
    
    cl = []
    for i in range(n):
        i2 = np.zeros((img.shape[0],img.shape[1]))
        p,q,r = contours[i].shape
        contours[i] = contours[i].reshape((p,r))
        
        for j in range(p):
            i2[contours[i][j][1],contours[i][j][0]] = 255
        disp(i2)
        cl.append(i2)
    return cl
    

# augment the images with their mirror-image 
def flip(fpath,spath):
    for path,subdir,files in os.walk(fpath):
        for file in files:
            full_path = path+ '\\' + file
            flipimg = cv2.flip(cv2.imread(full_path), 1)
            cv2.imwrite(spath+'\\'+file+'.jpg',flipimg)


# mix the annotated output and input image
def fusion(img,img2,path,name):
    img3 = cv2.add(7*np.uint8(img/12),5*np.uint8(img2/12))
    img3[img2==0] = img[img2==0]
    cv2.imwrite(path+'\\'+name,img3)
    return img3


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


def maxdist(img):
   
    disp(img)
    
    img[np.where(np.sum(img>[20,20,200],axis=-1)!=3)] =  0
    disp(img)

    edges = cv2.Canny(img,100,255)
    x,y = np.where(edges>0)
    disp(edges)
    print(x.shape)

    num = x.shape[0]
    p = {}
    px = -1   # previous x
    x2 = []
    y2 = []
    for i in range(num):
        if px != x[i]:
            if px != -1:
                p[px].sort()
                m = p[px][0]
                yc = p[px][1:].copy()
                for j in yc:
                    if j>m-10 and j<m+10:
                        p[px].remove(j)
                    else:
                        m = j
            p[x[i]] = []
            px = x[i]
        p[x[i]].append(y[i])
        
    for k,v in p.items():
        for val in v:
            x2.append(k)
            y2.append(val)
    
    x2 = np.array(x2)
    y2 = np.array(y2)
    print(x2.shape)
    m = np.uint8(np.zeros(img.shape))
    m1 = []
    ln = x2.shape[0]
    for i in range(ln):
        cv2.circle(m, (y2[i],x2[i]), 1, (255,255,255), 0) 
        m1.append(np.array([x2[i],y2[i]]))
    m1 = np.array(m1)
    
    disp(m)
    
    h = cv2.convexHull(m1, False)
    h = h.reshape(h.shape[0],2)
    print(h.shape)
    h2 = np.int32(np.zeros(h.shape))
    h2[:,0] = h[:,1]
    h2[:,1] = h[:,0]
    cv2.drawContours(m,[h2],-1,(0,180,200), 2,1)
    
    mid = np.int32(np.average(h,axis=0))
    cv2.circle(m,(mid[1],mid[0]),2,(100,255,180),-1)
    
    disp(m)
    
    # largest distance from a set of points
    def ldist(h):
        n,d = h.shape
        maxd = 0  #largest distance
        maxp = None
        xL,yL = mid
        dp = None   
        mcd = 0   # maximum distance through the center
        for i in range(n):
            x1,y1 = h[i]
            xL,yL = mid
            x,y = (2*xL-x1),(2*yL-y1)
            cnt = 0
            
            while True:
                pos = cv2.pointPolygonTest(h,(x,y),True)
                cnt += 1
                #print(pos,x,y,xL,yL,x1,y1,' ',0)
                if abs(pos)<3 or cnt>10:
                    #print(pos,x,y,xL,yL,x1,y1,' ',1)
                    break
                
                elif pos<0 and abs(pos)>=3:
                    x,y = np.int32(((x+xL)/2,(y+yL)/2))
                    #print(pos,x,y,xL,yL,x1,y1,' ',2)
                    
                elif pos>=0 and abs(pos)>=3:
                    cx,cy = x,y
                    x,y = (2*x-xL),(2*y-yL)
                    xL,yL = cx,cy
                    
                    #print(pos,x,y,xL,yL,x1,y1,' ',3)
               
            cd = calcdist(h[i],[x,y])
            if mcd < cd:
                mcd = cd
                dp = [h[i],np.int32([x,y])]
            
            for j in range(i+1,n):
                d = calcdist(h[i],h[j])
                if maxd < d:
                    maxd = d
                    maxp = [h[i],h[j]]
            #print('')
        return maxd,maxp,mcd,dp
    
    maxd,maxp,mcd,dp = ldist(h)
    print(maxd,maxp)
    print(mcd,dp)
    
    cv2.line(m,(maxp[0][1],maxp[0][0]),(maxp[1][1],maxp[1][0]),(100,200,50),1)
    cv2.line(m,(dp[0][1],dp[0][0]),(dp[1][1],dp[1][0]),(200,100,100),1)
    disp(m)




