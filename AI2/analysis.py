import numpy as np
import argparse
#import imutils
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import csv
import json

from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import PolynomialFeatures

def mouse(img):
    points = []
    def mousepoint(event,x,y,flags,param): 
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y,img[y,x],'  --------')
            points.append((x,y))
            
    def ctrl(img,points):
        while(1):
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',mousepoint)
            
            while(1):
                cv2.imshow('image',img)
                if cv2.waitKey(0) == 27:
                    break
            cv2.destroyAllWindows()
            break
    
    ctrl(img,points)

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

def load_csv(file):
    return pd.read_csv(file,delimiter=',',names=['f','x','y','w','h']).drop([0],axis=0)

def save_csv(file,df):
    df.to_csv(file,index=False)
    
def imgplot(img):
    img = img.copy()
    fig = plt.figure()
    
    if np.sum(img.shape)>1000:
        fig = plt.gcf()
        fig.set_size_inches(24,12)
        
    else:
        fig.set_size_inches(6,4)
        
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)
    plt.show()


# display picture in openCV
def dispcv(img,imgl=None):
    
    cv2.imshow('img',img)
    if not imgl is None:
        n = len(imgl)
        for i in range(n):
            if not imgl[i] is None:
                cv2.imshow('img'+str(i),imgl[i])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gt(df,conf=0.3):
    #df = pd.read_csv('/test/Downloads/MOT16/train/MOT16-04/gt/gt.txt',delimiter=',',names=['f','id','x','y','w','h','c1','c2','c3'])
    color = (100,180,255)
    for i in range(1,100):
        f = df[df['f']==i]
        f = f[f['c']>conf]
        p = 0
        n = i
        while n>=1:
            n = n/10
            p+=1
        sn = ''
        for _ in range(6-p):
            sn = sn + '0'
        sn = sn+str(i)
        img = cv2.imread('C:\\Users\\81807\\Documents\\RD\\MOT16\\train\\MOT16-04\\img1\\'+sn+'.jpg')
        for j in range(f.shape[0]):
            x,y,w,h = list(f.iloc[j].loc[['x','y','w','h']])
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)),color, 2)
            #cv2.putText(img, str(int(f.iloc[j].loc['id'])),(int(x+w), int(y+h+10)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (20,240,60), 2)
        cv2.imwrite('C:\\Users\\81807\\Downloads\\motfcnn\\'+str(i)+'.jpg',img)

# calculate iou of bbox
def bb_iou(boxA, boxB):
    
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
    
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
    # compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
    # compute the intersection over union by taking the intersection
	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

# calculate iou of bbox
def bb_iosb(boxA, boxB):
    
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
    
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
    # compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
    # compute the intersection over union by taking the intersection
	iosb = interArea / np.min((boxAArea,boxBArea))

	return iosb

# a - array of bbox
# bbox dimensions
def plot_d(a,hw):
    k = 1
    if hw == 'h':
        k = 2
    plt.plot(a[:,0],a[:,k+2] - a[:,k])
    for index,i in enumerate(a[:,0]):
        plt.text(i,a[index,k+2] - a[index,k],str(i))

# center
def plot_c(a):
    plt.plot((a[:,1]+a[:,3])/2,600-(a[:,2]+a[:,4])/2)
    for index,i in enumerate(a[:,0]):
        plt.text((a[index,1]+a[index,3])/2,600-(a[index,2]+a[index,4])/2,str(i))

# aspect ratio
def plot_a(a):
    plt.plot(a[:,0],(a[:,4]-a[:,2])/(a[:,3]-a[:,1]))
    for index,i in enumerate(a[:,0]):
        plt.text(i,(a[index,4]-a[index,2])/(a[index,3]-a[index,1]),str(i))
        
        

def curvefit(y,doplot = False,method = None):
    #y = [i[0] for i in df[(df[1]==319) & (df[0]>700)][5]]
    if len(y)<=3:
        print('cant perform sine fit')
        return 
    
    if method == 'sine':
        vy = np.array([y[i+1]-y[i] for i in range(len(y)-1)])
        t = np.arange(vy.shape[0])
        gmean = np.mean(vy)
        gamp = (vy[np.argsort(vy)[-5]] - vy[np.argsort(vy)[5]])/2
        gphase = 0

        merror = 512
        p = []
        for gfreq in np.arange(0,1.5,0.03):
            optimize_func = lambda x: (x[0]*np.sin(x[1]*t+x[2]) + x[3] - vy)**2
            est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [gamp, gfreq, gphase, gmean])[0]

            vp = est_amp*np.sin(est_freq*t+est_phase) + est_mean
            er = np.sum(np.square(vp-vy))
            if er < merror:
                merror = er
                p = (est_amp, est_freq, est_phase, est_mean)

        vp = p[0]*np.sin(p[1]*t+p[2]) + p[3]

        if doplot == True:
            plt.plot(t,vy)
            plt.plot(t,vp)

        t = len(y)-1
        return p[0]*np.sin(p[1]*t+p[2]) + p[3]
    
    elif method == 'acc':
        ay = np.array([y[i+2]+y[i]-2*y[i+1] for i in range(len(y)-2)])
        t = np.arange(ay.shape[0]).reshape(-1,1)
        reg = LinearRegression()
        reg.fit(t,ay.reshape(-1,1))
        
        return reg.predict(np.array([len(ay)]).reshape(-1,1))[0]
    
    return np.array([y[i+1]-y[i] for i in range(len(y)-1)]).mean()
        

def angle(pos):
    
    reg = LinearRegression().fit(pos[:,0].reshape(-1,1), pos[:,1].reshape(-1,1))
    return float(reg.coef_)

def pair_angle(pos1,pos2):
    d1 = pos1[1] - pos1[0]
    d2 = pos2[1] - pos2[0]
    dot = np.dot((d1,d2))/(np.sum(np.square(d1)) + np.sum(np.square(d2)))
    angle = np.arccos(dot)*180/np.pi  # angle in degrees
    
    return dot,angle

def eudist(pos1,pos2):
    (x1,y1) = pos1
    (x2,y2) = pos2
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))


def reshapeimg(image):
    (H, W) = image.shape[:2]
    
    if W/H > 10/6:
        H = int(1000*H/W + 0.5)
        W = 1000
    else:
        W = int(600*W/H + 0.5)
        H = 600
        
    image = cv2.resize(image,(W,H),interpolation = cv2.INTER_AREA)
    return image