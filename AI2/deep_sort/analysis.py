import numpy as np
import argparse
#import imutils
import cv2
import os
import sys
import timeit
import matplotlib.pyplot as plt
if '/test/Ito/Code' not in sys.path:
    sys.path.append('/test/Ito/Code')
from dataprep import imgplot,displt,calcdist
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def gt(df):
    #df = pd.read_csv('/test/Downloads/MOT16/train/MOT16-04/gt/gt.txt',delimiter=',',names=['f','id','x','y','w','h','c1','c2','c3'])
    color = (100,180,255)
    for i in range(1,10):
        f = df[df['f']==i]
        f = f[f['c2']==1]
        p = 0
        n = i
        while n>=1:
            n = n/10
            p+=1
        sn = ''
        for _ in range(6-p):
            sn = sn + '0'
        sn = sn+str(i)
        img = cv2.imread('/test/RD/MOT16/train/MOT16-04/img1/'+sn+'.jpg')
        for j in range(f.shape[0]):
            x,y,w,h = list(f.iloc[j].loc[['x','y','w','h']])
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)),color, 2)
            cv2.putText(img, str(int(f.iloc[j].loc['id'])),(int(x+w), int(y+h+10)),cv2.FONT_HERSHEY_SIMPLEX,0.5, (20,240,60), 2)
        cv2.imwrite('/test/RD/motout/gt/'+str(i)+'.jpg',img)

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
        
        

def sinfit(y,doplot = False):
    #y = [i[0] for i in df[(df[1]==319) & (df[0]>700)][5]]
    if len(y)<=6:
        print('cant perform sine fit')
        return 
    
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


def angle(pos):
    
    reg = LinearRegression().fit(pos[:,0].reshape(-1,1), pos[:,1].reshape(-1,1))
    return float(reg.coef_)

def pair_angle(pos1,pos2):
    d1 = pos1[-1] - pos1[0]
    d2 = pos2[-1] - pos2[0]
    dot = np.dot((d1,d2))/(np.sum(np.square(d1)) + np.sum(np.square(d2)))
    angle = np.arccos(dot)*180/np.pi  # angle in degrees
    
    return dot,angle


    
    