

import numpy as np
import cv2
import sys
#sys.path.append('C:\\Users\\AZEST-2019-07\\Desktop\\pyfiles')
import dataprep
from dataprep import cut,disp
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et
import gc
import os

f1 = 'C:\\Users\\AZEST-2019-07\\Desktop\\Ito\\Patient 1'
fxml = 'C:\\Users\\AZEST-2019-07\\Desktop\\Ito\\Patient 1\\annotation1.xml'


def transfer(dbase):

    for path,subdir,files in os.walk(dbase):
        if len(files) ==0:
            print('nofile ',path,subdir)
        else:
            for file in files:
                if 'Image' in file:
                    ipath = path + '/'+ file
                    spath = os.path.dirname(path).replace('Data','Images') + '/' + os.path.basename(path)+'_'+file
                    #print(ipath,spath)
                    img = cv2.imread(ipath)[40:-40,20:-20,:]
                    cv2.imwrite(spath,img)
                #break

def loadimg(fpath):
    imgdict = {}
    for path,subdir,files in os.walk(fpath):
        for file in files:
            full_path = path+ '\\' + file
            if '.jpg' in full_path:
                print(file)
                img = cv2.imread(full_path)
                imgdict[file] = img
            
    return imgdict

imgdict = loadimg(f1)


def readxml(fxml,imgdict):
    #B-G-R values for the labels
    # Thyroid: B=100, G=200, R=0
    # Trachea: B= 100,G=200, R=200
    # Nodule:  B=30 , G=60 , R=160
    # Artery:  B=200, G=100, R=0
    
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
                    
                    cv2.polylines(imgdict[ann.attrib['name']],[pts],True,color = mask)
                    cv2.fillConvexPoly(imgdict[ann.attrib['name']], points=pts, color=mask)
        disp(imgdict[ann.attrib['name']])  # display the picture in opencv
        #cv2.imwrite('C:\\Users\\AZEST-2019-07\\Desktop\\Ito\\Patient 1\\annotated'+ann.attrib['name'], imgdict[ann.attrib['name']])
    return imgdict

readxml(fxml,imgdict)


'''
def saveframe(vpath,spath,start,end=None):
    vs = cv2.VideoCapture(vpath)
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
        
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
    
    f_rate = 30 # total/length
    fno = 0
    # loop over frames from the video file stream
    if end == None:
        end = total/30
        
    for i in range(end*f_rate):
        (grabbed, frame) = vs.read()
        if i> start*f_rate:
            cv2.imwrite(spath+'/f-'+str(fno)+'.jpg',frame)
            fno += 1

'''
