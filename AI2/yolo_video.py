# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import sys
import csv
import matplotlib.pyplot as plt
import pandas as pd

basepath = 'c:\\users\\81807\\documents\\RD\\'
if basepath + 'deep_sort' not in sys.path:
    sys.path.append(basepath + 'deep_sort')
from analysis import imgplot
# construct the argument parse and parse the arguments

def load(basepath,confidence=0.5,threshold=0.3):
    path = {}
    path['yolo'] = basepath + 'yolo-frcnn\\yolo-coco\\'
    path['confidence'] = 0.5
    path['threshold'] = 0.3
    

    labelsPath = os.path.sep.join([path["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    	dtype="uint8")
    
    weightsPath = os.path.sep.join([path["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([path["yolo"], "yolov3.cfg"])
    
    
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    
    path['input'] = basepath + 'shibuya_crossing.mp4'
    #path['output'] = '/test/RD/tb2.mp4'
    return path,net,ln

def save_csv(file,df):
    df.to_csv(file,header=None,index=False,mode='a')
    
def load_csv(file):
    return pd.read_csv(file,delimiter=',',names=['f','x','y','w','h'])


path,net,ln = load(basepath)
vs = cv2.VideoCapture(path["input"])
writer = None
(W, H) = (None, None)

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

pos = []
ctrl = 1
ctrl2 = 1
frame_index = -1
doplot = True

# loop over frames from the video file stream
while frame_index<560:
    
    grabbed,frame = vs.read()
    
    if ctrl2 % min(5,ctrl) != 0 or frame_index<550:
        ctrl2 += 1
        frame_index += 1
        continue
    print(frame_index)
    if frame_index%2 == 0 and ctrl<6:
        ctrl += 1
        
    ctrl = 1
    
    H,W = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            if classID!=0:
                continue
            
            confidence = scores[classID]
            
            if confidence>path['confidence']:
                box = detection[0:4]*np.array([W,H,W,H])
                (centerX,centerY,width,height) = box.astype('int')
                
                x = int(centerX - width/2)
                y = int(centerY - height/2)
                
                boxes.append([x,y,int(width),int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    print(boxes,classIDs)
    if len(boxes)>0:
        idxs = cv2.dnn.NMSBoxes(boxes,confidences,path['confidence'],path['threshold'])
        
        if len(idxs)>0:
            print('hello')
            for i in idxs.flatten():
                if classIDs[i]==0:
                    (x,y) = (boxes[i][0],boxes[i][1])
                    (w,h) = (boxes[i][2],boxes[i][3])
                    pos.append([frame_index,x,y,w,h])
                    
                    if doplot:
                        color = (80,160,200)
                        cv2.rectangle(frame,(x, y), (x + w, y + h), color, 2)
        
    if frame_index%1000==0:
        print('hi')
        #imgplot(frame)
        df = pd.DataFrame(pos)
        save_csv(basepath + 'shibuya_bbox.txt',df)
        pos = []
        
    frame_index += 1
    
    


                
    