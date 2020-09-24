# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import time
import cv2
import os
import sys
from datetime import datetime
import pandas as pd

if 'c:\\users\\81807\\documents\\RD\\deep_sort' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\deep_sort')

from analysis import imgplot,dispcv, reshapeimg
import matplotlib.pyplot as plt
from camera import xyzpos, td3d, loadjson, corrpos
                    


def load(confidence = 0.5, threshold = 0.3):
    path = {}
    path['yolo'] = 'c:\\users\\81807\\documents\\RD\\yolo-frcnn\\yolo-coco'
    path['confidence'] = confidence
    path['threshold'] = threshold  # nms

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([path["yolo"], "coco.names"])
    path['LABELS'] = np.array(open(labelsPath).read().strip().split("\n"))

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    path['COLORS'] = np.random.randint(0, 255, size=(len(path['LABELS']), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([path["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([path["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return path,net


path,net = load()
basepath = 'C:\\Users\\81807\\Documents\\RD\\'



def givebox(path,net,image,frame_index,doplot = False,docv = False):
    
    
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    
    boxes = []
    confidences = []
    classIDs = []
    pos = []
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
    
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > path["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
    
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
    
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, path["confidence"],
        path["threshold"])
    
    camera = False
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if classIDs[i]==0:
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                
                pos.append([frame_index,x,y,w,h])
                
                color = (120,60,180)
                if camera=='True':
                    
                    pts,(rx,ry,rz) = corrpos(x,y,w,h)
                    print(rx,ry,rz)
                    # draw a bounding box rectangle and label on the image
                
                    pts = np.int32(pts)
                    for i in range(4):
                        cv2.line(image,tuple(pts[i]),tuple(pts[(i+1)%4]),(255,0,0),2)    
                    cv2.rectangle(image, (pts[0][0], y), (pts[0][0]+ w, pts[0][1]), color, 1)
                    text = "{}: {}".format('(x,y,z)', str((np.round(rx),np.round(ry),np.round(rz))))
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
                elif doplot == True or docv == True:
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 1)
                    text = str(np.round(confidences[i],2))
                    cv2.putText(image, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
    
    # show the output image
    if doplot==True:
        imgplot(image)
    elif docv == True:
        cv2.imshow('frame', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return pos


def getrstp(link):

    cap = cv2.VideoCapture("rtsp://admin:0000@192.168.99.3:554/trackID=1")
    ctrl = 1
    ctrl2 = 1
    frame_index = -1 
    
    path,net = load()
    basepath = 'C:\\Users\\81807\\Documents\\RD\\'
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if ctrl2 % min(5,ctrl) != 0:
            ctrl2 += 1
            frame_index += 1
            continue
        
        if frame_index%2==0 and ctrl<6:
            ctrl+=1
        ctrl2=1
        
        frame = reshapeimg(frame)
        
        frame_index += 1
        cv2.imwrite(basepath+'realdata\\'+datetime.now().strftime('%m%d%H%M%S%f'),frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

