
import numpy as np
import cv2
import sys
import imutils

import json
import matplotlib.pyplot as plt
from analysis import imgplot,write_json,read_json


'''
# Homography

img = cv2.imread('/test/RD/h3.jpg')
imgplot(img)

pts_src = np.array([[351, 200], [441, 200], [210, 443],[585, 443]])
pts_dst = np.array([[210, 200],[585, 200],[210,443],[585, 443]])

    
h, status = cv2.findHomography(pts_src, pts_dst)
im_dst = cv2.warpPerspective(img, h, (img.shape[1],img.shape[0]))

imgplot(im_dst)

'''

def chess(fno):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    img = cv2.imread('/test/RD/Images/frame'+str(fno)+'.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        #imgplot(img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return ret, mtx, dist, rvecs, tvecs, img

def newmat(mtx,dist):
    img2 = cv2.imread('/test/RD/Images/frame224.jpg')
    h,  w = img2.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    imgplot(dst)
    return newcameramtx
    
def savejson(ret, mtx, dist, rv, tv,wmse):
    
    cp = {}
    cp['ret'] = ret
    cp['mtx'] = mtx.tolist()
    cp['dist'] = dist.tolist()
    cp['rv'] = rv.tolist()
    cp['tv'] = tv.tolist()
    cp['wmse'] = wmse

    write_json('/test/RD/camera',cp)

def loadjson(path):
    cp = read_json(path)
    cp['mtx'] = np.float32(cp['mtx'])
    cp['dist'] = np.float32(cp['dist'])
    cp['rv'] = np.float32(cp['rv'])
    cp['tv'] = np.float32(cp['tv'])
    cp['op'][0] = np.float32(cp['op'][0])
    cp['ip'][0] = np.float32(cp['ip'][0])
    return cp

def td3d(px,mtx,dist,rv,tv):
    u,v = px
    x,y,z = np.matmul(np.linalg.inv(np.hstack((cv2.Rodrigues(rv)[0][:,:2],tv))),np.matmul(np.linalg.inv(mtx),[u,v,1]))
    x = int(x/z+0.5)
    y = int(y/z+0.5)
    emin = 163840
    rc = ()
    pc = ()
    for i in range(x-32,x+32,2):
        for j in range(y-18,y+18,2):
            u1,v1 = cv2.projectPoints(np.float32([i,j,0]), rv, tv, mtx, dist)[0].reshape(-1)
            er = np.sqrt(np.square(u-u1)+16/9*np.square(v-v1))
            if emin>er:
                emin = er
                rc = (i,j)
                pc = (u1,v1)
    return rc,pc    

def findh(rc,tpx,mtx,dist,rv,tv):
    u,v = tpx
    x,y = rc
    x = x - 15
    emin = 1024
    h = 150
    for z in range(150,200):
        u1,v1 = cv2.projectPoints(np.float32([x,y,z]), rv, tv, mtx, dist)[0].reshape(-1)
        er = np.sqrt(np.square(u-u1)+16/9*np.square(v-v1))
        if emin>er:
            emin = er
            h = z
    return h

def findxy(z,tpx,mtx,dist,rv,tv):
    u,v = tpx
    rdgs = cv2.Rodrigues(rv)[0]
    x,y,c = np.matmul(np.linalg.inv(np.hstack((rdgs[:,:2],rdgs[2]*z+tv))),np.matmul(np.linalg.inv(mtx),[u,v,1]))
    x = int(x/c+0.5)
    y = int(y/c+0.5)
    emin = 1024
    rc = ()
    pc = ()
    for i in range(x-16,x+16):
        for j in range(y-9,y+9):
            u1,v1 = cv2.projectPoints(np.float32([i,j,z]), rv, tv, mtx, dist)[0].reshape(-1)
            er = np.sqrt(np.square(u-u1)+16/9*np.square(v-v1))
            if emin>er:
                emin = er
                rc = (i,j)
                pc = (u1,v1)
                
    return rc,pc,z    
    
def xyzpos(bottom,top):
    cp = loadjson()
    mtx,dist,rv,tv = cp['mtx'],cp['dist'],cp['rv'],cp['tv']
    (x,y),_ = td3d(bottom,mtx,dist,rv,tv)
    z = findh((x,y),top,mtx,dist,rv,tv)
    return x,y,z


def perror(ip,op,mtx,dist):
    
    _,rv,tv = cv2.solvePnP(op[0], ip[0], mtx,dist,cv2.SOLVEPNP_AP3P)
    i1,_ = cv2.projectPoints(op[0], rv, tv, mtx, dist)
    i1 = i1.reshape((len(ip[0]),2))
    mse = (np.square(ip[0] - i1)).mean(axis=0)
    
    return rv,tv,np.sqrt(np.sum(np.multiply([1,16/9],mse)))/(len(ip[0])*2**0.5)


def corrpos(x,y,w,h):
    cp = loadjson()
    mtx,dist,rv,tv = cp['mtx'],cp['dist'],cp['rv'],cp['tv']
    
    (px,py),_ = td3d((x+w/2, y+h),mtx,dist,rv,tv)
    px = px - 15  # account for the foot size (30/2)
    
    for z in range(150,200):
        u1,v1 = cv2.projectPoints(np.float32([px,py,z]), rv, tv, mtx, dist)[0].reshape(-1)
        if v1<y or u1<x:
            print(u1,v1)
            break
    
    p0 = cv2.projectPoints(np.float32([px,py-25,0]), rv, tv, mtx, dist)[0].reshape(-1)
    p1 = cv2.projectPoints(np.float32([px,py+25,0]), rv, tv, mtx, dist)[0].reshape(-1)
    p2 = cv2.projectPoints(np.float32([px,py+25,z]), rv, tv, mtx, dist)[0].reshape(-1)
    p3 = cv2.projectPoints(np.float32([px,py-25,z]), rv, tv, mtx, dist)[0].reshape(-1)
    return (p0,p1,p2,p3),(px,py,z)

'''
ret, mtx, dist, rvecs, tvecs, img = chess(125)

img2 = cv2.imread('/test/RD/Images/f20.jpg')
op,ip = [np.float32([[  0,   0,   0],
        [180,   7,   0],
        [174,  76,   0],
        [174, 146,   0],
        [251,  67,   0],
        [265, 153,   0],
        [338, 205,   0],
        [420, 167,   0],
        [454, 121,   0]])],[np.float32([[176, 27],
        [1506, 1185],
        [1789, 1174],
        [2081, 1171],
        [1644, 1294],
        [2055, 1323],
        [2361, 1485],
        [2047, 1754],
        [1606, 1904]])]

'''


def saveframe(vpath,spath,start,end=None):
    vs = cv2.VideoCapture(vpath)
    #(W, H) = (None, None)

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



''' SIFT '''

def sift(img,img2):

    # find the keypoints and descriptors with SIFT
    k1,d1 = computeKeypointsAndDescriptors(img)
    k2,d2 = computeKeypointsAndDescriptors(img2)

    MIN_MATCH_COUNT = 16
    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(d1,d2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ k1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ k2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img,k1,img2,k2,good,None,**draw_params)
    imgplot(img3)
    return img3,M

