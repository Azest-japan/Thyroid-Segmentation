# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:37:53 2020

@author: 81807
"""

import numpy as np
import cv2
import os
import sys
import time
from datetime import datetime

import ftplib
from ftplib import FTP
import shutil
import urllib.request as request
from dateutil import parser
from contextlib import closing

if 'c:\\users\\81807\\documents\\RD\\yolo-frcnn' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\yolo-frcnn')
    
if 'c:\\users\\81807\\documents\\RD\\deep_sort' not in sys.path:
    sys.path.append('c:\\users\\81807\\documents\\RD\\deep_sort')

from dal import Dal


def select_vid(dal,day,baseloc = '/ftp_share/VideoVolume1/'):
    fullp = baseloc+day+'/'
    ftp = FTP('192.168.99.101','admin','admin')
    ftp.cwd(fullp)
    vlist = ftp.nlst()
    vtlist = []
    for vname in vlist:
        if vname[0] == '2':
            vsp = vname.split('-')
            dy,stime,etime,camIP,camname = vsp[:2] + vsp[2].split('_') + [vsp[3].split('.')[0]]
            camname = camIP+'_'+camname
    
            stime = datetime(int(dy[:4]),int(dy[4:6]),int(dy[6:]),int(stime[:2]),int(stime[2:4]),int(stime[4:]))
            etime = datetime(int(dy[:4]),int(dy[4:6]),int(dy[6:]),int(etime[:2]),int(etime[2:4]),int(etime[4:]))
            vtlist.append([vname,dy,stime,etime,camname])
    
    vtlist.sort(key=lambda x:x[2])
    stmax = vtlist[-1][2]
    
    for vname,dy,stime,etime,camname in vtlist:
        dal.insert_video(vname,day,stime,etime,camname,check=True)
    return stmax

def ftpconnect(startday='RecordFolder20190622',link='192.168.99.101',baseloc = '/ftp_share/VideoVolume1/'):
    
    
    """
    def savedb(link):
        with closing(request.urlopen('ftp://admin:admin@'+link+fullp+'nvr.db')) as r:
            with open('C:\\Users\\81807\\Documents\\RD\realdata\\a.db', 'wb') as f:
                shutil.copyfileobj(r, f)
    """
    waitc = 0
    dal = Dal()
    
    try:
        ftp = FTP(link,'admin','admin')
        ftp.cwd(baseloc)
        daylist = np.sort([i for i in ftp.nlst() if 'Record' in i])
        starti = 0
        
        if startday != None:
            starti = np.where(daylist == startday)[0]
            if len(starti) != 1: 
                return
            
        for day in daylist[starti[0]:]:
            stmax = select_vid(dal,day,baseloc)
            
        fullp = baseloc+daylist[-1]+'/'

        while True:
            ftp = FTP('192.168.99.101','admin','admin')
            ftp.cwd(fullp)
            vlist = ftp.nlst()
            vtlist = []
            for vname in vlist:
                if vname[0] == '2':
                    vsp = vname.split('-')
                    dy,stime,etime,camIP,camname = vsp[:2] + vsp[2].split('_') + vsp[3:]
                    camname = camIP+'_'+camname
                    
                    stime = datetime(int(dy[:4]),int(dy[4:6]),int(dy[6:]),int(stime[:2]),int(stime[2:4]),int(stime[4:]))
                    etime = datetime(int(dy[:4]),int(dy[4:6]),int(dy[6:]),int(etime[:2]),int(etime[2:4]),int(etime[4:]))
                    vtlist.append([vname,dy,stime,etime,camname])
                    
            vtmax = max(vtlist,key=lambda x:x[2])
            
            if vtmax[2] > stmax:
                stmax = vtmax[2]
                dal.insert_video(vname=vtmax[0],day=day,stime=vtmax[2],etime=vtmax[3],camname=vtmax[4],check=False)
            
            elif waitc <2:
                waitc += 1
                print('waiting ',waitc)
                time.sleep(1)
            else:
                break

    except ftplib.all_errors:
        return 


# 2)
def savebox(startday='RecordFolder20201005',link='192.168.99.101',baseloc='/ftp_share/VideoVolume1/',username='admin',password='admin'): 
    ftp = FTP(link,username,password)
    ftp.cwd(baseloc)
    daylist = np.sort([i for i in ftp.nlst() if 'Record' in i])
    starti = 0
    
    if startday != None:
        starti = np.where(daylist == startday)[0]
        dal = Dal()
        if len(starti)!=1: 
            return
    try:
        for day in daylist[starti[0]:]:
            fullp = baseloc+day+'/'
            dal.insert_box(ftplink='ftp://'+username+':'+password+'@'+link+fullp,day=day)
    except KeyboardInterrupt:
        return


# 3)

def savedeepbox(startday='RecordFolder20201005',link='192.168.99.101',baseloc='/ftp_share/VideoVolume1/',username='admin',password='admin'): 
    
    ftp = FTP(link,username,password)
    ftp.cwd(baseloc)
    daylist = np.sort([i for i in ftp.nlst() if 'Record' in i])
    starti = 0
    
    if startday != None:
        starti = np.where(daylist == startday)[0]
        dal = Dal()
        if len(starti)!=1: 
            return
        
    try:
        for day in daylist[starti[0]:]:
            print(day)
            fullp = baseloc+day+'/'
            dal.calc_id(ftplink='ftp://'+username+':'+password+'@'+link+fullp,day=day)
    except KeyboardInterrupt:
        return
            


