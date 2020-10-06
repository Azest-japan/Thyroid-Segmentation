# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from analysis import bb_iosb,eudist
#from copy import deepcopy

class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=32, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []             # stores tracks
        self.trackid_indexid = {}
        self.indexid_trackid = {}
        self._next_id = 0
        self.critical_tracks = {}
        self.middle_check = {}
        self.track_pos = []
        self.switchcount = 0
        self.swapcount = 0
        
    def predict(self,shape):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            if track.is_confirmed()==2:
                track.print_details()
                
            #print('\npredict track ',track.track_id)
            track.predict(self.kf,self.check_spot(track.to_xyah(),shape,8))

    def update(self, detections, frame, fno, encoder):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """ 

        shape = frame.shape[:2]  # used for track location during initialization - middle, edge
        # Run matching cascade stores track index. 
        total_pos_row,total_pos_col = np.ceil(shape).astype(int)
        self.track_pos = [[] for _ in range(total_pos_row*total_pos_col)]
        
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections,fno)
        
        '''
        print('update')
        print('matches\n',[(self.tracks[track_idx].track_id,det_id) for track_idx,det_id in matches])
        print('tracks-1 \n',[self.tracks[track_idx].track_id for track_idx in unmatched_tracks])
        print('detections-1 \n',unmatched_detections)
        '''

        m = {self.tracks[track_idx].track_id : det_id for track_idx, det_id in matches}
            
        self.trackid_indexid = {}
        self.indexid_trackid = {}
        
        # initialize iosb and create indexid_trackid_dictionary
        for i in range(len(self.tracks)):
            self.tracks[i].iosb = [0,-1]
            self.trackid_indexid[self.tracks[i].track_id] = i
            self.indexid_trackid[i] = self.tracks[i].track_id
        
            
                
        # calculate and update iosb
        for i in range(1,len(self.tracks)):
            ti = self.tracks[i]
            boxA = ti.to_tlbr()
            ti_id = ti.track_id
            
            for j in range(i):
                tj = self.tracks[j]
                boxB = tj.to_tlbr()
                tj_id = tj.track_id
                
                iosb_val = bb_iosb(boxA, boxB)
                
                if ti.iosb[0] < iosb_val:
                    ti.iosb = [iosb_val,tj_id]
                    
                if tj.iosb[0] < iosb_val:
                    tj.iosb = [iosb_val,ti_id]
                
                   
                # iosb>0.5 and the tuple is not present in the critical_tracks and the number of detection boxes is not 2
                if iosb_val>0.41 and self.tracks[i].is_confirmed() and self.tracks[j].is_confirmed() and (not min(ti_id,tj_id) in self.critical_tracks.keys()) and np.sum([ti.time_since_update,tj.time_since_update])>2:
                    self.critical_tracks[min(ti_id,tj_id)] = [max(ti_id,tj_id),iosb_val]

                elif min(ti_id,tj_id) in self.critical_tracks.keys() and self.critical_tracks[min(ti_id,tj_id)][0]==max(ti_id,tj_id):
                    if ti_id not in self.trackid_indexid.keys() or tj_id not in self.trackid_indexid.keys():
                        self.critical_tracks.pop(min(ti_id,tj_id))
                    
                    elif iosb_val<0.22:
                        print('finally')
                        print(self.critical_tracks)
                        print(ti_id,tj_id,ti_id in self.trackid_indexid.keys(),tj_id in self.trackid_indexid.keys())
                        
                        box_i = ti.to_tlwh()
                        box_j = tj.to_tlwh()
                        fts = encoder(frame,[box_i,box_j])
                        cost_matrix = self.metric.distance(fts, np.array([ti_id,tj_id]))
                        gating_threshold = 9.48 # 4 dimensions
                        gated_cost = 1e+5
                        for row, tr in enumerate([ti,tj]):
                            gating_distance = self.kf.gating_distance(tr.mean, tr.covariance,[ti.to_xyah(),tj.to_xyah()], only_position=False)
                            cost_matrix[row, gating_distance > gating_threshold] = gated_cost
                        #print('cost',cost_matrix)
                        
                        i_det = np.argmin(cost_matrix[0,:])
                        j_det = np.argmin(cost_matrix[1,:])
                        
                        def switch(ti,tj):
                            ti.track_id = tj_id
                            tj.track_id = ti_id

                        if i_det == 1 and j_det == 0:
                            print('switched')
                            switch(ti,tj)
                            self.switchcount += 1
                        
                        elif i_det == j_det:
                            col = i_det 
                            row = np.argmin(cost_matrix[:,col])
                            if row!=col:
                                switch(ti,tj)
                                print('switched')
                                self.switchcount += 1
                    
                        self.critical_tracks.pop(min(ti_id,tj_id))
                    else:
                        self.critical_tracks[min(ti_id,tj_id)] = [max(ti_id,tj_id),iosb_val]
                        
        m = {self.tracks[track_idx].track_id : det_id for track_idx, det_id in matches}
        #print('critical tracks',self.critical_tracks)
        
        
        
        # Update track set.
        for track_idx, detection_idx in matches:
            #print('\n update track ',track_idx)
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], shape, fno)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        
        
        
        self.trackid_indexid = {}
        self.indexid_trackid = {}
        
        for i in range(len(self.tracks)):
            self.trackid_indexid[self.tracks[i].track_id] = i
            self.indexid_trackid[i] = self.tracks[i].track_id
        
        # update track pos
        for tr in self.tracks:
            if tr.time_since_update>1 and tr.is_confirmed():
                # update track_pos for middle_check
                trbox = np.array(tr.to_xyah()[:2])
                trpos = np.floor(trbox/100).astype(int)
                
                self.track_pos[trpos[1]*total_pos_col+trpos[0]].append(tr.track_id)
        
        #print('track pos',self.track_pos)
        #print('middle check',self.middle_check)
        
        
        trs_todel = []
        # middle check
        for ntrid in list(self.middle_check):
            
            time = self.middle_check[ntrid]
            if time>8 or not ntrid in self.trackid_indexid.keys():
                self.middle_check.pop(ntrid)
                continue
            
            ntr = self.tracks[self.trackid_indexid[ntrid]]
            nbox = np.array(ntr.to_xyah()[:2])
            npos = np.floor(nbox/100).astype(int)
            nft = ntr.features

            if len(nft)==0:
                nft = encoder(frame,[ntr.to_tlwh()])

            dmax = 448
            saveotr = None
            
            for i in range(max(npos[1]-1,0),min(npos[1]+1,total_pos_row)):
                for j in range(max(npos[0]-1,0),min(npos[0]+1,total_pos_col)):
                    
                    if len(self.track_pos[i*total_pos_col+j])==0:
                        continue
  
                    for otrid in self.track_pos[i*total_pos_col+j]:
                        
                        if otrid == ntrid or otrid in self.middle_check.keys():
                            continue
                        
                        otr = self.tracks[self.trackid_indexid[otrid]]
                        obox = np.array(otr.to_xyah()[:2])
                        d = eudist(obox,nbox)
                        oft = otr.features
                        
                        #print(ntrid,otrid,d,dmax)
                        if len(oft)==0:
                            oft = encoder(frame,[otr.to_tlwh()])
                  
                        cosd = self.metric._metric(np.array(nft),np.array(oft))
                        print('cosd ',cosd,otrid,ntr.track_id)
                        if cosd<0.2:
                            if d<dmax:
                                self.swapcount +=1
                                print('grabbed!!!!!',self.swapcount)
                                dmax = d
                                saveotr = otr
                                ntr.checkspot = 'edge'
            
            
            if saveotr != None:
                trs_todel.append(ntr)
                saveotr.mean[:4] = ntr.mean[:4]
                saveotr.time_since_update = 0
                if ntrid in m.keys():
                    
                    m[saveotr.track_id] = m[ntrid]
                    m.pop(ntrid)
                    print('check swap match',m[saveotr.track_id])
                if len(ntr.pastpos)>0:
                    saveotr.pastpos[-1] = ntr.pastpos[-1]
                    
                self.middle_check.pop(ntrid)   # only way to delete items from middle check
                
            else:
                self.middle_check[ntrid] = time+1 # only way to update middle check
        
        
        for i in trs_todel:
            del(self.tracks[np.where(np.array(self.tracks)==i)[0][0]])
        
        
        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        #print('at ',active_targets,'\n')
        
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
                
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
            
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
        #print('matches ',m)
        return m
    



    def _match(self, detections,fno):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            cost_matrix = self.metric.distance(features, targets)

            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        
        
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update < 4]
        
        #print('iou_track_candidates',iou_track_candidates,[(k,self.tracks[k].track_id,self.tracks[k].time_since_update) for k in unmatched_tracks_a])
        
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update >= 4]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
            
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections


    def _initiate_track(self, detection, shape, fno):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        tr = Track(mean, covariance, self._next_id, self.n_init, self.max_age, self.check_spot(detection.to_xyah(),shape,fno),
            detection.feature)
        self.tracks.append(tr)
        
        #if fno>300:
        #    print('initiate ',tr.track_id,tr.checkspot)
            
        if tr.checkspot == 'middle':
            self.middle_check[tr.track_id] = 0
            
        self._next_id += 1
    
    def check_spot(self,box,shape,fno):
        if fno<4:
            return 'edge'
        
        else:
            x,y,a,h = box
            h_img,w_img = shape
            if y<0.9*h_img and y>0.1*h_img and x<0.9*w_img and x>0.1*w_img:
                return 'middle'
            else:
                return 'edge'
        