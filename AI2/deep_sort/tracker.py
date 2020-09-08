# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from analysis import bb_iosb
from copy import deepcopy

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

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []             # stores tracks
        self.trackid_indexid = {}
        self.indexid_trackid = {}
        self._next_id = 0
        self.critical_tracks = []

    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            if track.is_confirmed()==2:
                track.print_details()
                
            #print('\npredict track ',track.track_id)
            track.predict(self.kf)

    def update(self, detections, frame, fno, encoder):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """ 
        
        shape = frame.shape[:2]  # used for track location during initialization - middle, edge
        # Run matching cascade stores track index. 
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        
        '''
        print('update')
        print('matches\n',[(self.tracks[track_idx].track_id,det_id) for track_idx,det_id in matches])
        print('tracks-1 \n',[self.tracks[track_idx].track_id for track_idx in unmatched_tracks])
        print('detections-1 \n',unmatched_detections)
        '''
        # stores track id
        m = {self.tracks[track_idx].track_id : det_id for track_idx, det_id in matches}
        
        self.trackid_indexid = {}
        self.indexid_trackid = {}
        
        # initialize iosb and create indexid_trackid_dictionary
        for i in range(len(self.tracks)):
            self.tracks[i].iosb = [0,-1,False]
            self.trackid_indexid[self.tracks[i].track_id] = i
            self.indexid_trackid[i] = self.tracks[i].track_id
            
            if self.tracks[i].track_id in m.keys():
                self.tracks[i].iosb[2] = True
        
        # calculate and update iosb
        for i in range(1,len(self.tracks)):
            boxA = self.tracks[i].to_tlbr()
            ti_id = self.tracks[i].track_id
            ti = self.tracks[i]
            
            for j in range(i):
                boxB = self.tracks[j].to_tlbr()
                iosb = bb_iosb(boxA, boxB)
                tj_id = self.tracks[j].track_id
                tj = self.tracks[j]
                
                if self.tracks[i].iosb[0] < iosb:
                    self.tracks[i].iosb[:2] = [iosb,tj_id]
                    
                if self.tracks[j].iosb[0] < iosb:
                    self.tracks[j].iosb[:2] = [iosb,ti_id]
                
                # iosb>0.5 and the tuple is not present in the critical_tracks and the number of detection boxes is not 2
                if iosb>0.5 and (not np.sort((ti_id,tj_id)) in np.array(self.critical_tracks)[:,:2]) and np.sum([ti.iosb[2],tj.iosb[2]])<2:
                    self.critical_tracks.append(list(np.sort((ti_id,tj_id)))+[iosb])
                    continue
                
                for index,items in enumerate(self.critical_tracks):
                    t1_id,t2_id,_ = items
                    if [min(t1_id,t2_id),max(t1_id,t2_id)] == [min(ti_id,tj_id),max(ti_id,tj_id)]:
                        if iosb<0.2:
                            box_i = ti.to_tlwh()
                            box_j = tj.to_tlwh()
                            fts = encoder(frame,[box_i,box_j])
                            cost_matrix = self.metric.distance(fts, np.array([ti_id,tj_id]))
                            gating_threshold = 9.48 # 4 dimensions
                            gated_cost = 1e+5
                            for row, tr in enumerate([ti,tj]):
                                gating_distance = self.kf.gating_distance(tr.mean, tr.covariance,[ti.to_xyah(),tj.to_xyah()], only_position=False)
                                cost_matrix[row, gating_distance > gating_threshold] = gated_cost
                            
                            i_det = np.argmin(cost_matrix[0,:])
                            j_det = np.argmin(cost_matrix[1,:])
                            
                            def switch(ti,tj):
                                ti_copy = deepcopy(ti)
                                ti.copy_track(tj)
                                tj.copy_track(ti_copy)
                                return ti,tj
                            
                            if i_det == 1 and j_det == 0:
                                ti,tj = switch(ti,tj)
                            
                            elif i_det == j_det:
                                col = i_det 
                                row = np.argmin(cost_matrix[:,col])
                                if row!=col:
                                    ti,tj = switch(ti,tj)
                            
                        else:
                            self.critical_tracks[index] = [min(t1_id,t2_id),max(t1_id,t2_id),iosb]
                        

        # Update track set.
        for track_idx, detection_idx in matches:
            #print('\n update track ',track_idx)
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
            
        
        #print('m ',m)
        
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], shape, fno)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

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
            
        return m

    def _match(self, detections):

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
        '''
        print('match a\n')
        print('matches_a\n',[(self.tracks[track_idx].track_id,det_id) for track_idx,det_id in matches_a])
        print('tracks-1_a\n',[self.tracks[track_idx].track_id for track_idx in unmatched_tracks_a])
        print('det-1\n',unmatched_detections)
        print('unconf \n',[self.tracks[track_idx].track_id for track_idx in unconfirmed_tracks])
        '''
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]   
        
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        '''
        print('\nmatch b')
        print('matches_b\n',[(self.tracks[track_idx].track_id,det_id) for track_idx,det_id in matches_b])
        print('tracks-1_b\n',[self.tracks[track_idx].track_id for track_idx in unmatched_tracks_b])
        print('det-1\n',unmatched_detections)
        '''
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, shape, fno):
        mean, covariance = self.kf.initiate(detection.to_xyah())

        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, self.check_spot(detection.to_xyah(),shape,fno),
            detection.feature))
        
        self._next_id += 1
    
    def check_spot(self,box,shape,fno):
        if fno<3:
            return 'middle'
        else:
            x,y,a,h = box
            h_img,w_img = shape
            if y<0.9*h_img and y>0.1*h_img and x<0.9*w_img and x>0.1*w_img:
                return 'middle'
            else:
                return 'edge'
        