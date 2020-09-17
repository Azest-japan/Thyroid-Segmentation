# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, checkspot,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.pastpos = []
        self.dh = []     #
        self.dhd = 20    # 
        self.iosb = (0,-1)  # iosb, track number
        self.check_det = 0
        self.theta = 0
        self.check_switch = False
        self.checkspot = checkspot
        
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
       
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.to_tlwh()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.
        Returns
        -------
        ndarray
            The bounding box.
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def copy_track(self,tr):
        self.mean = tr.mean
        self.covariance = tr.covariance
        self.track_id = tr.track_id
        self.hits = tr.hits
        self.age = tr.age
        self.time_since_update = tr.time_since_update
        
        self.pastpos = tr.pastpos
        self.dh = tr.dh
        self.dhd = tr.dhd
        
        self.check_det = tr.check_det
        self.theta = tr.theta
        self.check_switch = False
        self.checkspot = tr.checkspot
        
    
    def caltheta(self):
        n = len(self.pastpos)
        start = 0
        for i in range(n):
            if self.pastpos[n-i-1] == []:
                start = n-i
                break
                
        if start < n-3:
            x1,y1 = self.pastpos[start][:2]
            x2,y2 = self.pastpos[-1][:2]
            if x1!=x2:
                self.theta = np.arctan((y2-y1)/(x2-x1))*180/np.pi
            else:
                self.theta = 90
                
            if (y2>y1 and x2<x1) or (y2<y1 and x2<=x1):
                self.theta = 180 + self.theta
            elif y2<y1 and x2>x1:
                self.theta = 360 + self.theta
            

    def predict(self, kf,checkspot):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """
        dhd = 15
        if len(self.dh) > 0:
            dhd = abs(self.mean[3] - self.dh[-1]) + 1

        self.mean, self.covariance = kf.predict(self.mean, self.covariance, self.pastpos, dhd + self.time_since_update*2)
        
        if len(self.dh) > 0:
            x,y,a,h = self.mean[:4]
            
            if h>1.1*self.dh[-1]:
                h = self.dh[-1]*1.1
            elif self.dh[-1]>1.1*h:
                h = self.dh[-1]/1.1
            
            w = a*h
            xl = x - w/2
            yl = y - h/2
            
            if self.time_since_update>0 and len(self.pastpos)>0 and checkspot=='middle':
                for i in self.pastpos:
                    if len(i)>0:
                        if h > 1.15*i[3]:
                            h = 1.15*i[3]
                        elif i[3] > 1.15*h:
                            h = i[3]/1.15
                        break
                
            x = xl + a*h/2
            y = yl + h/2
            
            self.mean[:4] = [x,y,a,h]
        
        self.age += 1
        self.time_since_update += 1
    
  
            
    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        
        (xl,yl,w,h) = detection.tlwh

        deth = h
        
        
        if len(self.dh)>0:
            if deth>1.15*self.dh[-1]:
                deth = 1.15*self.dh[-1]

            elif self.dh[-1]>1.15*deth:
                deth = self.dh[-1]/1.15
                
        if len(self.dh)>=1:
            self.dhd = abs(self.dh[-1] - deth) + 1
        
        if (self.iosb[0] < 0.1 and self.state == TrackState.Confirmed) or self.state == TrackState.Tentative:
            self.features.append(detection.feature)
            self.dh.append(deth)
            
        detection.set_det((xl,yl,w/h*deth,deth))
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah(),self.dhd)
        
        self.hits += 1
        self.time_since_update = 0
        self.pastpos.append(self.mean[:4])
        self.pastpos = self.pastpos[-8:]
        self.caltheta()
        
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        self.pastpos.append([])
        #self.dh = []
        
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            #print('Deleted ',self.track_id)
            
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            #print('Deleted ',self.track_id)

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    def print_details(self):
        print('track_id ', self.track_id)
        print('mean ', self.mean)
        print('hits ', self.hits)
        print('age ', self.age)
        print('time since update ', self.time_since_update)
        print('state ', self.state)
        
        print('pastpos ', self.pastpos)
        print('dh ', self.dh)
        print('iosb ',self.iosb)
        print('theta ',self.theta)
        print('check switch ', self.check_switch)
        print('checkspot ', self.checkspot)
