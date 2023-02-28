# The code refers to https://github.com/xinshuoweng/AB3DMOT

import numpy as np
from filterpy.kalman import KalmanFilter
from tracking.trajectory_state import TrajectoryState


class Trajectory_3D(object):

    def __init__(self, bbox3d, score, time_stamp, cfg, ID, cov_data):

        self.init_bbox = bbox3d
        self.cfg = cfg
        self.init_time = time_stamp
        self.id = ID
        self.trajectory = {}
        self.cov = cov_data

        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.init_kalman()
        self.init_trajectory(score)
        self.time_since_update = 0

    @property
    def age(self):
        return len(self.trajectory)

    def init_kalman(self):
        # define constant velocity model
        self.kf.F[:3, 7:] = np.eye(3)  # state transition matrix
        self.kf.H[:7, :7] = np.eye(7)  # measurement function

        self.kf.P = self.cov[0]   # state uncertainty

        self.kf.Q = self.cov[1]  # process uncertainty

        self.kf.R = self.cov[2]  # measurement uncertainty
        self.kf.x[:7] = self.init_bbox.reshape((7, 1))        # [x,y,z,theta,l,w,h]

    def init_trajectory(self, score):
        trajectory = TrajectoryState()
        trajectory.x_predict = self.kf.x
        trajectory.x_update = self.kf.x
        trajectory.P_predict = self.kf.P
        trajectory.P_update = self.kf.P
        trajectory.innovation_matrix = self.compute_innovation_matrix()
        trajectory.score = score
        trajectory.dets = self.init_bbox
        self.trajectory[self.init_time] = trajectory

    def update(self, bbox3D, time_stamp, score):
        """
        Updates the state vector with observed bbox.
        """
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
        predicted_theta = self.kf.x[3]

        new_theta = bbox3D[3]
        if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi: new_theta += np.pi * 2
        bbox3D[3] = new_theta

        if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
                new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        # 更新状态
        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the rage
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.time_since_update = 0
        self.trajectory[time_stamp].x_update = self.kf.x
        self.trajectory[time_stamp].P_update = self.kf.P
        self.trajectory[time_stamp].dets = bbox3D
        self.trajectory[time_stamp].score = score

    def predict(self, time_stamp):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.time_since_update += 1
        trajectory = TrajectoryState()
        trajectory.x_predict = self.kf.x
        trajectory.P_predict = self.kf.P

        # if self.time_since_update == 1:
        #     trajectory.x_update = self.kf.x
        #     trajectory.P_update = self.kf.P
        #     trajectory.score = self.trajectory[time_stamp-1].score

        trajectory.innovation_matrix = self.compute_innovation_matrix()
        self.trajectory[time_stamp] = trajectory

    def filtering(self):
        detected_num = 0
        total_score = 0
        for key in self.trajectory.keys():
            track_state = self.trajectory[key]
            if track_state.score is not None:  # 这就是匹配上
                detected_num += 1
                total_score += track_state.score

            if track_state.x_update is None:
                track_state.x_update = track_state.x_predict
        score = total_score / detected_num
        for key in self.trajectory.keys():
            track_state = self.trajectory[key]
            track_state.score = score

    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        innovation_matirx = np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
        return innovation_matirx




