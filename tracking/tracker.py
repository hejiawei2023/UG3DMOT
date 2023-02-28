# from tracking.trajectory import Trajectory
import numpy as np

from tracking.cov_data import Covariance
from tracking.kalman_filter_3d import Trajectory_3D
from utils.cost import js_distance, m_distance


class Tracker:
    def __init__(self, tracking_type, cfg):
        self.tracking_type = tracking_type
        self.current_frame = None
        self.current_bboxes = None
        self.bbox_scores = None
        self.config = cfg
        self.active_trajectories = {}
        self.history_dead_trajectories = {}
        self.ID_seed = 0
        self.cov_param = Covariance(cfg.covariance_id, tracking_type.lower())


    def tracking(self, bboxes, scores, frame):

        self.current_frame = frame
        if not isinstance(bboxes, (list, tuple)):
            self.current_bboxes = [bboxes[i] for i in range(len(bboxes))]
            self.bbox_scores = [scores[i] for i in range(len(scores))]
        else:
            self.current_bboxes = bboxes
            self.bbox_scores = scores
        self.det_distribution()
        self.predict()
        ids = self.association()
        self.update_init(ids)
        self.trajectory_management()

    def det_distribution(self):
        det_states = []

        for det, det_score in zip(self.current_bboxes, self.bbox_scores):
            temp_state = dict()
            if det[3] >= np.pi: det[3] -= np.pi * 2     # make the theta still in the range
            if det[3] < -np.pi: det[3] += np.pi * 2
            temp_state["state"] = det

            cov_param = self.cov_param.get_param()
            temp_state["cov"] = cov_param[2]
            det_states.append(temp_state)
        self.current_bboxes = det_states

    def predict(self):
        for key in self.active_trajectories.keys():
            self.active_trajectories[key].predict(self.current_frame)

    def association(self):

        if self.bbox_scores is None or len(self.bbox_scores) == 0:
            return []

        if len(self.active_trajectories) == 0 or self.active_trajectories is None:
            ids = []
            for i in range(len(self.bbox_scores)):
                ids.append(self.ID_seed)
                self.ID_seed += 1
            return ids

        distance_matrix = self.cost_fusion()

        trks_id = list(self.active_trajectories.keys())
        num_dets, num_trks = distance_matrix.shape
        distance_1d = distance_matrix.reshape(-1)
        index_1d = np.argsort(distance_1d)
        index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)
        trks_id_matches_to_dets = [-1] * num_dets   # the record is the id of the trks
        dets_id_matches_to_trks = [-1] * num_trks
        for sort_i in range(index_2d.shape[0]):
            detection_id = int(index_2d[sort_i][0])
            tracking_id = int(index_2d[sort_i][1])
            if dets_id_matches_to_trks[tracking_id] == -1 and trks_id_matches_to_dets[detection_id] == -1 \
                    and distance_1d[index_1d[sort_i]] < self.config.distance_threshold[self.tracking_type]:
                dets_id_matches_to_trks[tracking_id] = detection_id
                trks_id_matches_to_dets[detection_id] = trks_id[tracking_id]

        for i in range(len(trks_id_matches_to_dets)):
            if trks_id_matches_to_dets[i] == -1:
                trks_id_matches_to_dets[i] = self.ID_seed
                self.ID_seed += 1
        return trks_id_matches_to_dets

    def cost_fusion(self):
        dist_matrix = np.zeros((len(self.bbox_scores), len(self.active_trajectories)))
        cov_factor = np.zeros(len(self.active_trajectories))

        for i, det_i in enumerate(self.current_bboxes):
            for j, key in enumerate(self.active_trajectories.keys()):
                trk_j = self.active_trajectories[key].trajectory[self.current_frame]
                cov_factor[j] = np.mean(np.diagonal(trk_j.P_predict[:3, :3]))

                if self.config.asso == "js_dis":
                    dist_matrix[i, j] = js_distance(det_i, trk_j, self.config.dimensions)

                if self.config.asso == "m_dis":
                    dist_matrix[i, j] = m_distance(det_i, trk_j, self.config.dimensions)

        if self.config.matching_mechanism == 'cov_guide':
            dist_matrix = dist_matrix * cov_factor
            return dist_matrix
        else:
            return dist_matrix

    def update_init(self, ids):

        assert len(ids) == len(self.bbox_scores), 'ids shape error'
        for i in range(len(self.bbox_scores)):
            label = ids[i]
            bbox = self.current_bboxes[i]["state"]
            score = self.bbox_scores[i]
            if label in self.active_trajectories.keys() and score > self.config.update_score:
                track = self.active_trajectories[label]
                track.update(bbox3D=bbox, time_stamp=self.current_frame, score=score)
            elif score > self.config.init_score:
                track = Trajectory_3D(bbox3d=bbox,
                                      score=score,
                                      time_stamp=self.current_frame,
                                      cfg=self.config,
                                      ID=label,
                                      cov_data=self.cov_param.get_param())
                self.active_trajectories[label] = track

    def trajectory_management(self):
        history_dead_ids = []   # Historical death can be output but not participated in matching
        dead_ids = []
        for key in self.active_trajectories.keys():

            track = self.active_trajectories[key]
            if track.age <= self.config.min_hits:
                if track.time_since_update > 0:
                    dead_ids.append(key)
            else:
                if track.time_since_update > self.config.miss_age:
                    history_dead_ids.append(key)

        for dead_id in dead_ids:
            self.active_trajectories.pop(dead_id)

        for history_dead_id in history_dead_ids:
            trk = self.active_trajectories.pop(history_dead_id)
            self.history_dead_trajectories[history_dead_id] = trk

    def printout_trace(self):
        trks = {}
        trks.update(self.history_dead_trajectories)
        trks.update(self.active_trajectories)

        time_sequence_trajectory = {}
        for obj_id in trks.keys():
            track = trks[obj_id]
            for frame in track.trajectory.keys():
                obj = track.trajectory[frame]

                if obj.score is None:
                    continue
                if frame in time_sequence_trajectory.keys():
                    time_sequence_trajectory[frame][obj_id] = (np.array(obj.x_update.T), obj.score)
                else:
                    time_sequence_trajectory[frame] = {obj_id: (np.array(obj.x_update.T), obj.score)}
        return time_sequence_trajectory
