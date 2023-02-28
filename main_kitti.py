# Author: hejw

import os
from dataset.kitti_data_base import velo_to_cam
import tqdm
from dataset.kitti_dataset import KittiTrackingDataset
from utils.config import load_config
from tracking.tracker import Tracker
import time
from utils.box_op import *


def tracking(cfg):
    dataset_path = os.path.join(cfg.dataset_root, cfg.split)
    detections_path = os.path.join(cfg.detections_root, cfg.split, cfg.tracking_type)
    tracking_type = cfg.tracking_type
    seq_list = cfg.tracking_seqs

    total_time, frame_num = 0, 0
    for i in tqdm.trange(len(seq_list)):

        seq_id = seq_list[i]
        detections_seq_path = os.path.join(detections_path, str(seq_id).zfill(4))
        tracker = Tracker(tracking_type=tracking_type, cfg=cfg)
        dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_seq_path, type=[tracking_type])

        for j in range(len(dataset)):
            P2, V2C, _, _, dets, det_scores, pose = dataset[j]
            mask = det_scores > cfg.input_score
            dets = dets[mask]
            det_scores = det_scores[mask]

            if dets is not None and len(dets) != 0:
                dets = convert_bbs_type(dets, cfg.dataset)
                dets = register_bbs(dets, pose)
            start = time.time()
            tracker.tracking(dets, scores=det_scores, frame=j)
            end = time.time()

            total_time += end-start
            frame_num += 1

        save_results(dataset, tracker, cfg, seq_id)

    print("FPS: %.2f" % (frame_num/total_time))


def save_results(dataset, tracker, cfg, seq_id):

    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)
    object_type = cfg.tracking_type
    file_path = os.path.join(save_path, str(seq_id).zfill(4)+'.txt')

    time_sequence_trajectory = tracker.printout_trace()

    with open(file_path, 'w') as f:
        for i in range(len(dataset)):
            P2, V2C, _, _, _, _, pose = dataset[i]

            if i in time_sequence_trajectory.keys():
                objs = time_sequence_trajectory[i]

                for trk_id in objs.keys():
                    updated_state, score = objs[trk_id]

                    box_template = updated_state[0, 0:7].reshape((1, 7))      # (x,y,z,yaw,l,w,h)
                    box_template = register_bbs(box_template,np.mat(pose).I)

                    box_template[:, 3] = -box_template[:, 3] - np.pi / 2
                    box_template[:, 2] -= box_template[:, 6] / 2
                    box_template[:, 0:3] = velo_to_cam(box_template[:, 0:3], V2C)[:, 0:3]

                    box = np.zeros(shape=(1,7))
                    box[:, 0:3] = box_template[:, 0:3]
                    box[:, 3:6] = box_template[:, 4:]
                    box[:, 6] = box_template[:,3]

                    box = box[0]
                    box2d = bb3d_2_bb2d(box,P2)
                    print('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (i,trk_id,object_type,box2d[0][0],box2d[0][1],box2d[0][2],
                             box2d[0][3],box[5],box[4],box[3],box[0],box[1],box[2],box[6],score),file = f)


if __name__ =='__main__':
    cfg_root = 'config/kitti.yaml'
    cfg = load_config(cfg_root)
    tracking(cfg)




