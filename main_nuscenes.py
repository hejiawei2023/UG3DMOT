# Author: hejw

import multiprocessing
import numpy as np
import os
from dataset.nuscenes_loader import NuScenesLoader
from tracking.tracker import Tracker
from utils.config import load_config


def save_results(tracker, frame_nums, summary_folder, seq_id, seq_name):

    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)
    object_type = tracker.tracking_type
    trks = tracker.printout_trace()
    IDs, bboxes, types, states= list(), list(), list(), list()
    for frame in range(frame_nums):
        frame_ids, frame_bboxes, frame_types, frame_states = list(), list(), list(), list()
        if frame in trks.keys():
            objs = trks[frame]
            for trk_id in objs.keys():
                updated_state, score = objs[trk_id]
                assert updated_state.shape == (1, 10), 'box shape error!'
                updated_state[0, 7] = score
                updated_state = updated_state.flatten()[:8]   # (x,y,z,yaw,l,w,h,score)
                frame_bboxes.append(updated_state)
                frame_ids.append('{:}_{:}'.format(seq_id, trk_id))
                frame_types.append(object_type)
                frame_states.append("alive_1_0")

        IDs.append(frame_ids)
        bboxes.append(frame_bboxes)
        types.append(frame_types)
        states.append(frame_states)

    np.savez_compressed(os.path.join(summary_folder, '{}.npz'.format(seq_name)), ids=IDs, bboxes=bboxes, states=states, types=types)


def main(obj_types, configs, data_folder, det_data_folder, result_folder, token=0, process=1):
    for obj_type in obj_types:
        summary_folder = os.path.join(result_folder, 'summary', obj_type)
        file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))

        for file_index, file_name in enumerate(file_names[:]):
            if (file_index % process) != token:
                continue
            print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1, len(file_names)))
            segment_name = file_name.split('.')[0]

            data_loader = NuScenesLoader(configs, [obj_type], segment_name, data_folder, det_data_folder, 0)
            tracker = Tracker(tracking_type=obj_type, cfg=configs)

            frame_num = len(data_loader)
            for frame_index in range(data_loader.cur_frame, frame_num):
                if frame_index % 10 == 0:
                    print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(obj_type, file_index, frame_index + 1, frame_num))

                frame_data = next(data_loader)
                dets = []
                det_scores = []
                for det in frame_data["dets"]:
                    if det[7] < configs.input_score:
                        continue
                    dets.append(det[:7])
                    det_scores.append(det[7])
                tracker.tracking(dets, scores=det_scores, frame=frame_index)
            save_results(tracker, frame_num, summary_folder, file_index, segment_name)


if __name__ == '__main__':
    cfg_root = './config/nuscenes.yaml'
    cfg = load_config(cfg_root)
    if cfg.split == "test":
        data_folder = os.path.join(cfg.dataset_root, 'test_2hz')
        det_data_folder = os.path.join(cfg.detections_root, 'test_2hz', 'detection')
        result_folder = os.path.join(cfg.save_path, 'test_2hz', cfg.tracker_name)
    else:
        data_folder = os.path.join(cfg.dataset_root, 'validation_2hz')
        det_data_folder = os.path.join(cfg.detections_root, 'validation_2hz', 'detection')
        result_folder = os.path.join(cfg.save_path, 'validation_2hz', cfg.tracker_name)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    summary_folder = os.path.join(result_folder, 'summary')
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)

    det_data_folder = os.path.join(det_data_folder, cfg.det_name)

    obj_types = cfg.tracking_type.split(',')
    for obj_type in obj_types:
        tmp_summary_folder = os.path.join(summary_folder, obj_type)
        if not os.path.exists(tmp_summary_folder):
            os.makedirs(tmp_summary_folder)

    if cfg.process > 1:
        pool = multiprocessing.Pool(cfg.process)
        for token in range(cfg.process):
            result = pool.apply_async(main,
                                      args=(obj_types, cfg, data_folder, det_data_folder, result_folder, token, cfg.process))
        pool.close()
        pool.join()
    else:
        main(obj_types, cfg, data_folder, det_data_folder, result_folder, 0, 1)



