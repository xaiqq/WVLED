import logging
import os
import sys
from collections import defaultdict
import pickle

from utils.envs import pathmgr
from utils.general import LOGGER


def load_image_lists(cfg, is_train):
    """
    Loading image paths from corresponding files.

    Args:
        cfg (CfgNode): config.
        is_train (bool): if it is training dataset or not.

    Returns:
        image_paths (list[list]): a list of items. Each item (also a list)
            corresponds to one video and contains the paths of images for
            this video.
        video_idx_to_name (list): a list which stores video names.
    """
    list_filenames = str(cfg.UCF101_24.FRAME_DIR)
    with open(cfg.UCF101_24.ANNOTATIONS_FILE, "rb") as file:
        videos_gt = pickle.load(file, encoding='latin1')
        file.close()
    videos_gt = videos_gt["train_videos"] if is_train else videos_gt["test_videos"]
    # videos_gt = videos_gt["train_videos"] if is_train else videos_gt["train_videos"]
    image_paths = defaultdict(list)
    video_name_to_idx = {}
    video_idx_to_name = []
    for video_gt in videos_gt:
        for vg in video_gt:
            # --------------------------------------------
            # 过滤类别
            # video_name = vg.strip().split("/")[0]
            # if video_name in ['BasketballDunk', 'Biking', 'CricketBowling', "HorseRiding", 'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', 'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TrampolineJumping', 'WalkingWithDog']:
            #     continue
            # --------------------------------------------
            if vg not in video_name_to_idx:
                idx = len(video_idx_to_name)
                video_name_to_idx[vg] = idx
                video_idx_to_name.append(vg)
            frame_names = os.listdir(os.path.join(list_filenames, vg))
            # frame_names = [frame_name for frame_name in frame_names if not frame_name.startswith('.')]
            frame_names = sorted(frame_names, key=lambda x:x[:-4])
            data_key = video_name_to_idx[vg]
            for frame_name in frame_names:
                image_paths[data_key].append(
                    os.path.join(list_filenames, vg, frame_name)
                )
    image_paths = [image_paths[i] for i in range(len(image_paths))]
    LOGGER.info("Finished loading image paths from: %s" % "".join(list_filenames))

    return image_paths, video_idx_to_name

def load_boxes_and_labels(cfg, imgs, id2img_names):
    """
    Loading boxes and labels from csv files.

    Args:
        cfg (CfgNode): config.
        mode (str): 'train', 'val', or 'test' mode.
    Returns:
        all_boxes (dict): a dict which maps from `video_name` and
            `frame_sec` to a list of `box`. Each `box` is a
            [`box_coord`, `box_labels`] where `box_coord` is the
            coordinates of box and 'box_labels` are the corresponding
            labels for the box.
    """
    gt_labels = str(cfg.UCF101_24.FRAME_DIR)
    with open(cfg.UCF101_24.ANNOTATIONS_FILE, "rb") as file:
        videos_gt = pickle.load(file, encoding='latin1')
        file.close()
    video_labels = videos_gt["gttubes"]
    all_boxes = {}
    count = 0
    unique_box_count = 0
    for video_id, _ in enumerate(imgs):
        video_name = id2img_names[video_id]
        if video_name not in all_boxes:
            all_boxes[video_name] = {}
        video_box_and_labels = video_labels[video_name]
        for video_classes, video_boxes in video_box_and_labels.items():
            for video_box in video_boxes:
                for i in range(video_box.shape[0]):
                    frame = int(video_box[i,0])
                    boxes = video_box[i, 1:].tolist()
                    if frame not in all_boxes[video_name]:
                        all_boxes[video_name][frame] = []
                    all_boxes[video_name][frame].append([boxes, [video_classes]])
                    # -------------------------------------------------
                    # 过滤
                    # if video_name.split('/')[0] == "CliffDiving":
                    #     all_boxes[video_name][frame].append([boxes, [0]])
                    # elif video_name.split('/')[0] == "VolleyballSpiking":
                    #     all_boxes[video_name][frame].append([boxes, [1]])
                    # elif video_name.split('/')[0] == 'Basketball':
                    #     all_boxes[video_name][frame].append([boxes, [2]])
                    # elif video_name.split('/')[0] == 'TennisSwing':
                    #     all_boxes[video_name][frame].append([boxes, [3]])
                    # -------------------------------------------------
                    unique_box_count += 1
                    count += 1
    return all_boxes
    

def get_keyframe_data(cfg, boxes_and_labels):
    """
    Getting keyframe indices, boxes and labels in the dataset.

    Args:
        boxes_and_labels (list[dict]): a list which maps from video_idx to a dict.
            Each dict `frame_sec` to a list of boxes and corresponding labels.

    Returns:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.
    """

    keyframe_indices = []
    keyframe_boxes_and_labels = []
    count = 0
    for video_idx in range(len(boxes_and_labels)):
        keyframe_boxes_and_labels.append([])
        sec_idx = 0
        frame_list = list(boxes_and_labels[video_idx].keys())
        len_frames = len(boxes_and_labels[video_idx].keys())
        if cfg.DATA.KEY_FRAME >= len_frames:
            key_frame_sample = len_frames // 2
        else:
            key_frame_sample = cfg.DATA.KEY_FRAME

        for frame in boxes_and_labels[video_idx].keys():
            if frame % key_frame_sample == 0:
                keyframe_indices.append(
                    (video_idx, sec_idx, frame-1)
                )
                keyframe_boxes_and_labels[video_idx].append(
                    boxes_and_labels[video_idx][frame]
                )
                sec_idx += 1
                count += 1
    LOGGER.info("%d keyframes used." % count)
    return keyframe_indices, keyframe_boxes_and_labels

def get_num_boxes_used(keyframe_indices, keyframe_boxes_and_labels):
    """
    Get total number of used boxes.

    Args:
        keyframe_indices (list): a list of indices of the keyframes.
        keyframe_boxes_and_labels (list[list[list]]): a list of list which maps from
            video_idx and sec_idx to a list of boxes and corresponding labels.

    Returns:
        count (int): total number of used boxes.
    """
    count = 0
    for video_idx, sec_idx, _ in keyframe_indices:
        count += 1
    return count