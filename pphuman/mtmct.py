# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codeop
from pptracking.python.mot.visualize import plot_tracking
from python.visualize import visualize_attr
from scipy.optimize import linear_sum_assignment

import os
import re
import cv2
import gc
import numpy as np
import threading
import copy

try:
    from sklearn import preprocessing
    from sklearn.cluster import AgglomerativeClustering
except:
    print(
        "Warning: Unable to use MTMCT in PP-Human, please install sklearn, for example: `pip install sklearn`"
    )
    pass
import pandas as pd
from tqdm import tqdm
from functools import reduce
import warnings

warnings.filterwarnings("ignore")


def gen_restxt(output_dir_filename, map_tid, cid_tid_dict):
    pattern = re.compile(r"c(\d)_t(\d)")
    f_w = open(output_dir_filename, "w")
    for key, res in cid_tid_dict.items():
        cid, tid = pattern.search(key).groups()
        cid = int(cid) + 1
        rects = res["rects"]
        frames = res["frames"]
        for idx, bbox in enumerate(rects):
            bbox[0][3:] -= bbox[0][1:3]
            fid = frames[idx] + 1
            rect = [max(int(x), 0) for x in bbox[0][1:]]
            if key in map_tid:
                new_tid = map_tid[key]
                f_w.write(
                    str(cid)
                    + " "
                    + str(new_tid)
                    + " "
                    + str(fid)
                    + " "
                    + " ".join(map(str, rect))
                    + "\n"
                )
        # f_w.write('\n')
    print("gen_res: write file in {}".format(output_dir_filename))
    f_w.close()


def get_mtmct_matching_results(pred_mtmct_file, secs_interval=0.5, video_fps=20):
    res = np.loadtxt(pred_mtmct_file)  # 'cid, tid, fid, x1, y1, w, h, -1, -1'
    # print("res:",res)
    camera_ids = list(map(int, np.unique(res[:, 0])))

    res = res[:, :7]
    # each line in res: 'cid, tid, fid, x1, y1, w, h'

    camera_tids = []
    camera_results = dict()
    for c_id in camera_ids:
        camera_results[c_id] = res[res[:, 0] == c_id]
        tids = np.unique(camera_results[c_id][:, 1])
        tids = list(map(int, tids))
        camera_tids.append(tids)

    # select common tids throughout each video
    common_tids = reduce(np.intersect1d, camera_tids)

    # get mtmct matching results by cid_tid_fid_results[c_id][t_id][f_id]
    cid_tid_fid_results = dict()
    cid_tid_to_fids = dict()
    interval = int(secs_interval * video_fps)  # preferably less than 10
    for c_id in camera_ids:
        cid_tid_fid_results[c_id] = dict()
        cid_tid_to_fids[c_id] = dict()
        for t_id in common_tids:
            tid_mask = camera_results[c_id][:, 1] == t_id  # 进行比较 将符合的设为true
            cid_tid_fid_results[c_id][t_id] = dict()

            camera_trackid_results = camera_results[c_id][tid_mask]
            fids = np.unique(camera_trackid_results[:, 2])
            fids = fids[fids % interval == 0]
            fids = list(map(int, fids))
            cid_tid_to_fids[c_id][t_id] = fids

            for f_id in fids:
                st_frame = f_id
                ed_frame = f_id + interval

                st_mask = camera_trackid_results[:, 2] >= st_frame
                ed_mask = camera_trackid_results[:, 2] < ed_frame
                frame_mask = np.logical_and(st_mask, ed_mask)
                cid_tid_fid_results[c_id][t_id][f_id] = camera_trackid_results[
                    frame_mask
                ]
    # print("camera_results:", camera_results)
    # print("cid_tid_fid_res", cid_tid_fid_results)

    return camera_results, cid_tid_fid_results


def save_mtmct_vis_rtps(
    camera_results, video_file, save_dir, idx, writer, multi_res=None
):
    # camera_results: 'cid, tid, fid, x1, y1, w, h'
    camera_ids = list(camera_results.keys())

    capture = cv2.VideoCapture(video_file)
    cid = camera_ids[idx]
    # basename = os.path.basename(video_file)
    video_out_name = "vis_" + "output_rtsp" + str(cid) + ".flv"
    out_path = os.path.join(save_dir, video_out_name)
    print("SaveThead {} Start visualizing output video: {} ".format(idx, out_path))

    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"flv1")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    # while (1):
    # if frame_id % 50 == 0:
    print("SaveThread %d frame id: %d" % (idx, frame_id))
    ret, frame = capture.read()
    frame_id += 1
    frame_results = camera_results[cid][camera_results[cid][:, 2] == frame_id]
    boxes = frame_results[:, -4:]
    ids = frame_results[:, 1]
    # print(ids)
    image = plot_tracking(frame, boxes, ids, frame_id=frame_id, fps=fps)

    # add attr vis
    # print(multi_res)
    # if multi_res:
    #     # print("here")
    #     tid_list = multi_res.keys()  # c0_t1, c0_t2...
    #     all_attr_result = [multi_res[i]["attrs"]
    #                         for i in tid_list]  # all cid_tid result
    #     if any(
    #             all_attr_result
    #     ):  # at least one cid_tid[attrs] is not None will goes to attrs_vis
    #         attr_res = []
    #         cid_str = 'c' + str(cid - 1) + "_"
    #         for k in tid_list:
    #             if not k.startswith(cid_str):
    #                 continue
    #             if (frame_id - 1) >= len(multi_res[k]['attrs']):
    #                 t_attr = None
    #             else:
    #                 t_attr = multi_res[k]['attrs'][frame_id - 1]
    #                 attr_res.append(t_attr)
    #         assert len(attr_res) == len(boxes)
    #         image = visualize_attr(
    #             image, attr_res, boxes, is_mtmct=True)
    # cv2.imshow('Paddle-Pipeline'+str(idx), image)
    # cv2.waitKey(10)
    writer.write(image)
    # writer.release()


def save_mtmct_vis_results(
    camera_results, captures, frames, output_dir, frame_id, multi_res=None
):
    # camera_results: 'cid, tid, fid, x1, y1, w, h'
    camera_ids = list(camera_results.keys())
    # print("came")
    # print(camera_ids)
    # print(camera_results)
    import shutil

    save_dir = os.path.join(output_dir, "mtmct_vis")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for frameId, frame in enumerate(frames):
        print("farmeId %d" % (frameId))
        if len(camera_ids) == 1 & frameId == 1:
            cv2.imshow("Paddle-Pipeline2", frame)
            cv2.waitKey(1)
        else:
            cid = camera_ids[frameId]
            frame_results = camera_results[cid][camera_results[cid][:, 2] == frame_id]
            boxes = frame_results[:, -4:]
            ids = frame_results[:, 1]
            image = plot_tracking(frame, boxes, ids, frame_id=frame_id, fps=30)

            # add attr vis
            if multi_res:
                tid_list = multi_res.keys()  # c0_t1, c0_t2...
                all_attr_result = [
                    multi_res[i]["attrs"] for i in tid_list
                ]  # all cid_tid result
                if any(
                    all_attr_result
                ):  # at least one cid_tid[attrs] is not None will goes to attrs_vis
                    attr_res = []
                    cid_str = "c" + str(cid - 1) + "_"
                    for k in tid_list:
                        if not k.startswith(cid_str):
                            continue
                        if (frame_id - 1) >= len(multi_res[k]["attrs"]):
                            t_attr = None
                        else:
                            t_attr = multi_res[k]["attrs"][frame_id - 1]
                            attr_res.append(t_attr)
                    assert len(attr_res) == len(boxes)
                    image = visualize_attr(image, attr_res, boxes, is_mtmct=True)
            cv2.imshow("Paddle-Pipeline" + str(cid), image)
            cv2.waitKey(1)


def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        np.power(x, 2).sum(axis=1, keepdims=True).repeat(n, axis=1)
        + np.power(y, 2).sum(axis=1, keepdims=True).repeat(m, axis=1).T
    )
    distmat -= np.dot(2 * x, y.T)
    return distmat


def cosine_similarity(x, y, eps=1e-12):
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n, y_n = np.linalg.norm(x, axis=1, keepdims=True), np.linalg.norm(
        y, axis=1, keepdims=True
    )
    x_norm = x / np.maximum(x_n, eps * np.ones_like(x_n))
    y_norm = y / np.maximum(y_n, eps * np.ones_like(y_n))
    sim_mt = np.dot(x_norm, y_norm.T)
    return sim_mt


def get_cosine(x, y, eps=1e-12):
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behavior to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return sim_mt


def get_dist_mat(x, y, func_name="euclidean"):
    if func_name == "cosine":
        dist_mat = get_cosine(x, y)
    elif func_name == "euclidean":
        dist_mat = get_euclidean(x, y)
    print("Using {} as distance function during evaluation".format(func_name))
    return dist_mat


def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][1] == cid_tids[j][1]:
                st_mask[i, j] = 0.0
    return st_mask


def get_sim_matrix_new(cid_tid_dict, cid_tids):
    # Note: camera independent get_sim_matrix function,
    # which is different from the one in camera_utils.py.
    count = len(cid_tids)

    q_arr = np.array([cid_tid_dict[cid_tids[i]]["mean_feat"] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]["mean_feat"] for i in range(count)])
    # compute distmat
    distmat = get_dist_mat(q_arr, g_arr, func_name="cosine")

    # mask the element which belongs to same video
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    sim_matrix = distmat * st_mask
    np.fill_diagonal(sim_matrix, 0.0)
    return 1.0 - sim_matrix


def get_sim_matrix_lost(cid_tid_dict, cid_tids):
    # Note: camera independent get_sim_matrix function,
    # which is different from the one in camera_utils.py.
    count = len(cid_tids)

    q_arr = np.array([cid_tid_dict[cid_tids[i]] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]] for i in range(count)])
    # compute distmat
    distmat = get_dist_mat(q_arr, g_arr, func_name="cosine")

    # mask the element which belongs to same video
    st_mask = np.ones((count, count), dtype=np.float32)
    st_mask = intracam_ignore(st_mask, cid_tids)

    sim_matrix = distmat * st_mask
    np.fill_diagonal(sim_matrix, 0.0)
    return 1.0 - sim_matrix


def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster


def get_cid_tid(cluster_labels, cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster


def get_labels(cid_tid_dict, cid_tids):
    # compute cost matrix between features
    cost_matrix = get_sim_matrix_new(cid_tid_dict, cid_tids)

    # cluster all the features
    cluster1 = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.55,
        affinity="precomputed",
        linkage="complete",
    )
    cluster_labels1 = cluster1.fit_predict(cost_matrix)
    labels = get_match(cluster_labels1)
    """
    输出分类结果,序号即cid_tid_dict中的排序
    eg: cid_tid_dict有: c0_t1,c0_t2, c1_t1,c1_t2
        (其中c0_t1 c1_t1为一组,c0_t2 c1_t2为一组)
        则输出的labels为[[0,2] [1,3]]
    """
    print("cid_tids", cid_tids)
    # print("labels:", labels)

    sub_cluster = get_cid_tid(labels, cid_tids)
    # print("sub_cluster:", sub_cluster)
    return labels, sub_cluster


def get_labels_lost(cid_tid_dict, cid_tids):
    # compute cost matrix between features
    cost_matrix = get_sim_matrix_lost(cid_tid_dict, cid_tids)

    # cluster all the features
    cluster1 = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.4,
        affinity="precomputed",
        linkage="complete",
    )
    cluster_labels1 = cluster1.fit_predict(cost_matrix)
    labels = get_match(cluster_labels1)
    """
    输出分类结果,序号即cid_tid_dict中的排序
    eg: cid_tid_dict有: c0_t1,c0_t2, c1_t1,c1_t2
        (其中c0_t1 c1_t1为一组,c0_t2 c1_t2为一组)
        则输出的labels为[[0,2] [1,3]]
    """
    print("cid_tids", cid_tids)
    # print("labels:", labels)

    sub_cluster = get_cid_tid(labels, cid_tids)
    # print("sub_cluster:", sub_cluster)
    return labels, sub_cluster


def sub_lostcluster(cid_tid_dict):
    """
    cid_tid_dict: all camera_id and track_id
    """
    # get all keys
    cid_tids = sorted([key for key in cid_tid_dict.keys()])

    # cluster all trackid
    clu, cluKey = get_labels_lost(cid_tid_dict, cid_tids)

    # relabel every cluster groups
    new_clu = list()
    for c_list in clu:
        new_clu.append([cid_tids[c] for c in c_list])
    # print("new_clu:", new_clu)
    cid_tid_label = dict()
    for i, c_list in enumerate(new_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    # print("cid_tid_label", cid_tid_label)
    return cid_tid_label, cluKey


def sub_cluster(cid_tid_dict):
    """
    cid_tid_dict: all camera_id and track_id
    """
    # get all keys
    cid_tids = sorted([key for key in cid_tid_dict.keys()])

    # cluster all trackid
    clu, cluKey = get_labels(cid_tid_dict, cid_tids)

    # relabel every cluster groups
    new_clu = list()
    for c_list in clu:
        new_clu.append([cid_tids[c] for c in c_list])
    # print("new_clu:", new_clu)
    cid_tid_label = dict()
    for i, c_list in enumerate(new_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
    # print("cid_tid_label", cid_tid_label)
    return cid_tid_label, cluKey


def distill_idfeat(mot_res):
    qualities_list = mot_res["qualities"]
    feature_list = mot_res["features"]
    rects = mot_res["rects"]

    qualities_new = []
    feature_new = []
    # filter rect less than 100*20
    for idx, rect in enumerate(rects):
        conf, xmin, ymin, xmax, ymax = rect[0]
        if (xmax - xmin) * (ymax - ymin) and (xmax > xmin) > 2000:
            qualities_new.append(qualities_list[idx])
            feature_new.append(feature_list[idx])
    # take all features if available rect is less than 2
    if len(qualities_new) < 2:
        qualities_new = qualities_list
        feature_new = feature_list

    # if available frames number is more than 200, take one frame data per 20 frames
    skipf = 1
    if len(qualities_new) > 20:
        skipf = 2
    quality_skip = np.array(qualities_new[::skipf])
    feature_skip = np.array(feature_new[::skipf])

    # sort features with image qualities, take the most trustworth features
    topk_argq = np.argsort(quality_skip)[::-1]
    if (quality_skip > 0.6).sum() > 1:
        topk_feat = feature_skip[topk_argq[quality_skip > 0.6]]
    else:
        topk_feat = feature_skip[topk_argq]

    # get final features by mean or cluster, at most take five
    mean_feat = np.mean(topk_feat[:5], axis=0)
    return mean_feat


def res2dict(multi_res, captures, frameId):
    cid_tid_dict = {}
    crame0 = {}
    crame1 = {}
    lastCidTid = []
    # lastCidTid = list(captures["crame0all"].keys())
    for cid, c_res in enumerate(multi_res):
        for tid, res in c_res.items():
            key = "c" + str(cid) + "_t" + str(tid)
            # print(key)
            if key not in cid_tid_dict:
                if len(res["features"]) == 0:
                    continue
                cid_tid_dict[key] = res

                # print(res)
                cid_tid_dict[key]["mean_feat"] = distill_idfeat(res)
            if cid == 0:
                crame0[key] = cid_tid_dict[key]["mean_feat"]
                if frameId > 4:
                    if key in list(captures[frameId - 3]["crame0"].keys()):
                        captures["crame0all"][key] = captures[frameId - 3]["crame0"][
                            key
                        ]
            else:
                crame1[key] = cid_tid_dict[key]["mean_feat"]
                if frameId > 4:
                    if key in list(captures[frameId - 3]["crame1"].keys()):
                        captures["crame1all"][key] = captures[frameId - 3]["crame1"][
                            key
                        ]
            captures[frameId] = {
                "crame0": copy.deepcopy(crame0),
                "crame1": copy.deepcopy(crame1),
            }

            # print("len mean feat", len(cid_tid_dict[key]["mean_feat"]))
    # print(len(cid_tid_dict))
    return cid_tid_dict


class DeepSortTracker:
    def __init__(self):
        self.tracks = []
        self.track_id_counter = 0

    def update(self, detections):
        # 如果没有跟踪目标，直接创建新的跟踪
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks.append(Track(detection, self.track_id_counter))
                self.track_id_counter += 1
            return self.tracks

        # 计算跟踪目标和检测目标之间的距离
        track_features = np.array([track.features for track in self.tracks])
        detection_features = np.array(
            [detection["features"] for detection in detections]
        )
        distance_matrix = np.linalg.norm(
            track_features[:, np.newaxis] - detection_features, axis=2
        )
        # 使用匈牙利算法进行数据关联
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # 更新跟踪目标的状态和位置
        for i, track in enumerate(self.tracks):
            if i not in row_ind:
                track.mark_missed()

        for row, col in zip(row_ind, col_ind):
            self.tracks[row].update(detections[col])

        # 创建新的跟踪目标
        unmatched_detections = set(range(len(detections))) - set(col_ind)
        for detection_idx in unmatched_detections:
            self.tracks.append(Track(detections[detection_idx], self.track_id_counter))
            self.track_id_counter += 1

        # 移除已经结束的跟踪目标
        self.tracks = [track for track in self.tracks if not track.is_lost()]
        return self.tracks


class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.features = detection["features"]
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0

    def update(self, detection):
        self.features = detection["features"]
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0

    def mark_missed(self):
        self.consecutive_invisible_count += 1

    def is_lost(self):
        return self.consecutive_invisible_count > 5  # 根据需要调整阈值


def mtmct_process(
    multi_res, frames, frame_id, captures, mtmct_vis=True, output_dir="output"
):
    file = open("output1.txt", "a")
    print("**running mtmct**", frame_id, file=file)

    cid_tid_dict = res2dict(multi_res, captures, frame_id)
    print("captures0", captures["crame0all"].keys(), file=file)
    print("captures1", captures["crame1all"].keys(), file=file)
    if frame_id > 30:
        captures.pop(frame_id - 30)
    # print("captures", captures.keys())

    # print(cid_tid_dict)
    if len(cid_tid_dict) < 2:
        print("no tracking result found, mtmct will be skiped.")
        return
    # 1. 初步匹配,若两镜头中有相同对象将在这一步匹配
    lostCrame0 = dict()
    lostCrame1 = dict()
    map_tid, cluKeys = sub_cluster(cid_tid_dict)
    for cluKey in cluKeys:
        numCT = re.findall(r"\d+", cluKey[0])
        numCT = [int(num) for num in numCT]
        print("cluKey", cluKey, file=file)
        if len(cluKey) > 1:
            for Key in cluKey:
                map_tid[Key] = numCT[-1]
        else:
            if numCT[0] == 0:
                lostCrame0[cluKey[0]] = captures[frame_id]["crame0"][cluKey[0]]
            else:
                lostCrame1[cluKey[0]] = captures[frame_id]["crame1"][cluKey[0]]
            map_tid[cluKey[0]] = numCT[-1]
    # 2.获取未匹配成功的对象
    print("map_tid", map_tid)
    print("lostCrame0", lostCrame0.keys(), file=file)
    print("lostCrame1", lostCrame1.keys(), file=file)
    crame0lostTrackfeats = np.array(list(lostCrame0.values()))
    crame0lostKeys = list(lostCrame0.keys())
    crame1lostTrackfeats = np.array(list(lostCrame1.values()))
    crame1lostKeys = list(lostCrame1.keys())
    crame0Detectionfeats = np.array(list(captures["crame0all"].values()))
    crame0DetectionKeys = list(captures["crame0all"].keys())
    crame1Detectionfeats = np.array(list(captures["crame1all"].values()))
    crame1DetectionKeys = list(captures["crame1all"].keys())

    # 3.1 将c1中未匹配的再进行匹配
    if crame1lostTrackfeats.size > 0 and crame0Detectionfeats.size > 0:
        crame1lostDistance_matrix = np.linalg.norm(
            crame1lostTrackfeats[:, np.newaxis] - crame0Detectionfeats, axis=2
        )
        print("crame1_distance_matrix", crame1lostDistance_matrix, file=file)
        masks = np.all(crame1lostDistance_matrix > 25, axis=1)
        print("mask", masks, file=file)
        crame1lostDistance_matrix = crame1lostDistance_matrix[~masks]
        print("newdistance_matrix", crame1lostDistance_matrix, file=file)
        newTrackKeys = []
        # 选出匹配的track
        for i, mask in enumerate(masks):
            # print(i)
            if ~mask:
                newTrackKeys.append(crame1lostKeys[i])
        if crame1lostDistance_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(crame1lostDistance_matrix)
            print("row_ind", row_ind)
            print("col_ind", col_ind)
            # 若两镜头有相关的将ID设为一致
            for row in row_ind:
                print("row", row)
                numDetct = re.findall(r"\d+", crame0DetectionKeys[col_ind[row]])
                numDetct = [int(num) for num in numDetct]
                map_tid[newTrackKeys[row]] = numDetct[-1]
                captures["lostfindID"][newTrackKeys[row]] = numDetct[-1]

    # 3.2 将c0中未匹配的再进行匹配
    if crame0lostTrackfeats.size > 0 and crame1Detectionfeats.size > 0:
        crame0lostDistance_matrix = np.linalg.norm(
            crame0lostTrackfeats[:, np.newaxis] - crame1Detectionfeats, axis=2
        )
        print("crame0_distance_matrix", crame0lostDistance_matrix, file=file)
        masks = np.all(crame0lostDistance_matrix > 24.5, axis=1)
        print("mask", masks, file=file)
        crame0lostDistance_matrix = crame0lostDistance_matrix[~masks]
        print("newdistance_matrix", crame0lostDistance_matrix, file=file)
        newTrackKeys = []
        # 选出匹配的track
        for i, mask in enumerate(masks):
            # print(i)
            if ~mask:
                newTrackKeys.append(crame0lostKeys[i])
        if crame0lostDistance_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(crame0lostDistance_matrix)
            print("row_indc0", row_ind)
            print("col_indc0", col_ind)
            # 若两镜头有相关的将ID设为一致
            for row in row_ind:
                print("row", row)
                numDetct = re.findall(r"\d+", crame1DetectionKeys[col_ind[row]])
                numDetct = [int(num) for num in numDetct]
                if crame1DetectionKeys[col_ind[row]] in captures["lostfindID"]:
                    map_tid[newTrackKeys[row]] = captures["lostfindID"][
                        crame1DetectionKeys[col_ind[row]]
                    ]
                    captures["lostfindID"][newTrackKeys[row]] = map_tid[
                        newTrackKeys[row]
                    ]
                else:
                    map_tid[newTrackKeys[row]] = numDetct[-1]

    print("lostfindID", captures["lostfindID"], file=file)
    for key in captures["lostfindID"]:
        if key in map_tid:
            map_tid[key] = captures["lostfindID"][key]
    print("map_tid(after)", map_tid, file=file)

    # 计算跟踪目标和检测目标之间的距离
    # if frame_id > 30:
    #     lost_cid_tid = {}
    #     track_features = np.array(list(captures[frame_id]["crame1"].values()))
    #     trackKeys = list(captures[frame_id]["crame1"].keys())
    #     lostTrackkeys = []
    #     lostTrackfeatures = np.array([])
    #     detection_features = np.array(list(captures[frame_id]["crame0"].values()))
    #     detectionKeys = list(captures[frame_id]["crame0"].keys())
    #     # lostDetectionfeatures = np.array(list(captures["crame0all"].values()))
    #     lostDetectionfeatures = np.array(list(captures["crame0all"].values()))
    #     lostDetectionkeys = list(captures["crame0all"].keys())
    #     print("detection_features", detection_features.shape, detection_features.size)
    #     print("track_features", track_features.shape, track_features.size)

    # if track_features.size > 0 and detection_features.size > 0:
    #     distance_matrix = np.linalg.norm(
    #         track_features[:, np.newaxis] - detection_features, axis=2
    #     )
    #     print("distance_matrix", distance_matrix, file=file)
    #     masks = np.all(distance_matrix > 11, axis=1)
    #     print("mask", masks, file=file)

    #     # 使用布尔索引删除满足条件的行,距离太大认为不相关
    #     distance_matrix = distance_matrix[~masks]
    #     print("newdistance_matrix", distance_matrix, file=file)

    #     newTrackKeys = []
    #     for i, mask in enumerate(masks):
    #         # print(i)
    #         if ~mask:
    #             newTrackKeys.append(trackKeys[i])
    #         else:
    #             lostTrackkeys.append(trackKeys[i])
    #     print("new", newTrackKeys, lostTrackkeys, file=file)
    #     print("old", detectionKeys, file=file)
    #     # 当距离矩阵存在时,使用匈牙利算法进行数据关联
    #     if distance_matrix.size > 0:
    #         row_ind, col_ind = linear_sum_assignment(distance_matrix)
    #         print("row_ind", row_ind)
    #         print("col_ind", col_ind)

    #         # 若两镜头有相关的将ID设为一致
    #         for row in row_ind:
    #             print("row", row)
    #             numDetct = re.findall(r"\d+", detectionKeys[col_ind[row]])
    #             numDetct = [int(num) for num in numDetct]
    #             map_tid[newTrackKeys[row]] = numDetct[-1]
    #         print("map_tid(after)", map_tid, file=file)
    #     lostTrackfeatures = track_features[masks]
    #     # 收集未匹配的目标再次匹配
    #     frame = 0
    #     while frame < 2 and lostTrackfeatures.size > 0:
    #         frame = frame + 1
    #         # lostDetectionfeatures = np.array(
    #         #     list(captures[frame_id - frame]["crame0"].values())
    #         # )
    #         # lostDetectionkeys = list(captures[frame_id - frame]["crame0"].keys())
    #         lostDistance_matrix = np.linalg.norm(
    #             lostTrackfeatures[:, np.newaxis] - lostDetectionfeatures, axis=2
    #         )
    #         print(
    #             "frame and key",
    #             frame,
    #             captures[frame_id - frame]["crame0"].keys(),
    #             file=file,
    #         )
    #         print("lostDistance_matrix", lostDistance_matrix, file=file)
    #         lostMasks = np.all(lostDistance_matrix > 23, axis=1)
    #         print("lostMasks", lostMasks, file=file)
    #         lostDistance_matrix = lostDistance_matrix[~lostMasks]
    #         lostTrackfeatures = lostTrackfeatures[lostMasks]
    #         newlostTrackkeys = []
    #         for i, mask in enumerate(lostMasks):
    #             if ~mask:
    #                 newlostTrackkeys.append(lostTrackkeys[i])
    #         # 当丢失距离矩阵存在时,使用匈牙利算法进行数据关联
    #         if lostDistance_matrix.size > 0:
    #             lostRow_ind, lostCol_ind = linear_sum_assignment(
    #                 lostDistance_matrix
    #             )
    #             print("lostRow_ind", lostRow_ind, file=file)
    #             print("lostCol_ind", lostCol_ind, file=file)

    #             # 若两镜头有相关的将ID设为一致
    #             for row in lostRow_ind:
    #                 numDetct = re.findall(
    #                     r"\d+", lostDetectionkeys[lostCol_ind[row]]
    #                 )
    #                 numDetct = [int(num) for num in numDetct]
    #                 map_tid[newlostTrackkeys[row]] = numDetct[-1]
    #                 captures["lostfindID"][newlostTrackkeys[row]] = numDetct[-1]
    #         for key in captures["lostfindID"]:
    #             if key in map_tid:
    #                 map_tid[key] = captures["lostfindID"][key]
    #         print("map_tid(afterafter)", map_tid, file=file)
    #         print("lostTrack", newlostTrackkeys)

    # print("c1", trackKeys[row_ind[0]])
    # print("c0", detectionKeys[col_ind[0]])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pred_mtmct_file = os.path.join(output_dir, "mtmct_result.txt")
    gen_restxt(pred_mtmct_file, map_tid, cid_tid_dict)

    if mtmct_vis:
        camera_results, cid_tid_fid_res = get_mtmct_matching_results(pred_mtmct_file)

        save_mtmct_vis_results(
            camera_results,
            captures,
            frames,
            output_dir=output_dir,
            frame_id=frame_id,
            multi_res=cid_tid_dict,
        )
