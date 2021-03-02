# coding:utf-8
from __future__ import division
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def bbox_iou(bbox1, bbox2):
    # [x1,y1,x2,y2]
    xx1 = np.maximum(bbox1[0], bbox2[0])
    yy1 = np.maximum(bbox1[1], bbox2[1])
    xx2 = np.minimum(bbox1[2], bbox2[2])
    yy2 = np.minimum(bbox1[3], bbox2[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    b1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    b2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = wh / (b1_area + b2_area - wh + 1e-16)

    return iou

def convert_bbox_to_z(bbox):
    # [x1,y1,x2,y2] -> [x,y,s,r]^T
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (bbox[2] + bbox[0]) / 2.
    y = (bbox[3] + bbox[1]) / 2.
    s = w * h
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    # [x,y,s,r] -> [x1,y1,x2,y2]
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    x1 = x[0] - w / 2.
    y1 = x[1] - h / 2.
    x2 = x[0] + w / 2.
    y2 = x[1] + h / 2.

    return np.array([x1, y1, x2, y2]).reshape((1, 4))

class KalmanBoxTracker(object):

    count = 0

    def __init__(self, bbox):
        # 使用初始边界框初始化跟踪器 (定义匀速模型)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # 状态变量7维: [u,v,s,r,u',v',s']--中心坐标xy 面积s 长宽比(即为观测输入) 及各自变化率
        # 观测输入4维
        # 状态转移矩阵F 测量系统参数矩阵H 预测过程噪声矩阵Q 测量过程噪声矩阵R 不确定性协方差矩阵P
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # 对未观测到的初始速度给出高的不确定性
        self.kf.P *= 10.            # 默认定义的协方差矩阵是np.eye(dim_x) 将P中数值乘以10 1000 赋值不确定性
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # bbox->[x,y,s,r]^T 赋值给状态变量前4位
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):

        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 1), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = bbox_iou(det, trk)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_indices = np.zeros(shape=(row_ind.shape[0], 2), dtype=np.int16)
    matched_indices[:, 0] = row_ind
    matched_indices[:, 1] = col_ind

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low iou
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold) or (not int(detections[m[0]][5]) == int(trackers[m[1]][5])):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.types = []
        self.frame_count = 0

    def update(self, dets):
        # dets: [x1,y1,x2,y2,type]
        # trks: [x1,y1,x2,y2,type]
        # ret: [x1,y1,x2,y2,obj_id,type]
        self.frame_count += 1  # 帧计数器
        trks = np.zeros((len(self.trackers), 5))  # 根据当前卡尔曼跟踪器个数创建
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]  # 预测对应物体当前帧bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.types[t]]
            if np.any(np.isnan(pos)):  # 如果预测的bbox为空 删除第t个卡尔曼跟踪器
                to_del.append(t)
        # 删除预测为空跟踪器所在行 trks中存放的是上一帧中被跟踪的所有物体在当前帧预测非空的bbox
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):  # 从跟踪器中删除to_del中的上一帧跟踪器ID
            self.trackers.pop(t)
            self.types.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        # 遍历跟踪器 如果上一帧的t还在当前帧中 说明跟踪器t关联成功 在matched中找到与其关联的检测器d 用d更新卡尔曼跟踪器
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
                self.types[t] = dets[d, :][0][4]

        # 对于新增的未匹配的检测结果 创建初始化跟踪器trk 并传入trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, 0:4])
            self.trackers.append(trk)
            self.types.append(dets[i, :][4])
        i = len(self.trackers)
        for trk in reversed(self.trackers):  # 倒序遍历新的卡尔曼跟踪器集合
            pos = trk.get_state()[0]  # 获取trk跟踪器状态[x1,y1,x2,y2]
            i -= 1
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((pos, [trk.id + 1], [self.types[i]])).reshape(1, -1))
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                self.types.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 6))

def sort_image(sort_calss, out_boxes, out_classes):
    dets = []

    for i in range(len(out_boxes)):
        # boxes: [x1,y1,x2,y2]
        dets.append([out_boxes[i][0], out_boxes[i][1], out_boxes[i][2], out_boxes[i][3], out_classes[i]])

    dets = np.array(dets)
    # update
    trackers = sort_calss.update(dets)

    out_boxes = []
    out_classes = []
    object_id = []

    for d in trackers:
        out_boxes.append(list([int(float(d[0])), int(float(d[1])), int(float(d[2])), int(float(d[3]))]))
        object_id.append(int(d[4]))
        out_classes.append(int(d[5]))

    return np.array(out_boxes), np.array(out_classes), np.array(object_id)
