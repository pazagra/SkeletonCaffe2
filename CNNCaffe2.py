# assumes being a subdirectory of caffe2
CAFFE_MODELS = "model/"
# if you have a mean file, place it in the same dir as the model

import cv2 as cv
import math
from caffe2.proto import caffe2_pb2
import numpy as np
from caffe2.python import core, workspace
from config_reader import config_reader
import util
import os
from scipy.ndimage.filters import gaussian_filter
import time

class skeleton:
    def __init__(self,str):
	# This is to be able to have two instances at the same time
        self.str =str
        workspace.SwitchWorkspace(self.str, True)
        #Inizializate the parameters and the Caffe2 files
        self.param, self.model = config_reader()
        INIT_NET = os.path.join(CAFFE_MODELS, "init_net.pb")
        print 'INIT_NET = ', INIT_NET
        PREDICT_NET = os.path.join(CAFFE_MODELS, "predict_net.pb")
        print 'PREDICT_NET = ', PREDICT_NET
        self.device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)
        self.init_def = caffe2_pb2.NetDef()
        with open(INIT_NET) as f:
            self.init_def.ParseFromString(f.read())
            self.init_def.device_option.CopyFrom(self.device_opts)
            workspace.RunNetOnce(self.init_def.SerializeToString())

        self.net_def = caffe2_pb2.NetDef()
        with open(PREDICT_NET) as f:
            self.net_def.ParseFromString(f.read())
            self.net_def.device_option.CopyFrom(self.device_opts)
            workspace.CreateNet(self.net_def.SerializeToString(),True)

    def draw_skeleton(self, canvas, skel):
        limbSeq = [['chest', 'shoulder right'], ['chest', 'shoulder left'], ['shoulder right', 'arm right'],
                   ['arm right', 'hand right'], ['shoulder left', 'arm left'], ['arm left', 'hand left'],
                   ['chest', 'hip right'], ['hip right', 'knee right'], ['knee right', 'foot right'],
                   ['chest', 'hip left'], ['hip left', 'knee left'], ['knee left', 'foot left'], ['chest', 'face'],
                   ['face', 'eye right'], ['eye right', 'ear right'], ['face', 'eye left'],
                   ['eye left', 'ear left'], ['shoulder right', 'ear right'], ['shoulder left', 'ear left']]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        stickwidth = 4
        for x in range(17):
            index = limbSeq[x]
            if index[0] not in skel.keys() or index[1] not in skel.keys():
                continue
            cur_canvas = canvas.copy()
            Y = [skel[index[0]][1], skel[index[1]][1]]
            X = [skel[index[0]][0], skel[index[1]][0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[x])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas


    def get_skeleton(self,oriImg,scales):
        workspace.SwitchWorkspace(self.str, True)
        #Preprocess the images for the Network input and execute the Net
        test_image = oriImg.copy()
        multiplier = [x * self.model['boxsize'] / oriImg.shape[0] for x in scales]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        for m in range(len(multiplier)):
            data = np.zeros((1, 3, 368, 368))
            scale = multiplier[m]
            imageToTest = cv.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.model['stride'], self.model['padValue'])
            data.resize(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
            data = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
            workspace.FeedBlob('data', data, device_option=self.device_opts)
            workspace.RunNet('')
            heatmap = workspace.FetchBlob('Mconv7_stage6_L2')

            heatmap = np.transpose(np.squeeze(heatmap[0]), (1, 2, 0))
            heatmap = cv.resize(heatmap, (0, 0), fx=self.model['stride'], fy=self.model['stride'], interpolation=cv.INTER_CUBIC)

            heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

            paf = workspace.FetchBlob('Mconv7_stage6_L1')
            paf = np.transpose(np.squeeze(paf[0]), (1, 2, 0))

            paf = cv.resize(paf, (0, 0), fx=self.model['stride'], fy=self.model['stride'], interpolation=cv.INTER_CUBIC)

            paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        #Process the Heatmap and the Paf
        from numpy import ma
        U = paf_avg[:, :, 16] * -1
        V = paf_avg[:, :, 17]
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
        M = np.zeros(U.shape, dtype='bool')
        M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
        U = ma.masked_array(U, mask=M)
        V = ma.masked_array(V, mask=M)

        s = 5

        all_peaks = []
        peak_counter = 0

        for part in range(19 - 1):
            x_list = []
            y_list = []
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > self.param['thre1']))
            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)

        # find connection in the specified sequence, center 29 is in the position 15
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]
        connection_all = []
        special_k = []
        mid_num = 10

        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        vec = np.divide(vec, norm)

                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))

                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        if norm == 0:
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts)
                        else:
                            score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                                0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > self.param['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print "found = 2"
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        deleteIdx = [];
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)

        # VISUALIZACION
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        Positions = ["face", "chest", "shoulder right", "arm right", "hand right", "shoulder left", "arm left",
                     "hand left", "hip right", "knee right", "foot right", "hip left", "knee left", "foot left",
                     "eye right", "eye left", "ear right", "ear left"]
        canvas = test_image  # B,G,R order
        K = {}
        l = []
        if subset.shape[0] == 0:
            print "ERROR"

        for i in range(18):
            for j in range(len(all_peaks[i])):
                l.append(i)
        skel_size = []
        if len(subset) > 1:
            print "Two or more skeletons found, choosing the bigger"
            for n in range(len(subset)):
                x_min = 640
                y_min = 640
                x_max = 0
                y_max = 0
                for i in range(17):
                    index = subset[n][np.array(limbSeq[i]) - 1]
                    if -1 in index:
                        continue
                    Y = candidate[index.astype(int), 0]
                    X = candidate[index.astype(int), 1]
                    x_min = min(X[0], X[1], x_min)
                    y_min = min(Y[0], Y[1], y_min)
                    x_max = max(X[0], X[1], x_max)
                    y_max = max(Y[0], Y[1], y_max)
                dx = x_max - x_min
                dy = y_max - y_min
                size = dx * dy
                skel_size.append(size)
            n = 0
            s = skel_size[n]
            for m in range(len(subset)):
                if s < skel_size[m]:
                    n = m
                    s = skel_size[m]
        else:
            # choosing default skeleton
            n = 0

        stickwidth = 4
        for i in range(17):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            K[Positions[l[index[0].astype(int)]]] = (int(X[0]), int(Y[0]))
            K[Positions[l[index[1].astype(int)]]] = (int(X[1]), int(Y[1]))
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas, K
