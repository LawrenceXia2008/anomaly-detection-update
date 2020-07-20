# coding=UTF-8
import pandas as pd
import math
import numpy as np
import datetime

from sklearn.metrics import roc_curve, auc, average_precision_score, recall_score, precision_score, f1_score
from sklearn import preprocessing
from scipy.spatial.distance import minkowski
import heapq



def ewma_vectorized(data, alpha, offset=None, dtype='float64', order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    row_size = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype='float64', order='C', out=None):
    """
    Calculates the exponential moving average over a given axis.

    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out


def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon) / np.log(1 - alpha)) + 1


def window_size(alpha, sum_proportion):
    # solve (1-alpha)**window_size = (1-sum_proportion) for window_size
    return int(np.log(1 - sum_proportion) / np.log(1 - alpha))


def ewma_vectorized_safe(data, alpha, row_size=None, dtype='float64', order='C', out=None):
    """
    The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float
    else:
        dtype = np.dtype(dtype)

    row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C', out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C', out=out[-trailing_n:])
    return out


def p_power_error(y_true, y_pred):
    return np.mean(np.power(y_pred - y_true, 6), axis=-1)


def dataset_norm_minmax(train_set, test_set):
    
    train_set1 = []
    test_set1 = []
    for i in range(train_set.shape[1]):
        min_max_scaler = preprocessing.MinMaxScaler()
        c = min_max_scaler.fit_transform(train_set[:, i].reshape(-1, 1)).reshape(-1)
        d = min_max_scaler.transform(test_set[:, i].reshape(-1, 1)).reshape(-1)
        train_set1.append(c)
        test_set1.append(d)
    return np.array(train_set1, dtype=np.float64).T, np.array(test_set1, dtype=np.float64).T


# 保证进入deep_processing之前数据concatenate为2维（point_id,dim）
# 顺序是先进行取范数再进行指数滑动平均
def deep_processing(y_true, y_pred, _train_, predict_win, power):
    
    window = predict_win
    sum_proportion = .5
    
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    powered_error = np.mean(np.power(y_pred - y_true, power), axis=-1)
    
    if _train_ == 1:
        
        threshold = np.percentile(
            ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None), 100)
        return threshold
    else:
        return ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None)


def deep_processing_abs(y_true, y_pred, _train_, predict_win):
    
    window = predict_win
    sum_proportion = .5
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    powered_error = np.mean(abs(y_pred - y_true), axis=-1)
    if _train_ == 1:
        threshold = np.percentile(
            ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None), 99)
        return threshold
    else:
        return ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None)


def get_anomaly(train_true, train_predict, test_true, test_predict, predict_win):
    
    scores2 = deep_processing(test_true, test_predict, 0, predict_win, 2)
    threshold = deep_processing(train_true, train_predict, 1, predict_win, 2)
    return zip(scores2 >= threshold, scores2)


def report_evaluation_metrics(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Precision: {0:0.2f}'.format(precision))
    print('Recall: {0:0.2f}'.format(recall))
    print('F1: {0:0.2f}'.format(f1))


def get_label(train_true, train_predict, test_true, test_predict, predict_win):
    pred_label = []
    anomaly_information = get_anomaly(train_true, train_predict, test_true, test_predict, predict_win)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        predicted_label = 1 if is_anomaly else 0
        pred_label.append(predicted_label)
    return np.array(pred_label)



def error_weighting(train_true, train_predict, test_true, test_predict, power, predict_win, norm_or_not, epsilon=None):
    # 在变量上取了平均
    # print(np.mean(np.power(train_true - train_predict, power), axis=1))

    # min_max_scaler = preprocessing.MinMaxScaler()
    # normed_train_error = preprocessing.minmax_scale(np.power(train_true - train_predict, power),axis=1)
    # normed_test_error=preprocessing.minmax_scale(np.power(test_true - test_predict, power),axis=1)
    '''ewma初始化'''
    window = 240
    sum_proportion = .5
    # 取自TEP论文中的half-life标准，因此我们取为
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    
    # no_normed_train_error=np.power(np.power(np.abs(train_true-train_predict), power),1/power)
    no_normed_train_error = np.power(np.abs(train_true - train_predict), power)
    # normed_train_error=preprocessing.scale(no_normed_train_error,axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    normed_train_error = preprocessing.minmax_scale(no_normed_train_error, axis=0)

    # normed_test_error = preprocessing.scale(np.power(test_true - test_predict, power),axis=1)
    
    error_avgintime = np.mean(normed_train_error, axis=0)
    error_avgintime = preprocessing.minmax_scale(error_avgintime)
    weight = []
    
    for i in range(error_avgintime.shape[0]):
        if epsilon != None:
            weight.append(1 / np.sum(
                np.power((error_avgintime[i] + np.std(error_avgintime)) / (error_avgintime + np.std(error_avgintime)),
                         1 / epsilon)))
        else:
            weight.append(1 / np.sum(
                np.power((error_avgintime[i] + np.std(error_avgintime)) / (error_avgintime + np.std(error_avgintime)),
                         1 / (power - 1))))
            # weight.append(1/np.sum(np.power((error_avgintime[i]+100000)/(error_avgintime+100000),1/(power-1))))

    # weight=np.array(weight)
    # weight=np.array([1/51 for i in range(51)])
    if epsilon == None:
        weight1 = np.power(np.array(weight), power)
    else:
        weight1 = np.power(np.array(weight), epsilon)
    
    # neg_exp_error=np.exp(-error_avgintime)
    # weight=neg_exp_error/np.sum(neg_exp_error)

    print("weight" + str(weight))
    # a = 2
    # actuator_wgt = [1, 1, a, a, a, 1, 1, 1, 1, a
    #     , a, a, a, a, a, a, 1, 1, 1, a
    #     , a, a, a, a, a, 1, 1, 1, a, a
    #     , a, a, a, a, 1, 1, 1, 1, 1, 1
    #     , 1, 1, a, a, 1, 1, 1, 1, a, a, a]

    # actuator_wgt = np.array(actuator_wgt)
    # actuator_wgt_matrix_te = [actuator_wgt for i in range(test_predict.shape[0])]
    # actuator_wgt_matrix_te = np.array(actuator_wgt_matrix_te)
    # test_elemwise = np.abs(test_true - test_predict) * actuator_wgt_matrix_te
    test_elemwise = np.abs(test_true - test_predict)
   
    # 矩阵乘法求加权平均
    # if norm_or_not==False:
    #     error_out = np.power(np.dot(np.power(np.abs(test_true - test_predict), power), weight1.reshape(-1, 1)), 1 / power)
    # else:
    #     error_out = np.power(np.dot(np.power(preprocessing.scale(np.abs(test_true - test_predict)), power), weight1.reshape(-1, 1)), 1 / power)

    if norm_or_not == False:
        error_out = np.power(np.dot(np.power(test_elemwise, power), weight1.reshape(-1, 1)), 1 / power)
    else:
        error_out = np.power(np.dot(np.power(preprocessing.scale(test_elemwise), power), weight1.reshape(-1, 1)),
                             1 / power)
    
    error_out = error_out.reshape(-1)
    error_out_ewma = ewma_vectorized_safe(error_out, alpha, row_size=None, dtype='float64', order='C', out=None)
    
    

    # actuator_wgt = np.array(actuator_wgt)
    # actuator_wgt_matrix_tr = [actuator_wgt for i in range(train_predict.shape[0])]
    # actuator_wgt_matrix_tr = np.array(actuator_wgt_matrix_tr)
    # train_elemwise = np.abs(train_true - train_predict) * actuator_wgt_matrix_tr

    train_elemwise = np.abs(train_true - train_predict)
    # if norm_or_not==False:
    #     error_thre = np.power(np.dot(np.power(np.abs(train_true - train_predict), power), weight1.reshape(-1, 1)),1/power)
    # else:
    #     error_thre = np.power(np.dot(np.power(preprocessing.scale(np.abs(train_true - train_predict)), power), weight1.reshape(-1, 1)), 1 / power)

    if norm_or_not == False:
        error_thre = np.power(np.dot(np.power(train_elemwise, power), weight1.reshape(-1, 1)), 1 / power)
    else:
        error_thre = np.power(np.dot(np.power(preprocessing.scale(train_elemwise), power), weight1.reshape(-1, 1)),
                              1 / power)
    
    error_thre = error_thre.reshape(-1)
    print("error_training: " + str(np.mean(error_thre)))
    error_thre_ewma = ewma_vectorized_safe(error_thre, alpha, row_size=None, dtype='float64', order='C', out=None)
    
    return error_out, error_thre_ewma


def error_weighting_second(train_true, train_predict, test_true, test_predict, power, predict_win, norm_or_not):
    
    # print(np.mean(np.power(train_true - train_predict, power), axis=1))

    # min_max_scaler = preprocessing.MinMaxScaler()
    # normed_train_error = preprocessing.minmax_scale(np.power(train_true - train_predict, power),axis=1)
    # normed_test_error=preprocessing.minmax_scale(np.power(test_true - test_predict, power),axis=1)
    window = predict_win
    sum_proportion = .5
    
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)

    normed_train_error = preprocessing.scale(np.power(train_true - train_predict, power), axis=1)
    normed_test_error = preprocessing.scale(np.power(test_true - test_predict, power), axis=1)
   
    error_avgintime = np.mean(normed_train_error, axis=0)
    neg_exp_error = np.exp(-error_avgintime)
    weight = neg_exp_error / np.sum(neg_exp_error)
    print(weight)
    
    if norm_or_not == False:
        error_out = np.power(np.dot(np.power(np.abs(test_true - test_predict), power), weight.reshape(-1, 1)),
                             1 / power)
    else:
        error_out = np.power(
            np.dot(np.power(preprocessing.scale(np.abs(test_true - test_predict)), power), weight.reshape(-1, 1)),
            1 / power)
    error_out = error_out.reshape(-1)
    error_out_ewma = ewma_vectorized_safe(error_out, alpha, row_size=None, dtype='float64', order='C', out=None)
    
    
    if norm_or_not == False:
        error_thre = np.power(np.dot(np.power(np.abs(train_true - train_predict), power), weight.reshape(-1, 1)),
                              1 / power)
    else:
        error_thre = np.power(
            np.dot(np.power(preprocessing.scale(np.abs(train_true - train_predict)), power), weight.reshape(-1, 1)),
            1 / power)
    error_thre = error_thre.reshape(-1)
    return error_out_ewma, error_thre


def error_no_weighting(train_true, train_predict, test_true, test_predict, power, predict_win, norm_or_not):
    '''ewma初始化'''
    window = 120
    sum_proportion = .5
   
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    
    
    
    if norm_or_not == False:
        error_out = np.power(np.mean(np.power(np.abs(test_true - test_predict), power), axis=1), 1 / power)
    else:
        # error_out = np.mean(np.power(np.power(preprocessing.scale(np.abs(test_true - test_predict)), power), 1 / power),axis=1)
        error_out = np.power(np.mean(np.power(preprocessing.scale(np.abs(test_true - test_predict)), power), axis=1),
                             1 / power)
    error_out_ewma = ewma_vectorized_safe(error_out, alpha, row_size=None, dtype='float64', order='C', out=None)
   
    if norm_or_not == False:
        # error_thre = np.mean(np.power(np.power(np.abs(train_true - train_predict), power), 1 / power),axis=1)
        error_thre = np.power(np.mean(np.power(np.abs(train_true - train_predict), power), axis=1), 1 / power)
    else:
        error_thre = np.mean(
            np.power(np.power(preprocessing.scale(np.abs(train_true - train_predict)), power), 1 / power), axis=1)
    print("error_training" + str(np.mean(error_thre)))
    error_thre_ewma = ewma_vectorized_safe(error_thre, alpha, row_size=None, dtype='float64', order='C', out=None)
    

    return error_out, error_thre
   


#######################################################################  threshold  ##############################################################################
def interval_generation(sequence, proportion=None):
    if proportion == None:
        flag0 = 0
        flage = 0
        interval = []

        for i in range(len(sequence)):
            if i != len(sequence) - 1:
                if sequence[i + 1] - sequence[i] > 1:
                    flage = i
                    interval.append((sequence[flag0], sequence[flage]))
                    flag0 = i + 1
            else:
                interval.append((sequence[flag0], sequence[i]))
        return interval
    else:
        flag0 = 0
        flage = 0
        interval = []

        for i in range(len(sequence)):
            if i != len(sequence) - 1:
                if sequence[i + 1] - sequence[i] > 1:
                    flage = i
                    add_len = int((sequence[flage] - sequence[flag0] + 1) * proportion)
                    if sequence[flage] + add_len < sequence[i + 1] - 2:
                        interval.append((sequence[flag0], sequence[flage] + add_len))
                    else:
                        interval.append((sequence[flag0], sequence[i + 1] - 2))
                    flag0 = i + 1
            else:
                interval.append((sequence[flag0], sequence[i]))
        return interval


'''yuting   mergeinterval'''
def mergeIntervals(attacks):
    attacks = sorted(attacks)
    merged_attacks = []
    merged_start = None
    merged_end = None
    for aidx in range(len(attacks)):
        if merged_start is None:
            merged_start = attacks[aidx][0]
        if merged_end is None:
            merged_end = attacks[aidx][1]
        else:  # can we merge it with the previous?
            if attacks[aidx][0] - merged_end <= 50:  # TODO - some generic criteria
                merged_end = attacks[aidx][1]
            else:  # we are far away, add the previous and start a new one
                merged_attacks.append((merged_start, merged_end))
                merged_start = attacks[aidx][0]
                merged_end = attacks[aidx][1]

    # when we get to the end of the loop, we have the last one to add
    if merged_start is not None:
        merged_attacks.append((merged_start, merged_end))
    return merged_attacks


# threshold_finetuning1
def threshold_finetuning_for_outlier1(error_out_ewma, error_thre_ewma, number, scope0, scopee, ground_truth,
                                      window, proportion=0.0,fine_or_not=False):#proportion=0.0
    def bianchenglabel(some_interval):
        label = np.zeros((ground_truth.shape[0],), dtype=np.int)

        for i in range(len(some_interval)):
            for j in range(some_interval[i][1] - some_interval[i][0] + 1):
                label[some_interval[i][0] + j] = 1
        return label

    
    seq_ground = np.argwhere(ground_truth == 1).reshape(-1)
    ground_truth_interval = interval_generation(seq_ground, proportion)
    ground_truth = bianchenglabel(ground_truth_interval)

    scope_str = scope0
    scope_end = scopee

    scope = np.linspace(scope_str, scope_end, number, endpoint=True)

    thre_x = []
    precision = []
    recall = []
    f1 = []
    tpr = []
    fpr = []
    percents = scope
    '''yuting merge_interval'''
    for i in range(len(scope)):
        thre_x.append(scope[i])
        anomaly_information = zip(error_out_ewma >= scope[i], error_out_ewma)

        attacks = []
        start = -1
        end = -1
        
        print('error_out_ewma',len(error_out_ewma))
        for d_idx in range(len(error_out_ewma)):
            # look at the largest difference
            if error_out_ewma[d_idx] > scope[i]:
                if start == -1:  # start new attack candidate
                    start = d_idx
                    end = d_idx
                else:  # extend the existing
                    end = d_idx
            else:
                if end != -1:
                    # ignore changes that don't last
                    if end - start > window:
                        attacks.append((start, end))
                    start = -1
                    end = -1
        if end != -1 and end - start > window:
            attacks.append((start, end))

        #print("attacks!!!!!!!!",attacks)
        attacks=mergeIntervals(attacks)
        #print("attacks!!!!!!!!",attacks)
       
        pred_labels = np.zeros(ground_truth.shape)
        for k in range(len(attacks)):
            for j in range(attacks[k][0], attacks[k][1] + 1):

                pred_labels[j] = 1

        '''
        delete long attrack
        '''
        # index = range(225919, 263729)
        # new_pred_labels = np.delete(pred_labels, index)
        # new_ground_truth = np.delete(ground_truth, index)

        '''
        only long attrack
        '''
        # index_start = 225919
        # index_end = 263730#
        # new_pred_labels = pred_labels[index_start:index_end]
        # new_ground_truth = ground_truth[index_start:index_end]


        precision.append(precision_score(list(ground_truth), pred_labels, labels=[0, 1], pos_label=1))
        recall.append(recall_score(list(ground_truth), pred_labels, labels=[0, 1], pos_label=1))
        f1.append(f1_score(list(ground_truth), pred_labels, labels=[0, 1], pos_label=1))
        ##########  去除掉使得F1=0.0的无效值  ##########  
        if f1[i]!=0.0:
            """
            计算FPR TPR
            """
            fpr_i, tpr_i, thresholds_i = roc_curve(ground_truth, pred_labels, pos_label=1)
            # print("!!!!!!!计算FPR TPR!!!!!!!!!!")
            # print("fpr,tpr, thresholds 的 shape: ", fpr_i.shape, tpr_i.shape, thresholds_i.shape)
            # print("fpr:  ",fpr_i)
            # print("tpr:  ",tpr_i)
            # print("thre", thresholds_i)
            fpr_ii = fpr_i[1]
            tpr_ii = tpr_i[1]
            fpr.append(fpr_ii)
            tpr.append(tpr_ii)
        else:
            tpr_ii=None
            fpr_ii=None

        print("the" + str(i) + "th search result :")
        print("when Threshold is" + str(thre_x[i]) )
        print("fpr is :" + str(fpr_ii))
        print("tpr is  :" + str(tpr_ii))
        print("Precision is :" + str(precision[i]))
        print("Recall is :" + str(recall[i]))
        print("F1 is :" + str(f1[i]) + "\n\n\n")
        
    
    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    best = np.argwhere(f1 == np.sort(f1)[-1])[0][0]

    best_dict = {'percents': percents[best] / np.percentile(error_thre_ewma, 100),
                 'thre_x': thre_x[best],
                 'recall': recall[best],
                 'precision': precision[best],
                 'f1': f1[best]
                 }
    print("percentile" + str(np.percentile(error_thre_ewma, 100)))
    print("the best reslut :" + str(best_dict))
    #################### 完成上述的 thres，F1，Precision，Recall的计算后，如果roc_curve为TRUE，则进行roc_curve的计算############
    # error_out_ewma 是评分，
    
    
    

    return percents, thre_x, precision, recall, f1, fpr,tpr



def threshold_finetuning_for_consecutive1(error_out_ewma, error_thre_ewma, number, scope0, scopee, ground_truth,
                                          input_win, fine_or_not, filter_or_not=False):

    scope_str = scope0
    scope_end = scopee
    scope = np.linspace(scope_str, scope_end, number, endpoint=True)
    percents = scope

    thre_x = []
    precision = []
    recall = []
    f1 = []
    TP = []
    FP = []
    FN = []

    for i in range(len(scope)):
        thre_x.append(scope[i])
        print("the" + str(i) + "th search result :")
        
        Precision, Recall, F1, tp, fp, fn = consecutive_res_analysis(scope[i], error_out_ewma, ground_truth, input_win)
        precision.append(Precision)
        recall.append(Recall)
        f1.append(F1)
        TP.append(len(tp))
        FP.append(len(fp))
        FN.append(len(fn))
    best = np.argwhere(f1 == np.sort(f1)[-1])[0][0]
    a_a, b_b, c_c, best_tp_interval, best_fp_interval, best_fn_interval = consecutive_res_analysis(thre_x[best],
                                                                                                   error_out_ewma,
                                                                                                   ground_truth,
                                                                                                   input_win)
    best_dict = {
                 'thres': thre_x[best],
                 'recall': recall[best],
                 'precision': precision[best],
                 'f1': f1[best]}
    print("percentile" + str(np.percentile(error_thre_ewma, 100)))
    print("the best results" + str(best_dict) + "\n，FP: "+ str(FP[best]) + "\n" + "FN: " + str(
        FN[best]) + "\n" + "TP: " + str(TP[best]) + "\n")
    if filter_or_not == False:
        return percents, thre_x, precision, recall, f1
    else:
        return best_tp_interval, best_fp_interval, best_fn_interval, percents[best], thre_x[best]


def consecutive_res_analysis(threshold, error_out_ewma, ground_truth, offset_win,
                             proportion=0.2):

    def bianchenglabel(some_interval):
        label = np.zeros((ground_truth.shape[0],), dtype=np.int)
        for i in range(len(some_interval)):
            for j in range(some_interval[i][1] - some_interval[i][0] + 1):
                label[some_interval[i][0] + j] = 1
        return label

    

    '''yuting try'''
    attacks = []
    start = -1
    end = -1
    window = 200
    
    print('error_out_ewma', len(error_out_ewma))
    for d_idx in range(len(error_out_ewma)):
        # look at the largest difference
        if error_out_ewma[d_idx] > threshold:
            if start == -1:  # start new attack candidate
                start = d_idx
                end = d_idx
            else:  # extend the existing
                end = d_idx
        else:
            if end != -1:
                # ignore changes that don't last
                if end - start > window:
                    attacks.append((start, end))
                start = -1
                end = -1
    if end != -1 and end - start > window:
        attacks.append((start, end))

    print(attacks)
    attacks = mergeIntervals(attacks)
    print(attacks)
    pred_labels = np.zeros(ground_truth.shape)
    for k in range(len(attacks)):
        for j in range(attacks[k][0], attacks[k][1] + 1):
            pred_labels[j] = 1
    NN_label=pred_labels
    '''yuting try'''

    # NN_label = np.where(threshold >= error_out_ewma, 0, 1)

    true_label = ground_truth
    # print(np.argwhere(true_label==1))
    
    seq_true = np.argwhere(true_label == 1).reshape(-1)

    interval_true0 = interval_generation(seq_true)
    print('interval_true0',interval_true0)
   
    seq_true = np.argwhere(bianchenglabel(interval_true0) == 1).reshape(-1)
    
    interval_true = interval_generation(seq_true, proportion)
    # true_label
    true_label = bianchenglabel(interval_true)
    # print('true_label',np.argwhere(true_label == 1))

    intersect = np.array(list((map(lambda a, b: a * b, true_label, NN_label))))
    seq_intersect = np.argwhere(intersect == 1).reshape(-1)
    interval_intersect = interval_generation(seq_intersect)
    
    
    seq_NN = np.argwhere(NN_label == 1).reshape(-1)
    interval_NN = interval_generation(seq_NN)

    
    interval_error_flag = 0
    for item in range(len(interval_NN)):
        if interval_NN[item][1] - interval_NN[item][0] > 51312:
            interval_error_flag = 1
    if interval_error_flag == 1:
        return 0, 0, 0, (), (), ()

    judge = [((interval_intersect[i][0] + interval_intersect[i][1]) / 2) for i in range(len(interval_intersect))]
    interval_TP_in_true = []
    interval_TP_in_NN = []

    for i in range(len(judge)):
        for j in range(len(interval_true)):
            if interval_true[j][0] <= judge[i] <= interval_true[j][1]:
                interval_TP_in_true.append(interval_true[j])
    for i in range(len(judge)):
        for j in range(len(interval_NN)):
            if interval_NN[j][0] <= judge[i] <= interval_NN[j][1]:
                interval_TP_in_NN.append(interval_NN[j])

    set_TP_in_true = set(interval_TP_in_true)

    set_TP_in_NN = set(interval_TP_in_NN)
    
    set_true = set(interval_true)

    set_NN = set(interval_NN)
    set_FN = set_true - set_TP_in_true
    set_FP = set_NN - set_TP_in_NN
    
    interval_FN = sorted(set_FN, key=interval_true.index)
    interval_FP = sorted(set_FP, key=interval_NN.index)
    interval_TP_pure = sorted(set_TP_in_true, key=interval_true.index)

    if len(interval_TP_pure) + len(interval_FP) != 0:
        precision = len(interval_TP_pure) / (len(interval_TP_pure) + len(interval_FP))
    else:
        precision = 0
    if len(interval_FN) + len(interval_TP_pure) != 0:
        recall = len(interval_TP_pure) / (len(interval_FN) + len(interval_TP_pure))
    else:
        recall = 0
    if precision != 0 and recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    print("True: " + str(len(interval_true)))
    print("FP: " + str(len(interval_FP)))
    print("FN: " + str(len(interval_FN)))
    tp_report = "TP: {tp_num}"

    print(tp_report.format(
        tp_num=str(len(interval_TP_pure) + 1) if interval_true[2] in interval_TP_pure else str(len(interval_TP_pure))))

    '''yuting try'''
    if(recall>precision):
        f1=0
        recall=0
        precision=0
    '''yuting try'''
    return precision, recall, f1, interval_TP_pure, interval_FP, interval_FN






if __name__ == '__main__':

    pass
