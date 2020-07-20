import datetime
# coding=utf-8
import argparse
import json
import numpy as np
import os
import os.path as osp
import scipy.io as scio
import random
from sklearn import metrics
from keras import backend as K
import tensorflow as tf
from sklearn import preprocessing, metrics
import pandas as pd
import logging

from keras.models import load_model
import re
import PreprocessingSWAT_1 as PreprocessingSWAT
import h5py
# My Implementation
import models
import conditional_test_add_pure_merge_interval_try

# 解决 Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR 问题

# from tf.compat.v1 import ConfigProto
# from tf.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# 解决 Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR 问题

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

'''GPU Limitation'''
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

'''function'''
class Logger(object):
    def __init__(self,log_file_name,log_level,logger_name):
        #第一步，创建一个logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        #第二步，创建一个handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        #第三步,定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s '
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        #第四步,将Hander添加到logger中
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
    
def NN_predict(model,model_name,dataX,dataY):
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1)
    dataY = dataY.reshape(dataY.shape[0], dataY.shape[1], dataY.shape[2], dataY.shape[3], 1)

    dataX_zeros=np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)
    dataX_decoder_zeros = np.zeros((dataX.shape[0], 1, dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)

    if dataY.shape[1]==1:
        [predict_label1, predict_label2] = model.predict([dataX, dataX_zeros,dataX_decoder_zeros], batch_size=32, verbose=2)  #
    else:
        if model_name in ['MCCED','MCCED_wo_1stg']:
            [predict_label1, predict_label2] = model.predict([dataX, dataX_decoder_zeros], batch_size=32, verbose=2)  #
        elif model_name in ['MCCED_wo_atten_2stg', 'MCCED_wo_2stg','CCED']:
            [predict_label1, predict_label2] = model.predict(dataX, batch_size=32, verbose=1)  #callbacks=[checkpoint], validation_split=0.2)
        elif model_name == 'MCCED_wo_recon':
            predict_label2 = model.predict([dataX, dataX_decoder_zeros], batch_size=32, verbose=2)  #
            predict_label1=predict_label2
        else:
            raise ValueError('模型错误')
        
    predict_label1 = predict_label1[:, ::-1, :, :, :]  # 把时间轴取反

    data_predict_NN = predict_label2.reshape(
        predict_label2.shape[0] * predict_label2.shape[1] * predict_label2.shape[2], predict_label2.shape[3])
    data_rcstr_NN = predict_label1.reshape(predict_label1.shape[0] * predict_label1.shape[1] * predict_label1.shape[2],
                                           predict_label1.shape[3])
    data_predict_true = dataY.reshape(dataY.shape[0] * dataY.shape[1] * dataY.shape[2], dataY.shape[3])
    data_rcstr_true = dataX.reshape(dataX.shape[0] * dataX.shape[1] * dataX.shape[2], dataX.shape[3])

    return data_predict_NN, data_rcstr_NN, data_predict_true, data_rcstr_true


def temp_delete_period(test_input_raw,test_predict_raw,ground_truth):
    input_win=test_input_raw.shape[1]*test_input_raw.shape[2]
    
    # 源序列重组
    test_data=np.append(test_input_raw,test_predict_raw[-1,:,:,:][np.newaxis,:],axis=0)
    test_data=test_data.reshape(test_data.shape[0]*test_data.shape[1]*test_data.shape[2],test_data.shape[3])
    # test_predict=test_predict_raw.reshape(test_predict_raw.shape[0]*test_predict_raw.shape[1]*test_predict_raw.shape[2],test_predict_raw.shape[3])
    ground_truth1=ground_truth

    
    basetime = datetime.datetime.strptime('2015-12-28 10:00:00 AM', '%Y-%m-%d %I:%M:%S %p')
    deleted_period1=[('2015-12-31 1:45:19 AM','2015-12-31 11:15:27 AM'),('2015-12-30 9:56:29 AM','2015-12-30 10:20:21 AM')]

    deleted_period2=[('2015-12-31 1:45:19 AM','2015-12-31 11:15:27 AM')]
    
    for period in deleted_period2:
        start_strptime = datetime.datetime.strptime(period[0], '%Y-%m-%d %I:%M:%S %p')
        end_strptime = datetime.datetime.strptime(period[1], '%Y-%m-%d %I:%M:%S %p')
        start = int((start_strptime - basetime).total_seconds())
        end = int((end_strptime - basetime).total_seconds())

        print('start',start)
        print('end',end)

        test_data=np.delete(test_data,range(start,end),axis=0)
        # test_predict=np.delete(test_predict,range(start-input_win,end-input_win),axis=0)
        ground_truth=np.delete(ground_truth,range(start-input_win,end-input_win))


    # print('ground_truth前五第2次测试-------1',ground_truth[1700:1800])
    # print('ground_truth前五第2次测试-------2',ground_truth[229519:229600])

    ground_truth = ground_truth[:(ground_truth.shape[0] // input_win) * input_win].reshape(-1)
    # print('------！！！ground truth现在的形状？！！！------',ground_truth.shape)
    
    test_input, test_predict=PreprocessingSWAT.Dataset_Processing(test_data, input_win*2, input_win, input_win,
                                              input_win, test_input_raw.shape[2], 1, 51,
                                              True)
    # print('test_input shape删完之后剩多少？',test_input.shape[0]*test_input_raw.shape[1]*test_input_raw.shape[2],test_input_raw.shape[3])
    # print('------！！！ground truth现在的形状？！！！------',ground_truth.shape)
    # print('test_predict shape删完之后剩多少？',test_predict.shape[0]*test_input_raw.shape[1]*test_input_raw.shape[2],test_input_raw.shape[3])
    assert ground_truth.shape[0]==test_input.shape[0]*test_input_raw.shape[1]*test_input_raw.shape[2]
    

    # test_input=test_input.reshape(-1,test_input_raw.shape[1],test_input_raw.shape[2],test_input_raw.shape[3])
    # test_predict=test_predict.reshape(-1,test_predict_raw.shape[1],test_predict_raw.shape[2],test_predict_raw.shape[3])

    return test_input,test_predict,ground_truth

def options_parser():
    parser = argparse.ArgumentParser(description='Train a neural network to handle real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    parser.add_argument('--pre_path', help='preprocessing_timepath', type=str, default='')
    parser.add_argument('--res_path', help='result_timepath', type=str, default='')
    parser.add_argument('--model', help='result_timepath', type=str, default='')
    return parser

def load_settings_from_file(settings):
    # settings可以是任何一个txt形式的字典文件
    settings_path = "./"+settings['settings_file'] + ".txt"
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r',encoding='utf-8'))
    # check for settings missing in file
    return settings_loaded



def get_settings_and_files():
    parser = options_parser()
    settings_raw = vars(parser.parse_args())
    
    if settings_raw['settings_file']:
        settings = load_settings_from_file(settings_raw)

    result_path = "../seconddata/" + settings_raw['pre_path'] + "ConvLstm预处理结果(conditional版本)/"
    resultlist = os.listdir(result_path)  # 列出文件夹下所有的目录与文件
    total_result = []
    for i in range(0, len(resultlist)):
        if resultlist[i].endswith('.npz'):
            path = os.path.join(result_path, resultlist[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])
    choice_result = total_result[0]

#     使用kstest之后的传感器序列
#     sensor_list=[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 39, 41, 42, 43, 45, 47, 48, 49, 50]
    
    # train_input = choice_result['train_input'][:,:,:,sensor_list]
    # train_predict = choice_result["train_predict"][:,:,:,sensor_list]
    # test_input = choice_result["test_input"][:,:,:,sensor_list]
    # test_predict = choice_result["test_predict"][:,:,:,sensor_list]

    train_input = choice_result['train_input']
    train_predict = choice_result["train_predict"]
    test_input = choice_result["test_input"]
    test_predict = choice_result["test_predict"]

    ground_truth=choice_result["ground_truth"]
    params = settings

    print(test_input.shape)

    # print('universal里的传感器有{}个'.format(test_input.shape[-1]))
    ##################### 是否删除某一时间段 #####################
    # test_input,test_predict,ground_truth=temp_delete_period(test_input,test_predict,ground_truth)
    network = models.__dict__[settings_raw['model']]()
    input_win=test_input.shape[1]*test_input.shape[2]
    # print('------Test一下input的值------',test_input[3,1,1,5])
    return params,input_win, network, ground_truth,train_input,train_predict,test_input,test_predict



def sort_model_by_time(model_path):
    models = os.listdir(model_path)
    if not models:
        return
    else:
        
        models=[models[i] for i in range(len(models)) if models[i][-2:]=='h5']
        # files = sorted(models, key=lambda x: os.path.getmtime(os.path.join(model_path, x)),reverse=True)
        files = sorted(models, key=lambda x: round(float(x[19:22])),reverse=True)
        
        return files


def one_model_operation_conditional(encoder_model,past_decoder_model,fu_decoder_model,model_path):
    
    # 
    with model_graph.as_default():
        model_session=tf.Session()
        with model_session.as_default():
            encoder_model.load_weights(model_path, by_name=True)
            past_decoder_model.load_weights(model_path, by_name=True)
            fu_decoder_model.load_weights(model_path, by_name=True)

            _,_, train_input, train_predict, test_input, test_predict = get_settings_and_files()
    
    # with model_graph.as_default():
    #     with model_session.as_default():
            train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true = network.NN_predict(encoder_model,
                                                                                                past_decoder_model,
                                                                                                fu_decoder_model, train_input,
                                                                                                train_predict)
            test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = network.NN_predict(encoder_model, past_decoder_model,
                                                                                            fu_decoder_model, test_input,
                                                                                            test_predict)
    model_session.close()
    # K.clear_session()
    return train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true,test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true

def one_model_operation_unconditional(model,model_name,model_path,params, train_input, train_predict,test_input,test_predict):
    # model_graph=tf.Graph()
    # model_graph = tf.Graph()
    # sess = tf.Session(graph=model_graph)
    # K.set_session(sess)
    # K.get_session().run(tf.global_variables_initializer())

    # with tf.Graph().as_default() as model_graph:
    #     with tf.Session() as model_session:
            # 7月7日注释
            # model = model(params,train_input, train_predict)
            # print('model_name是什么',model_name)
            # print('********************  开始预测  ********************')
            
    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true = NN_predict(model, model_name, train_input, train_predict)
    test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = NN_predict(model, model_name, test_input, test_predict)
            
            
    print('********************  已经全都预测完  ********************')
    # model_session=None
    # model_graph=None
    # model_session.close()
    # K.clear_session()
    return train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true,test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true


def test(model,model_name,model_path,fuse,logger,params, input_win, ground_truth_raw, train_input, train_predict, test_input, test_predict):
    weight_or_not = False
    nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M')
    logger.info('\n################## {}搜索: 融合方式为{}  ##################\n'.format(nowTime,fuse))
    
    '''预测+重构'''

    alpha = 0.5
    beta = 0.5
    # train_predict_NN = beta * train_predict_NN[:-window] + alpha * train_rcstr_NN[window:]
    # train_predict_true = beta * train_predict_true[:-window] + alpha * train_rcstr_true[window:]
    # test_predict_NN = beta * test_predict_NN[:-window] + alpha *test_rcstr_NN[window:]
    # test_predict_true = beta * test_predict_true[:-window] + alpha * test_rcstr_true[window:]
    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true, test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true= one_model_operation_unconditional(model,model_name,model_path,params,train_input, train_predict,test_input,test_predict)
    ground_truth = ground_truth_raw[:-input_win]
    train_predict_true=train_predict_true[:-input_win]
    train_predict_NN=train_predict_NN[:-input_win]
    test_predict_NN = test_predict_NN[:-input_win]
    test_predict_true = test_predict_true[:-input_win]

    train_rcstr_true=train_rcstr_true[input_win:]
    train_rcstr_NN=train_rcstr_NN[input_win:]
    test_rcstr_NN = test_rcstr_NN[input_win:]
    test_rcstr_true = test_rcstr_true[input_win:]

    '''预测+重构融合方式'''
    if fuse == 'A':
        alpha = params['rcstr_weight']
        beta = params['predict_weight']
        train_predict_NN = beta * train_predict_NN + alpha * train_rcstr_NN
        train_predict_true = beta * train_predict_true + alpha * train_rcstr_true
        test_predict_NN = beta * test_predict_NN + alpha *test_rcstr_NN
        test_predict_true = beta * test_predict_true + alpha * test_rcstr_true
    elif fuse == 'B':
        pass

    # 2月27日之前删除：[20, 31, 6, 32, 30, 33, 12, 34, 24, 4]
    # 2月27日之后删除：[0, 7, 35, 26, 28]
    # delete_sensor=5
    assert test_predict_true.shape[0]==ground_truth.shape[0]

    # '''post processing'''
    #WADI 72个特征
    # for j in range(train_predict_NN.shape[1]):
    #     if (j == 0 or j == 1 or j == 2 or j == 5 or j == 16 or j == 17 or j == 18 or j == 19 or j == 20
    #         or j == 21 or j == 22 or j == 23 or j == 24 or j == 37 or j == 38 or j == 39 or j == 40
    #         or j == 41 or j == 42 or j == 43 or j == 43 or j == 45 or j == 59 or j == 61 or j == 63 or j == 64 or j == 71):
    #         continue
    #     else:
    #         train_predict_NN[:, j] = np.around(train_predict_NN[:, j])
    #         test_predict_NN[:, j] = np.around(test_predict_NN[:, j])
    # '''post processing'''

    # '''post processing'''
    # # SWAT 40个特征
    # for j in range(train_predict_NN.shape[1]):
    #     if (j == 0 or j == 1 or j == 5 or j == 13 or j == 14 or j == 21 or j == 22 or j == 23 or j == 29
    #         or j == 30 or j == 31 or j == 32 or j == 35 or j == 36):
    #         continue
    #     else:
    #         train_predict_NN[:, j] = np.around(train_predict_NN[:, j])
    #         test_predict_NN[:, j] = np.around(test_predict_NN[:, j])
    # '''post processing'''


    '''临时存储字典'''
    start_t1 = '2015-12-28 10:02:00 AM'
    end_t1 = '2016-1-2 2:59:59 PM'
    # start_t1 = '2017-10-9 06:02:00 PM'
    # end_t1 = '2017-10-11 06:00:00 PM'

    '''单点'''
#         weight_or_not将由函数形参指定
    test_obj = conditional_test_add_pure_merge_interval_try.analysis(train_rcstr_true, train_rcstr_NN, test_rcstr_true, test_rcstr_NN, train_predict_true, train_predict_NN, test_predict_true, test_predict_NN,
 ground_truth, start_t1, end_t1, input_win, weight_or_not=weight_or_not,
                                              weight_type=1,
                                              norm_or_not=False,rcstr_weight=params['rcstr_weight'],predict_weight=params['predict_weight'], fuse=fuse, specific=osp.basename(model_path))  # 这里的False指的是是否使用归一化误差

    max_wgt_error_value=np.max(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma)
    near_max_wgt_error_value=np.percentile(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma,99.99)
    min_wgt_error_value=np.min(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma)
    near_min_wgt_error_value=np.percentile(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma,0.01)
    logger.info('Max:{}'.format(max_wgt_error_value))
    logger.info('Min:{}'.format(min_wgt_error_value))
    logger.info('Near Min:{}'.format(near_min_wgt_error_value))
    logger.info('Near Max:{}'.format(near_max_wgt_error_value))

    '''指定绘图的区间'''
    period_specific='period2'
    roc_img_path=os.path.join(os.path.dirname(model_path[:-1]),'{0}_{1}_roc曲线'.format('加权' if weight_or_not==True else '不加权',period_specific))
    if not os.path.exists(roc_img_path):
        os.makedirs(roc_img_path)

    # near_min_wgt_error_value=0.001
    # near_min_wgt_error_value=0.7936


    my_dict1 = test_obj.threshold_grid_search_no_plotting(weight_or_not=weight_or_not, number=100, lower=near_min_wgt_error_value, upper=near_max_wgt_error_value,
                                                         consecutive_or_not=False, fine_or_not=True,roc_curve=False, roc_img_path=roc_img_path)
    logger.info(my_dict1)  # 输出每一个模型最好的结果

    # '''event_baesd'''
    # test_obj1 = conditional_test_add_pure.analysis(train_predict_true, train_predict_NN, test_predict_true,
    #                                                   test_predict_NN,
    #                                                   ground_truth, start_t1, end_t1, 120, weight_or_not=True,
    #                                                   weight_type=1,
    #                                                   norm_or_not=False,
    #                                                   specific=lastk_models_name[i])  # 这里的False指的是是否使用归一化误差
    # my_dict1 = test_obj1.threshold_grid_search_no_plotting(weight_or_not=True, number=100, lower=0.01, upper=0.5,
    #                                                      consecutive_or_not=True, fine_or_not=True)
    # print(my_dict1) # 输出每一个模型最好的结果
    # model_dict1["模型"].append(my_dict1["模型"])
    # model_dict1["阈值"].append(my_dict1["阈值"])
    # model_dict1["最好的F1值"].append(my_dict1["最好的F1值"])
    # model_dict1["此时的Precision"].append(my_dict1["此时的Precision"])
    # model_dict1["此时的Recall"].append(my_dict1["此时的Recall"])

    '''输出预测误差'''
    from sklearn import metrics
    seq_anomaly = list(np.argwhere(ground_truth == 1).reshape(-1))
    seq_norm = list(np.argwhere(ground_truth == 0).reshape(-1))

    # 只取前100个验证一下
    if fuse=='A':
        test_pure_anomaly_error = metrics.mean_absolute_error(test_predict_NN[seq_anomaly, :],
                                                             test_predict_true[seq_anomaly, :])
        test_pure_norm_error = metrics.mean_absolute_error(test_predict_NN[seq_norm, :],
                                                          test_predict_true[seq_norm, :])
        train_pure_norm_error = 0.5*metrics.mean_absolute_error(train_rcstr_NN,train_rcstr_true)
        train_pure_norm_error += 0.5*metrics.mean_absolute_error(train_predict_NN,train_predict_true)
    elif fuse=='B':
        test_pure_anomaly_error = 0.5*metrics.mean_absolute_error(test_predict_NN[seq_anomaly, :],
                                                             test_predict_true[seq_anomaly, :])
        test_pure_norm_error = 0.5*metrics.mean_absolute_error(test_predict_NN[seq_norm, :],
                                                          test_predict_true[seq_norm, :])
        test_pure_anomaly_error += 0.5*metrics.mean_absolute_error(test_rcstr_NN[seq_anomaly, :],
                                                             test_rcstr_true[seq_anomaly, :])
        test_pure_norm_error += 0.5*metrics.mean_absolute_error(test_rcstr_NN[seq_norm, :],
                                                          test_rcstr_true[seq_norm, :])
        
        train_pure_norm_error = 0.5*metrics.mean_absolute_error(train_rcstr_NN,train_rcstr_true)
        train_pure_norm_error += 0.5*metrics.mean_absolute_error(train_predict_NN,train_predict_true)
        
    elif fuse=='C':
        test_pure_anomaly_error = metrics.mean_absolute_error(test_predict_NN[seq_anomaly, :],
                                                             test_predict_true[seq_anomaly, :])
        test_pure_norm_error = metrics.mean_absolute_error(test_predict_NN[seq_norm, :],
                                                          test_predict_true[seq_norm, :])
    elif fuse=='D':
        test_pure_anomaly_error = metrics.mean_absolute_error(test_rcstr_NN[seq_anomaly, :],
                                                             test_rcstr_true[seq_anomaly, :])
        test_pure_norm_error = metrics.mean_absolute_error(test_rcstr_NN[seq_norm, :],
                                                          test_rcstr_true[seq_norm, :])

    my_dict2 = {
        '训练集中的预测误差': train_pure_norm_error,
        '测试集中纯正常的预测误差': test_pure_norm_error,
        '测试集中纯异常的预测误差': test_pure_anomaly_error,
        '测试集中异常与正常的平均预测误差gap': test_pure_anomaly_error - test_pure_norm_error
    }
    logger.info(my_dict2)
    

    # del test_obj
    return my_dict1, my_dict2

if __name__ == '__main__':
    # Instruction:
    # 1.根据不同情况有不同的backwards_num 如果picture正常画出，backwards_num=2，没有=1
    # 2.换了不同的网络要换不同的模型 只改动import和as之间就可以
    # 3.teacher_forcing_or_not

    # import 无卷积_无attention_1layer_非teacher_forcing as network
    # model_path="../resultdata/12_29_12_55conditional训练结果(conditional版本)/模型"\
    # import mem_无卷积_无attention_1layer_非teacher_forcing as network


    # import 无卷积_无attention_1layer_非teacher_forcing as network
    # model_path="../resultdata/12_29_12_55conditional训练结果(conditional版本)/模型"\
    # import mem_无卷积_无attention_1layer_非teacher_forcing as network
    
    # model_path="../resultdata/01_04_07_09conditional训练结果(conditional版本)/模型"  # 0.3 0.7 0.5 0.5 zyx训测
    #model_path="../resultdata/01_04_12_10conditional训练结果(conditional版本)/模型"  # 0.4 0.6 0.4 0.6 zyx训测 
    # 三无用噪声训练
    # model_path='../resultdata/01_12_23_51conditional训练结果(conditional版本)/模型'
    # mem缺省了一些名字
    # model_path="../resultdata/01_12_14_52conditional训练结果(conditional版本)/模型/"
    # 三无没有噪声训练
    # model_path="../resultdata/01_31_22_31conditional训练结果(conditional版本)/模型/"
    # advance MEM
    
    parser = options_parser()
    settings_raw = vars(parser.parse_args())
    model_path="../resultdata/"+settings_raw['res_path']+"conditional训练结果(conditional版本)/模型/"
    logger = Logger(log_file_name=osp.join("../resultdata/"+settings_raw['res_path']+"conditional训练结果(conditional版本)",'fuse_B_error_log.txt'), log_level=logging.DEBUG, logger_name='SWAT').get_log()
    model_name = settings_raw['model']
    start_NN_time = datetime.datetime.now()
    find_best_in_trained_models(models_path=model_path, fuse = 'B', weight_or_not = False)

    end_NN_time = datetime.datetime.now()
    m, s = divmod(((end_NN_time - start_NN_time).total_seconds()), 60)
    h, m = divmod(m, 60)
    print("过程" + "用时" + str(h) + "小时" + str(m) + "分" + str(s) + "秒")




