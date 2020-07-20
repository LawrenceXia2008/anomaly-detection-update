import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os.path as osp
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Reshape, Conv3D, Flatten, RepeatVector
from keras import regularizers
import numpy as np
from keras.models import Model
import scipy.io as scio
import random
from keras.models import Model
from keras.layers import Input, LSTM, Dense, ConvLSTM2D, BatchNormalization, Conv3D, TimeDistributed,Activation
from sklearn import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import Lambda
import matplotlib
import argparse
import json
import datetime
import os
import TIED_ED
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
import h5py
import logging
import shutil
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
np.random.seed(10)
tf.set_random_seed(10)

## My Implementation ##
import models
from test_model import *

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

def options_parser():
    parser = argparse.ArgumentParser(description='Train a neural network to handle real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    parser.add_argument('--pre_path', help='preprocessing_timepath', type=str, default='')
    parser.add_argument('--model', help='model name', type=str, default='')
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
    print('seconddata',result_path)
    sec_list = os.listdir(result_path)  # 列出文件夹下所有的目录与文件
    total_result = []
    for i in range(0, len(sec_list)):
        if sec_list[i].endswith('.npz'):
            path = os.path.join(result_path, sec_list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])
    choice_result = total_result[0]
    train_input = choice_result['train_input']
    train_predict = choice_result["train_predict"]
    test_input = choice_result["test_input"]
    test_predict = choice_result["test_predict"]
    ground_truth=choice_result["ground_truth"]
    params = settings
    input_win=test_input.shape[1]*test_input.shape[2]
    return params,input_win,ground_truth,train_input,train_predict,test_input,test_predict

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


def model_operation(COMPOSITE_ED,model_name):
    params, input_win, ground_truth, train_input, train_predict, test_input, test_predict=get_settings_and_files()
    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    logger.info("开始训练")
    print(train_input.shape, train_predict.shape, test_input.shape, test_predict.shape)
    #(4807, 5, 20, 52) (4807, 5, 20, 52) (4498, 5, 20, 52) (4498, 5, 20, 52)
    #打乱数据顺序
    np.random.seed(10)
    permutation = np.random.permutation(train_input.shape[0])
    train_input = train_input[permutation, :]
    train_predict = train_predict[permutation,:]

    train_input = train_input.reshape(train_input.shape[0],train_input.shape[1],train_input.shape[2],train_input.shape[3],1)

    train_predict = train_predict.reshape(train_predict.shape[0],train_predict.shape[1],train_predict.shape[2],train_predict.shape[3],1)
    
    train_decoder_zeros = np.zeros((train_input.shape[0],1,train_input.shape[2],train_input.shape[3],1)) #(smaple,1,20,51,1)  初始化状态

    train_input_rev = train_input[:, ::-1, :, :, :]

    if train_predict.shape[1] == 1:
        filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min',period=1)
        # COMPOSITE_ED.summary()
        history = COMPOSITE_ED.fit([train_input, train_zeros], [train_input_rev, train_predict],
                                   nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'],
                                   callbacks=[checkpoint], validation_split=0.2)
    else:
        filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min',period=1)
        #COMPOSITE_ED.summary()
        print('##############  model name  ##############',model_name)
        if model_name in ['MCCED','MCCED_wo_1stg']:
            history=COMPOSITE_ED.fit([train_input, train_decoder_zeros], [train_input_rev, train_predict], nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'], callbacks=[checkpoint], validation_split=0.2)
        elif model_name in ['MCCED_wo_atten_2stg', 'MCCED_wo_2stg','CCED']:
            history=COMPOSITE_ED.fit(train_input, [train_input_rev, train_predict], nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'], callbacks=[checkpoint], validation_split=0.2)
        elif model_name == 'MCCED_wo_recon':
            history=COMPOSITE_ED.fit([train_input, train_decoder_zeros], train_predict, nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'], callbacks=[checkpoint], validation_split=0.2)
        else:
            raise ValueError('模型错误')
    # COMPOSITE_ED.save(pathm + nowTime+'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5')
    


    np.savez(pathm+nowTime+"历史数据.npz",epoch=history.epoch,history=history.history)
    fig = plt.figure()  # 新建一张图
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig(pathm+nowTime + '_loss.png')

    ##############################################    数据保存    ##############################################

    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true =NN_predict(COMPOSITE_ED,train_input,train_predict)
    test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = NN_predict(COMPOSITE_ED,test_input,test_predict)
    my_dict = {"tag": tag,
                "train_predict_NN": train_predict_NN,
               "train_rcstr_NN": train_rcstr_NN,
               "test_predict_NN": test_predict_NN,
               "test_rcstr_NN": test_rcstr_NN,
               "train_predict_true": train_predict_true,
               "train_rcstr_true": train_rcstr_true,
               "test_predict_true": test_predict_true,
               "test_rcstr_true": test_rcstr_true,
               "ground_truth": ground_truth}
    return my_dict


def check_load_success(model,model_path):
    error_flag = 0

    f = h5py.File(model_path, 'r')
    layer_names_in_weights = [s.decode() for s in f.attrs['layer_names']]
    for name in layer_names_in_weights:
        if type(name) in [tuple, list]:
            layer_name = name[1]
            name = name[0]
        else:
            layer_name = name
        g = f[name]
        weights = [np.array(g[wn]) for wn in g.attrs['weight_names']]
        try:
            layer = model.get_layer(layer_name)
            #assert layer is not None
        except:
            print('layer missing %s' % (layer_name))
            print('    file  %s' % ([w.shape for w in weights]))
            error_flag+=1
            continue
        try:
            #print('load %s' % (layer_name))
            layer.set_weights(weights)
        except Exception as e:
            print('something went wrong %s' % (layer_name))
            print('    model %s' % ([w.shape.as_list() for w in layer.weights]))
            print('    file  %s' % ([w.shape for w in weights]))
            print(e)
    if error_flag>0:
        print('********************  Warning: 加载过程中出现层不匹配  ********************')
        return False
    else:
        print('********************  Success: 加载过程中所有对应层成功加载  ********************')
        return True


def train_model(COMPOSITE_ED,model_name):
    test_freq = 3

    val_loss_min = 1e7
    
    filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min',period=1)
    global params, input_win, ground_truth_raw, train_input, train_predict, test_input, test_predict
    params, input_win, ground_truth_raw, train_input, train_predict, test_input, test_predict=get_settings_and_files()
    
    input_dict = dict(params=params, 
    input_win=input_win, 
    ground_truth_raw=ground_truth_raw, 
    train_input=train_input, 
    train_predict=train_predict, 
    test_input=test_input, 
    test_predict=test_predict)

    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    logger.info("开始训练")
    print(train_input.shape, train_predict.shape, test_input.shape, test_predict.shape)
    #(4807, 5, 20, 52) (4807, 5, 20, 52) (4498, 5, 20, 52) (4498, 5, 20, 52)


    #打乱数据顺序
    np.random.seed(10)
    permutation = np.random.permutation(train_input.shape[0])
    train_input = train_input[permutation, :]
    train_predict = train_predict[permutation,:]

    train_input = train_input.reshape(train_input.shape[0],train_input.shape[1],train_input.shape[2],train_input.shape[3],1)

    train_predict = train_predict.reshape(train_predict.shape[0],train_predict.shape[1],train_predict.shape[2],train_predict.shape[3],1)
    
    train_decoder_zeros = np.zeros((train_input.shape[0],1,train_input.shape[2],train_input.shape[3],1)) #(smaple,1,20,51,1)  初始化状态

    train_input_rev = train_input[:, ::-1, :, :, :]

    epochs = params['nb_epochs']
    epochs = 200
    load_success_ornot = False
    valid_epochs = 0  # 有效轮数
    test_epochs = 0
    model_dict1 = {"最好的模型":[],"阈值":[],"最好的F1值":[],"此时的Precision":[],"此时的Recall":[],"window":[],"auc":[]}
    model_dict2 = {"训练集中的预测误差":[],"测试集中纯正常的预测误差":[],"测试集中纯异常的预测误差":[],"测试集中异常与正常的平均预测误差gap":[]}
    train_loss_stat = []
    val_loss_stat = []
    for epoch in range(1,epochs):
        logger.info('########### training in epoch{} ###########'.format(epoch))

        if model_name in ['MCCED','MCCED_wo_1stg']:
            history=COMPOSITE_ED.fit([train_input, train_decoder_zeros], [train_input_rev, train_predict], epochs=1, batch_size=settings['batch_size'], validation_split=0.2, verbose=2)
        elif model_name in ['MCCED_wo_atten_2stg', 'MCCED_wo_2stg','CCED']:
            history=COMPOSITE_ED.fit(train_input, [train_input_rev, train_predict], epochs=1, batch_size=settings['batch_size'], validation_split=0.2, verbose=2)
        elif model_name == 'MCCED_wo_recon':
            history=COMPOSITE_ED.fit([train_input, train_decoder_zeros], train_predict, epochs=1, batch_size=settings['batch_size'], validation_split=0.2, verbose=2)
        else:
            raise ValueError('模型错误')
        # history=COMPOSITE_ED.fit([train_input, train_decoder_zeros], [train_input_rev, train_predict], nb_epoch=1, batch_size=settings['batch_size'], validation_split=0.2)

        

        train_loss = float(history.history['loss'][0])
        val_loss = float(history.history['val_loss'][0])
        
        train_loss_stat.append(train_loss)
        val_loss_stat.append(val_loss)

        # 每一轮都要先保存模型，然后如果不符合要求再将其删去
        model_path = filepath.format(epoch=epoch,loss=train_loss,val_loss=val_loss)
        COMPOSITE_ED.save_weights(model_path)

        if val_loss<=val_loss_min:
        # print(filepath.format(epoch=1,loss=train_loss,val_loss=val_loss))
            valid_epochs+=1
            
            # COMPOSITE_ED.save_weights(model_path)


            COMPOSITE_ED.load_weights(model_path, by_name=True)
            #### 验证模型加载正确 ####
            if valid_epochs==1:
                load_success_ornot = check_load_success(COMPOSITE_ED,model_path)
                if load_success_ornot:
                    print('模型加载无误')
                else:
                    raise ValueError('模型加载有误')



            if valid_epochs%test_freq==0:
                # 测试
                # NN_predict(COMPOSITE_ED,model_name,train_input,train_predict)
                my_dict1, my_dict2 = test(COMPOSITE_ED,model_name,model_path,fuse='A',logger=logger,**input_dict)


                model_dict1["最好的模型"].append(my_dict1["model"])
                model_dict1["阈值"].append(my_dict1["thres"])
                model_dict1["最好的F1值"].append(my_dict1["F1"])
                model_dict1["此时的Precision"].append(my_dict1["Precision"])
                model_dict1["此时的Recall"].append(my_dict1["Recall"])
                model_dict1["window"].append(my_dict1["window"])
                model_dict1["auc"].append(my_dict1["auc"])
                
                model_dict2["训练集中的预测误差"].append(my_dict2["训练集中的预测误差"])
                model_dict2["测试集中纯正常的预测误差"].append(my_dict2["测试集中纯正常的预测误差"])
                model_dict2["测试集中纯异常的预测误差"].append(my_dict2["测试集中纯异常的预测误差"])
                model_dict2["测试集中异常与正常的平均预测误差gap"].append(my_dict2["测试集中异常与正常的平均预测误差gap"])

                test_epochs+=1
            val_loss_min = val_loss
        else:
            # 如果不符合loss的要求，则删去这个文件
            # load_success_ornot = check_load_success(COMPOSITE_ED,model_path)
            COMPOSITE_ED.load_weights(model_path, by_name=True)
            os.remove(model_path)

    try:
        fig = plt.figure()  # 新建一张图
        plt.plot(np.array(train_loss_stat), label='training loss')
        plt.plot(np.array(val_loss_stat), label='val loss')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        fig.savefig(pathm+nowTime + '_loss.png')
    except:
        pass
        print('遇到错误')

    logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<单点>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # 累计多个模型中最好的F1值
    f1max = max(model_dict1["最好的F1值"])
    best = np.argwhere(model_dict1["最好的F1值"] == f1max)[0][0]
    # 全部结果输出
    
    for _ in range(test_epochs):

        print_dict0 = {"最好的模型": model_dict1["最好的模型"][_], "阈值": model_dict1["阈值"][_],"最好的F1值": model_dict1["最好的F1值"][_], "此时的Precision": model_dict1["此时的Precision"][_], "此时的Recall": model_dict1["此时的Recall"][_], "window":model_dict1["window"][_],"auc":model_dict1["auc"][_]}
        logger.info(print_dict0)
    logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<best>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.info("最好的情况" + str({"模型": model_dict1["最好的模型"][best], "阈值": model_dict1["阈值"][best],"最好的F1值": model_dict1["最好的F1值"][best],
                         "此时的Precision": model_dict1["此时的Precision"][best], "此时的Recall": model_dict1["此时的Recall"][best], "window":model_dict1["window"][best], "auc":model_dict1["auc"][best]}))

    # 专门为F1值再输出一次结果
    for _ in range(test_epochs):
        print_dict1 = {"最好的模型": model_dict1["最好的模型"][_], "最好的F1值": model_dict1["最好的F1值"][_],"auc":model_dict1["auc"][_]}
        logger.info(print_dict1)
    # print("\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<event-based>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # # 累计多个模型中最好的F1值
    # f1max1 = max(model_dict1["最好的F1值"])
    # best1 = np.argwhere(model_dict1["最好的F1值"] == f1max1)[0][0]
    # # 全部结果输出
    #
    # for _ in range(lastk):
    #     my_dict1 = {"模型": model_dict1["模型"][_], "阈值": model_dict1["阈值"][_],"最好的F1值": model_dict1["最好的F1值"][_],
    #                "此时的Precision": model_dict1["此时的Precision"][_], "此时的Recall": model_dict1["此时的Recall"][_]}
    #     print(my_dict1)
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<best>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("最好的情况" + str({"模型": model_dict1["模型"][best1], "阈值": model_dict1["阈值"][best1],"最好的F1值": model_dict1["最好的F1值"][best1],
    #                      "此时的Precision": model_dict1["此时的Precision"][best1], "此时的Recall": model_dict1["此时的Recall"][best1]}))
    #

    logger.info("\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<预测误差>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for _ in range(test_epochs):
        
        logger.info('--------------------------第{}个模型--------------------------:{}'.format(_,model_dict1["最好的模型"][_][17:22]))
        print_dict2 = {"训练集中的预测+重构残差": model_dict2["训练集中的预测误差"][_],
        "测试集中纯正常的预测+重构残差": model_dict2["测试集中纯正常的预测误差"][_],
               "测试集中纯异常的预测+重构残差": model_dict2["测试集中纯异常的预测误差"][_],
                "测试集中异常与正常的平均残差gap": model_dict2["测试集中异常与正常的平均预测误差gap"][_]}
        logger.info(print_dict2)

        
    # print('********************  加载网络了之后加载了权重  ********************')


    

if __name__=="__main__":
    parser = options_parser()
    settings_raw = vars(parser.parse_args())

    if settings_raw['settings_file']:
        settings = load_settings_from_file(settings_raw)

    result_path = "../seconddata/" + settings_raw['pre_path'] + "ConvLstm预处理结果(conditional版本)/"

    nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M')  # 现在
    path_list = os.listdir(result_path)  # 列出文件夹下所有的目录与文件
    total_result = []
    for i in range(0, len(path_list)):
        if path_list[i].endswith('.npz'):
            path = os.path.join(result_path, path_list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])

    pathc = "../resultdata/" + nowTime + "conditional训练结果" + "(conditional版本)/"
    os.makedirs(pathc)
    pathm = pathc+"模型/"
    os.makedirs(pathm)
    # 将settings的txt复制到模型保存文件夹中一份
    shutil.copyfile(settings_raw['settings_file'] + ".txt", pathc+settings_raw['settings_file'] + ".txt")
    # 设置logger
    logger = Logger(log_file_name=osp.join(pathc,'log.txt'), log_level=logging.DEBUG, logger_name='SWAT').get_log()
    logger.info('#####  各参数设置为  #####\n{}'.format(settings))
    network = models.__dict__[settings_raw['model']]()
    logger.info('#####  model被定义为  #####\n{}'.format(settings_raw['model']))
    logger.info('#####  预处理路径为  #####\n{}'.format(settings_raw['pre_path']))

    for i in range(len(total_result)):
        pathm_i = pathm + total_result[i]['tag']+"的模型/"
        # os.makedirs(pathm_i)
        start_NN_time = datetime.datetime.now()
        
        COMPOSITE_ED = network(settings, total_result[i]["train_input"],total_result[i]["train_predict"])
        model_name = settings_raw['model']
        train_model(COMPOSITE_ED,model_name) # 此处传进model_name，因为model_name对model_operation中的操作是有影响的
        end_NN_time = datetime.datetime.now()
        m, s = divmod(((end_NN_time - start_NN_time).total_seconds()), 60)
        h, m = divmod(m, 60)
        logger.info("过程" + "用时" + str(h) + "小时" + str(m) + "分" + str(s) + "秒")
        # time_result = pathc + nowTime + "conditional训练结果" + NN_choice_result['tag']
        # np.savez(time_result, result=NN_choice_result)