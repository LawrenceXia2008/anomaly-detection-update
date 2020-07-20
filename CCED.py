import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Reshape, Conv3D, Flatten, RepeatVector
from keras import regularizers
import numpy as np
from keras.models import Model
import scipy.io as scio
import random
from keras.models import Model
from keras.layers import Input, LSTM, Dense, ConvLSTM2D, BatchNormalization, Conv3D, TimeDistributed
from sklearn import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import Lambda
import matplotlib
matplotlib.use('Agg')
import argparse
import json
import datetime
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
weight = 0.0#不加噪声改变weight就可以了
mean = 0.0
scale = 1.0

def NN_predict(model,dataX,dataY):
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # dataX_noise = dataX + weight * np.random.normal(loc = mean, scale = scale, size = dataX.shape)
    dataY = dataY.reshape(dataY.shape[0], dataY.shape[1], dataY.shape[2], dataY.shape[3], 1)
    
    # dataX_zeros_noise=dataX_zeros + weight * np.random.normal(loc = mean, scale = scale, size = dataX_zeros.shape)
    # initial_s_noise = initial_s + weight * np.random.normal(loc=mean, scale=scale, size=initial_s.shape)
        
    [predict_label1, predict_label2] = model.predict(dataX, batch_size=10, verbose=1)
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
    parser.add_argument('--pre_path', help='preprocessing_timepath', type=str, default='')
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    return parser

def load_settings_from_file(settings):
    # settings可以是任何一个txt形式的字典文件
    settings_path = "./"+settings['settings_file'] + ".txt"
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r',encoding='utf-8'))
    # check for settings missing in file
    return settings_loaded
# def weighted_loss()
def get_settings_and_files():
    parser = options_parser()
    settings_raw = vars(parser.parse_args())
    
    if settings_raw['settings_file']:
        settings = load_settings_from_file(settings_raw)

    result_path = "../seconddata/" + settings_raw['pre_path'] + "ConvLstm预处理结果(conditional版本)/"
    list = os.listdir(result_path)  # 列出文件夹下所有的目录与文件
    total_result = []
    for i in range(0, len(list)):
        path = os.path.join(result_path, list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])
    choice_result = total_result[0]
    train_input = choice_result['train_input']
    train_predict = choice_result["train_predict"]
    test_input = choice_result["test_input"]
    test_predict = choice_result["test_predict"]
    ground_truth=choice_result["ground_truth"]
    params = settings
    return params,choice_result["tag"],ground_truth,train_input,train_predict,test_input,test_predict

def network(params,train_input,train_predict):

    ##############################################    ENCODER    #############################################
    # try:
    encoder_inputs = Input(shape=(train_input.shape[1], train_input.shape[2],train_input.shape[3], 1),name='encoder_inputs')
    # encoder_outputs = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_inputs)
    # encoder_outputs = BatchNormalization()(encoder_outputs)
    # encoder_outputs = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_outputs)

    encoder = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same',return_sequences=True,return_state=False, kernel_initializer='he_uniform',activity_regularizer=regularizers.l1(params["regularizer"]),name='encoder')  # ruturn_sequences默认是False
    encoder_outputs = encoder(encoder_inputs)
    BN1=BatchNormalization(name='BN1')
    encoder_outputs=BN1(encoder_outputs)
    # print(encoder_outputs,encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    # encoder_states = [state_h, state_c]  #copy的最近一个时刻的状态值

    ##############################################    PAST DECODER    ##############################################
    # Set up the decoder, using `encoder_states` as initial state.
    
    decoder_rec = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True, kernel_initializer='he_uniform', name='past_decoder')
    decoder_rec_outputs = decoder_rec(encoder_outputs)
    decoder_dense_rec = Conv3D(filters=1, kernel_size=(3, 3, 3),activation=params['activation'],padding='same', kernel_initializer='he_uniform',data_format='channels_last',name='past_decoder_dense')
    BN3 = BatchNormalization(name="BN3")
    decoder_rec_outputs=BN3(decoder_rec_outputs)
    decoder_rec_outputs = decoder_dense_rec(decoder_rec_outputs)
    # print(decoder_outputs)
    # pre_conditional

    ##############################################    FUTURE DECODER    ##############################################
    
    
    decoder_pre = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True, kernel_initializer='he_uniform',name='fu_decoder')
    decoder_dense_pre = Conv3D(filters=1, kernel_size=(3, 3, 3),activation=params['activation'],padding='same', data_format='channels_last', kernel_initializer='he_uniform',name='fu_decoder_dense')
    BN4 = BatchNormalization(name="BN4")
    max_decoder_seq_length = train_predict.shape[1]# max_decoder_seq_length表示目标输出的时间戳长度，timesteps表示编码器输入的时间戳长度

    decoder_pre_outputs = decoder_pre(encoder_outputs)
    decoder_pre_outputs = BN4(decoder_pre_outputs)
    decoder_pre_outputs = decoder_dense_pre(decoder_pre_outputs)
    
    COMPOSITE_ED = Model(inputs=encoder_inputs,
                         outputs=[decoder_rec_outputs, decoder_pre_outputs])
    # COMPOSITE_ED.summary()
    rmsprop = optimizers.RMSprop(lr=params['lr'], rho=0.9, epsilon=None, decay=0.0)
    COMPOSITE_ED.compile(loss=[params['loss1'], params['loss2']], loss_weights=[params['rcstr_weight'], params['predict_weight']],
                         optimizer=rmsprop)
        ##############################################    Model saving    ##############################################

    return COMPOSITE_ED
# 返回的将要被存储的训练好参数的数据字典带有"tag"标签
def model_operation(COMPOSITE_ED):
    params, tag, ground_truth, train_input, train_predict, test_input, test_predict=get_settings_and_files()
    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    print("开始训练")
    print(train_input.shape, train_predict.shape, test_input.shape, test_predict.shape)
    #(4807, 5, 20, 52) (4807, 5, 20, 52) (4498, 5, 20, 52) (4498, 5, 20, 52)


    #打乱数据顺序
    np.random.seed(0)
    permutation = np.random.permutation(train_input.shape[0])
    train_input = train_input[permutation, :]
    train_predict = train_predict[permutation,:]

    train_input = train_input.reshape(train_input.shape[0],train_input.shape[1],train_input.shape[2],train_input.shape[3],1)

    train_predict = train_predict.reshape(train_predict.shape[0],train_predict.shape[1],train_predict.shape[2],train_predict.shape[3],1)
    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    initial = np.zeros((train_input.shape[0],1,train_input.shape[2],train_input.shape[3],1)) #(smaple,1,20,51,1)  初始化状态
    print('input',train_input.shape, train_zeros.shape, initial.shape,train_predict.shape)

    train_input_rev = train_input[:, ::-1, :, :, :]
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!加入噪声
    # train_input_noise = train_input + weight*np.random.normal(loc=mean, scale=scale, size=train_input.shape)
    # train_zeros_noise = train_zeros + weight*np.random.normal(loc=mean, scale =scale, size = train_zeros.shape)
    # initial_noise = initial + weight*np.random.normal(loc=mean, scale =scale, size = initial.shape)

    
    filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_weights_only=True, save_best_only=True, mode='min')
    #COMPOSITE_ED.summary()
    history=COMPOSITE_ED.fit(train_input, [train_input_rev, train_predict], nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'], callbacks=[checkpoint], validation_split=0.2)

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





if __name__=="__main__":

    nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M')  # 现在
    parser = options_parser()
    settings_raw = vars(parser.parse_args())
    
    if settings_raw['settings_file']:
        settings = load_settings_from_file(settings_raw)

    result_path = "../seconddata/" + settings_raw['pre_path'] + "ConvLstm预处理结果(conditional版本)/"
    list = os.listdir(result_path)  # 列出文件夹下所有的目录与文件
    total_result = []
    for i in range(0, len(list)):
        path = os.path.join(result_path, list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])

    pathc = "../resultdata/" + nowTime + "conditional训练结果" + "(conditional版本)/"
 
    pathm = pathc+"模型/"
    if not os.path.exists(pathc):
       os.makedirs(pathm)

    for i in range(len(total_result)):
        pathm_i = pathm + total_result[i]['tag']+"的模型/"
        os.makedirs(pathm_i)
        start_NN_time = datetime.datetime.now()
        COMPOSITE_ED = network(settings, total_result[i]["train_input"],total_result[i]["train_predict"])
        NN_choice_result=model_operation(COMPOSITE_ED)
        end_NN_time = datetime.datetime.now()
        m, s = divmod(((end_NN_time - start_NN_time).total_seconds()), 60)
        h, m = divmod(m, 60)
        print(NN_choice_result["tag"] + "过程" + "用时" + str(h) + "小时" + str(m) + "分" + str(s) + "秒")
        time_result = pathc + nowTime + "conditional训练结果" + NN_choice_result['tag']
        np.savez(time_result, result=NN_choice_result)




