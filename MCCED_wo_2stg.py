import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import datetime
import os
import TIED_ED
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


def NN_predict(model,dataX,dataY):
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1)
    dataY = dataY.reshape(dataY.shape[0], dataY.shape[1], dataY.shape[2], dataY.shape[3], 1)

    dataX_zeros=np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)

    if dataY.shape[1]==1:
        [predict_label1, predict_label2] = model.predict([dataX, dataX_zeros], batch_size=10, verbose=1)  #
    else:
        [predict_label1, predict_label2] = model.predict(dataX, batch_size=10, verbose=1)  #
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
    encoder_inputs = Input(shape=(train_input.shape[1], train_input.shape[2],train_input.shape[3], 1))
    # encoder_outputs = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_inputs)
    # encoder_outputs = BatchNormalization()(encoder_outputs)
    # encoder_outputs = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True)(encoder_outputs)

    encoder = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same',return_sequences=True,return_state=True, kernel_initializer='he_uniform',activity_regularizer=regularizers.l1(params["regularizer"]),name='encoder')  # ruturn_sequences默认是False
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    BN1=BatchNormalization(name='BN1')
    BN2=BatchNormalization(name='BN2')
    encoder_outputs=BN1(encoder_outputs)
    state_c=BN2(state_c)
    # print(encoder_outputs,encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    # encoder_states = [state_h, state_c]  #copy的最近一个时刻的状态值

    ##############################################    Meomory 模块    ##############################################
    # memory 层定义:寻址过程
    memory_size = params["memory_size"]
    stateh_addressing = TIED_ED.DenseLayerAutoencoder([memory_size], use_bias=False, name="stateh_addressing")
    # statec_addressing = TIED_ED.DenseLayerAutoencoder([memory_size], use_bias=False, name="statec_addressing")
    statec_addressing = stateh_addressing
    # encoder:c*memory_size
    # attention weight (?,memory_size)
    # decoder:memory_size*c
    # 获得attention之后的latent_vector
    flat_encoder_outputs = TimeDistributed(Flatten())(encoder_outputs)
    print("flat_encoder_outputs.shape", flat_encoder_outputs.shape)
    memoried_encoder_outputs0 = []

    for i in range(train_input.shape[1]):
        print("flat_encoder_outputs[:, i, :].shape", flat_encoder_outputs[:, i, :].shape)
        print("train_input.shape[2]*train_input.shape[3]*params",
              train_input.shape[2] * train_input.shape[3] * params["filter"])
        flat_encoder_outputs_i0 = Lambda(lambda x: x[:, i, :])(flat_encoder_outputs)
        addressed_state0 = stateh_addressing(flat_encoder_outputs_i0)
        addressed_state = Reshape((1, train_input.shape[2], train_input.shape[3], params["filter"]))(addressed_state0)
        memoried_encoder_outputs0.append(addressed_state)
    memoried_encoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(memoried_encoder_outputs0)

    state_h_use=Lambda(lambda x: x[:, -1, :, :, :])(memoried_encoder_outputs)
    state_c=Flatten()(state_c)
    state_c_use0=statec_addressing(state_c)
    state_c_use=Reshape((train_input.shape[2], train_input.shape[3], params["filter"]))(state_c_use0)

    ##############################################    PAST DECODER    ##############################################
    # Set up the decoder, using `encoder_states` as initial state.
    rec_convlstm_decoder = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_sequences=True, kernel_initializer='he_uniform', name='rec_convlstm_decoder')
    past_decoder_outputs = rec_convlstm_decoder(memoried_encoder_outputs)
    decoder_dense = Conv3D(filters=1, kernel_size=(3, 3, 3),activation=params['activation'],padding='same', kernel_initializer='he_uniform',data_format='channels_last',name='decoder_dense')
    BN3 = BatchNormalization(name="BN3")
    past_decoder_outputs=BN3(past_decoder_outputs)
    past_decoder_outputs = decoder_dense(past_decoder_outputs)



    ##############################################    FUTURE DECODER    ##############################################
    ##########    全局层初始化    ##########
    '''全局共用层'''
    # 对encoder_outputs进行维度变换方便其与attention相乘得到context_vector
    encoder_outputs_for_mul = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1, 4)), name="encoder_outputs_for_mul")

    context_vector_mul = Lambda(
        lambda x: K.permute_dimensions(K.batch_dot(x, atten_wgt_in_step_T, axes=[3, 4]), (0, 4, 1, 2, 3)),
        name="context_vector_mul")

    decoder_dense_pre = Conv2D(filters=1, kernel_size=(1, 1), activation=params['activation'], padding='same',
                               data_format='channels_last', kernel_initializer='he_uniform',name='decoder_dense_pre')
    BN4 = BatchNormalization(name="BN4")
    ##########    attention 模块     ##########
    # attention功能模块1:得到attention_weight
    def atten_wgt(EN_hidden_states, last_hidden_state):
        score_in_step = []
        EN_time_steps = train_input.shape[1]
        for j in range(EN_time_steps):
            score_in_step_j=K.sum(EN_hidden_states[:, j, :, :, :]*last_hidden_state,axis=[1,2,3])
            score_in_step.append(score_in_step_j)
        score_in_step=K.stack(score_in_step,axis=0)
        score_in_step=K.permute_dimensions(score_in_step, (1, 0))
        score_in_step=Reshape((EN_time_steps,1))(score_in_step)/params["scale_factor"]
        atten_wgt_in_step = K.softmax(score_in_step, axis=1)
        return atten_wgt_in_step
    get_atten_wgt_layer = Lambda(lambda x: atten_wgt(encoder_outputs, x), name="get_atten_wgt_layer")

    # attention功能模块2:调整attention_weight维度
    def adjust_atten_wgt_in_step(x):
        x = K.permute_dimensions(x, (0, 2, 1))  # atten_wgt_in_step (?, 1, 3)
        x = K.expand_dims(x, axis=1)
        x = K.expand_dims(x, axis=1)
        x = K.repeat_elements(x, train_predict.shape[2], 1)
        x = K.repeat_elements(x, train_predict.shape[3], 2)
        return x
    adjust_atten_wgt_in_step_layer = Lambda(lambda x: adjust_atten_wgt_in_step(x), name="adjust_atten_wgt_in_step_layer")

    '''全局共用层'''
    '''长度>1专用'''
    decoder_lstm_pre = ConvLSTM2D(filters=params['filter'], kernel_size=(3, 3), padding='same', return_state=True,
                                  return_sequences=False, kernel_initializer='he_uniform', name='decode_pre')
    
    output_reshape = Reshape((1, train_input.shape[2], train_input.shape[3], 1), name="output_reshape")
    '''长度>1专用'''
    ##########    长度为1的变窗预测（特殊）    ##########
    if train_predict.shape[1] == 1:
        # 使用AAAI方法获得context vector就不再进入新的解码器的convlstm层
        # 因之前全局定义了层，这里只调用函数
        '''<START> 求出context_vector'''
        memoried_encoder_outputs_T = encoder_outputs_for_mul(memoried_encoder_outputs)   # 已经改成经过memory的encoder
        atten_wgt_in_step = get_atten_wgt_layer(state_h)
        # atten_wgt_in_step (?, 3, 1)
        atten_wgt_in_step_T = Lambda(lambda x: adjust_atten_wgt_in_step(x), name="adjust_atten_wgt_in_step")(
            atten_wgt_in_step)
        '''<END> 8层求出context_vector'''
        # 将context_vector赋值给解码器输出
        context_vector = context_vector_mul(memoried_encoder_outputs_T)  # shape=(?, 40, 51, 64)
        future_decoder_outputs = context_vector
        future_decoder_outputs = decoder_dense_pre(future_decoder_outputs)
        future_decoder_outputs = Reshape((train_predict.shape[1], train_predict.shape[2], train_predict.shape[3], 1))(
            future_decoder_outputs)
        ##############################################    Model Compile    ##############################################
        COMPOSITE_ED = Model(inputs=[encoder_inputs, past_decoder_inputs],
                             outputs=[past_decoder_outputs, future_decoder_outputs])
        # COMPOSITE_ED.summary()
        rmsprop = optimizers.RMSprop(lr=params['lr'], rho=0.9, epsilon=None, decay=0.0)
        COMPOSITE_ED.compile(loss=[params['loss1'], params['loss2']], loss_weights=[params['rcstr_weight'], params['predict_weight']], optimizer=rmsprop)#9月14晚上只改了这里为[0.0,1.0]就训练并测试了。。。
        
    ##########    长度>1的变窗预测（特殊）    ##########
    else:
        # future_decoder_inputs = Input(shape=(
        # train_predict.shape[1], train_predict.shape[2], train_predict.shape[3], 1))  # 此时输入被设计好的future_decoder_inputs

        all_outputs = []
        max_decoder_seq_length = train_predict.shape[1]  # max_decoder_seq_length表示目标输出的时间戳长度，timesteps表示编码器输入的时间戳长度

        state_in_h=state_h_use
        state_in_c=state_c_use

        '''8月24日已加入attention'''
        '''唯一超参数是全连接层的unit数目，暂设128'''
        for i in range(max_decoder_seq_length):
            # attention

            '''<START> 求出context_vector'''
            memoried_encoder_outputs_T = encoder_outputs_for_mul(memoried_encoder_outputs)  # 已经改成经过memory的encoder
            atten_wgt_in_step = get_atten_wgt_layer(state_in_h)
            # shape(?,3,1)
            atten_wgt_in_step_T = adjust_atten_wgt_in_step_layer(atten_wgt_in_step)
            # atten_wgt_in_step_T形状变为 (?, 1, 1, 1, 3)
            # encoder_outputs_T形状变为(?, ?, ?, 3, ?)
            # 两者做 batch_dot
            context_vector = context_vector_mul(memoried_encoder_outputs_T)  # shape=(?, 40, 51, 64)
            # shape=(?, 1, 40, 51, 64)
            '''<END> 8层求出context_vector'''
            output, state_in_h, state_in_c = decoder_lstm_pre([context_vector, state_in_h, state_in_c])
            output = BN4(output)
            output = decoder_dense_pre(output)  # 把通道数降为1
            output = output_reshape(output)
            print("output shape", output.shape)
            all_outputs.append(output)
            # Reinject the outputs as inputs for the next loop iteration
            # as well as update the states
        # Concatenate all predictions
        # 层10
        outputs_concat = Lambda(lambda x: K.concatenate(x, axis=1), name="outputs_concat")
        future_decoder_outputs = outputs_concat(all_outputs)
        # 层11
        outputs_reshape = Reshape((train_predict.shape[1], train_predict.shape[2], train_predict.shape[3], 1),
                                  name="outputs_reshape")
        future_decoder_outputs = outputs_reshape(future_decoder_outputs)
        COMPOSITE_ED = Model(inputs=encoder_inputs, outputs=[past_decoder_outputs, future_decoder_outputs])
        # COMPOSITE_ED.summary()
        rmsprop = optimizers.RMSprop(lr=params['lr'], rho=0.9, epsilon=None, decay=0.0)
        COMPOSITE_ED.compile(loss=[params['loss1'], params['loss2']], loss_weights=[params['rcstr_weight'], params['predict_weight']],###记得修改loss_weights
                             optimizer=rmsprop)
        print("tiantiantian<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        ##############################################    Model saving    ##############################################

    return COMPOSITE_ED
# 返回的将要被存储的训练好参数的数据字典带有"tag"标签
def model_operation(COMPOSITE_ED):
    params, tag, ground_truth, train_input, train_predict, test_input, test_predict=get_settings_and_files()

    print("开始训练")
    print(train_input.shape, train_predict.shape, test_input.shape, test_predict.shape)
    #(4807, 5, 20, 52) (4807, 5, 20, 52) (4498, 5, 20, 52) (4498, 5, 20, 52)


    #打乱数据顺序
    np.random.seed(0)
    permutation = np.random.permutation(train_input.shape[0])
    print(permutation)
    train_input = train_input[permutation, :]
    train_predict = train_predict[permutation,:]

    train_input = train_input.reshape(train_input.shape[0],train_input.shape[1],train_input.shape[2],train_input.shape[3],1)

    train_predict = train_predict.reshape(train_predict.shape[0],train_predict.shape[1],train_predict.shape[2],train_predict.shape[3],1)
    train_zeros = np.zeros((train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3], 1))
    initial = np.zeros((train_input.shape[0],1,train_input.shape[2],train_input.shape[3],1)) #(smaple,1,20,51,1)  初始化状态
    print('input',train_input.shape, train_zeros.shape, initial.shape,train_predict.shape)
    print('Trying', params)
    train_input_rev = train_input[:, ::-1, :, :, :]
    initial = np.zeros((train_input.shape[0], train_predict.shape[1], train_input.shape[2],train_input.shape[3], 1))  # (smaple,1,20,51,1)

    if train_predict.shape[1] == 1:
        filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
        # COMPOSITE_ED.summary()
        history = COMPOSITE_ED.fit([train_input, train_zeros], [train_input_rev, train_predict],
                                   nb_epoch=params['nb_epochs'], batch_size=settings['batch_size'],
                                   callbacks=[checkpoint], validation_split=0.2)
    else:
        filepath = pathm + nowTime + 'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
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
    os.makedirs(pathc)
    pathm = pathc+"模型/"
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



