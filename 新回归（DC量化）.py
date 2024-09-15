 #放大+dct+截取+回归+非均匀量化（DC量化）
 #用来预测位置信号

import random
import copy
import pandas as pd
import numpy as np
import math
from math import sqrt, acos, pi
from scipy.fftpack import dct, idct
from matplotlib import pyplot
import matplotlib.pylab as plt # 导入绘图包
import matplotlib.pyplot as mp
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#导入预处理数据
length = 8              # 这里要调参
a = 20                  # LDVLC的参数
Inte = 1                # 截取的长度，不能超过length
L = 5
T = 4

SNR_xyz = 0
PSNR_xyz = 0
CR_xyz = 0
Compression_ratio_xyz = 0
HSSIM_xyz = 0

def data1_V_x():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data1_V_y():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data1_V_z():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data1_F_x():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data1_F_y():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data1_F_z():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data1_P_x():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data1_P_y():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data1_P_z():
    ori = pd.read_csv("Dynamic_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori
##########################################################

def data2_V_x():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data2_V_y():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data2_V_z():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data2_F_x():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data2_F_y():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data2_F_z():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data2_P_x():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data2_P_y():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data2_P_z():
    ori = pd.read_csv("Dynamic_interaction_pushing_side_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


##########################################################

def data3_V_x():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data3_V_y():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data3_V_z():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data3_F_x():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data3_F_y():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data3_F_z():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data3_P_x():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data3_P_y():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data3_P_z():
    ori = pd.read_csv("Dynamic_iteraction_pressing_top_face_cube_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori
##########################################################

def data4_V_x():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data4_V_y():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data4_V_z():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data4_F_x():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data4_F_y():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data4_F_z():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data4_P_x():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data4_P_y():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data4_P_z():
    ori = pd.read_csv("Static_interaction_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori
##########################################################

def data5_V_x():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data5_V_y():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data5_V_z():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data5_F_x():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data5_F_y():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data5_F_z():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data5_P_x():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data5_P_y():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data5_P_z():
    ori = pd.read_csv("Static_interaction_dragging_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori
##########################################################

def data6_V_x():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data6_V_y():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data6_V_z():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MV_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data6_F_x():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data6_F_y():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data6_F_z():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['SF_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori


def data6_P_x():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_x'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data6_P_y():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_y'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori

def data6_P_z():
    ori = pd.read_csv("Static_interaction_tapping_surface_25s_RecordedTOPSession_DB.csv")
    ori = pd.DataFrame(ori, columns=['MP_z'])
    ori = np.array(ori.T)
    ori = ori[0]
    ori = list(ori)
    return ori
##########################################################

def Cut_arr(arr): #将序列切分为长度为25的序列段
    cut_arr = []
    for i in range(int(len(arr) / length)):
        cut_arr_ = []
        for j in range(length):
            cut_arr_.append(arr[i * length + j])
        cut_arr.append(cut_arr_)
    return cut_arr


def DCT(arr): #对每组数据进行dct变换
    dct_arr = []
    dct_arr_ = []
    #print("len(arr):", len(arr))
    for i in range(len(arr)):
        dct_arr_ = dct(arr[i], norm='ortho')
        #print("dct_arr_[", i, "]:", dct_arr_)
        dct_arr.append(dct_arr_)
    return dct_arr

def IDCT(arr):
    idct_arr = []
    # print(len(arr))
    for i in range(len(arr)):
        # print(arr[i])
        idct_arr_ = idct(arr[i], norm='ortho')
        # print(idct_arr_)
        idct_arr.append(idct_arr_)
    return idct_arr

def Q_arr(arr, m):
    q_arr = []
    n = 0
    for i in range(len(arr)):
        n += 1
        q_arr.append(int(arr[i] / (m * (n / 2)))) #m为待定量化系数
        if n == length:
            n = 0
    return q_arr

def IQ_arr(arr, m):
    iq_arr = []
    n = 0
    for i in range(len(arr)):
        n += 1
        iq_arr.append(arr[i] * (m * (n / 2)))
        if n == length:
            n = 0
    return iq_arr

def D2TD1(arr):
    arr = list(arr)
    D1_arr = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            D1_arr.append(arr[i][j])
    return D1_arr

def D1TD2(arr):
    d2_arr = []
    dec = []
    i = 0
    n = 0
    while i < len(arr):
        if len(dec) < length:
            dec.append(arr[i])
            i = i + 1
        else:
            #print("dec", n, ":", dec)
            d2_arr.append(dec)
            dec = []
            n = n + 1
    d2_arr.append(dec)
    return d2_arr

def P_arr(arr):
    p_arr = []
    p_arr_ = []
    for i in range(int(len(arr) / length)):
        for j in range(length):
            p_arr_.append(arr[i * length + j])
        p_arr.append(p_arr_)
        p_arr_ = []
    return p_arr

def dec2bin_(x):
    def q_dec2bin(x):  #整数部分转为二进制
        re = []
        x, m = divmod(x, 2)
        re.append(m)
        while x != 0:
            x, m = divmod(x, 2)
            re.append(m)
        re = list(reversed(re))
        return re

    b_ = []
    if x >= 0:
        b_.append(0)
    else:
        b_.append(1)
    x = abs(x)
    a_ = q_dec2bin(x)
    for i in range(7 - len(a_)):
        b_.append(0)
    for j in range(len(a_)):
        b_.append(a_[j])
    return b_



def f_dec2bin(x):  #整数部分转为二进制
    re = []
    b_ = []
    if x >= 0:
        b_.append(0)
    else:
        b_.append(1)
    x = abs(x)
    x, m = divmod(x, 2)
    re.append(m)
    while x != 0:
        x, m = divmod(x, 2)
        re.append(m)
    re = list(reversed(re))
    for i in range(3 - len(re)):
        b_.append(0)
    for j in range(len(re)):
        b_.append(re[j])
    return b_

def bin2dec(arr):
    dec_ = []
    for i in range(len(arr)):
        s = ''.join('%s' % id for id in (arr[i][1:]))
        t = int(s, 2)
        if arr[i][0] == 1:
            t = 0 - t
            dec_.append(t)
        else:
            dec_.append(t)
    return dec_


import sys
if sys.version > '3':
    def _byte(b):
        return bytes((b, ))
else:
    def _byte(b):
        return chr(b)


def AMP(arr):
    for i in range(len(arr)):
        #arr[i] = int(arr[i] * Mp)
        #print("arr[", i, "]:", arr[i])
        arr[i] = arr[i] * Mp
        if arr[i] == 128:
            arr[i] = 127
        elif arr[i] == -128:
            arr[i] = -127
    return arr

def IAMP(arr):
    for i in range(len(arr)):
        arr[i] = arr[i] / Mp
    return arr

def Gradient(mj):
    angle = acos(1 / sqrt(1 + np.square(mj)))
    return angle

def Regress(ori):
    X_value = []
    X = []
    Encoder = []                # 编码器
    Encoder_X = []
    n = 0
    p = 0
    q = 0
    k = 0.15                    # 死区阈值
    angle_max = 0.00001
    temp = ori[0]
    t = 0
    for i in range(len(ori)):
        #if i == 0:
        #    mj = 0
        #else:
        #    mj = ori[i]-ori[i-1]
        #angle = Gradient(mj)
        #if angle > angle_max:
        #    angle_max = angle
        #Qj = 1 - abs(angle / angle_max)
        ##print("Qj:", Qj)
        #L = int(L_max * (Qj))+1
        ##print("L:", L)
        if ((i+1) % L != 0): 
            X_value.append(ori[i])          # X_value加上每一个时刻的值
        elif ((i+1) % L == 0):
            X_value.append(ori[i])          # 同上，X_value加上每一个时刻的值
            X.append(X_value)                  # 把数据集分成9个一小段
            t += 1
            for j in range(len(X_value)):
                if (i+1) == len(X_value) or t == T:
                    if j == 0:
                        Encoder.append(quantizer(X_value[j]))
                        q += 1
                    if j == 1:
                        slope = (X_value[len(X_value)-1] - X_value[0]) / (len(X_value)-1)
                        #print("len(X_value)-1:", len(X_value)-1)
                        #print("X_value[len(X_value)-1]:", X_value[len(X_value)-1])
                        #print("slope:", slope)
                        Encoder.append(quantizer(slope))
                        q += 1
                    if j != 0 and j != 1:                     # 从第三个开始
                        difference = X_value[j] - (X_value[0] + slope * j)           # 计算当前时刻差值
                        dead_zone = np.abs(difference / ((X_value[0] + slope * j) + 0.00001))    # 计算死区大小
                        if dead_zone > k:         # 若死区值大于0.15或第二个值，则要传差值
                            Encoder.append(quantizer(difference))       # Encoder加上当前时刻差值                      
                            q += 1
                        else:
                            a = 20
                            Encoder.append(quantizer(a))                 # 在死区内，不用传                       
                            n += 1
                else:
                    if j == 0 and i >= len(X_value):
                        slope = (X_value[len(X_value)-1] - temp) / (len(X_value))
                        Encoder.append(quantizer(slope))
                        q += 1
                    if j != 0:                  
                        difference = X_value[j] - (X_value[0] + slope * j)           # 计算当前时刻差值
                        dead_zone = np.abs(difference / ((X_value[0] + slope * j) + 0.00001))    # 计算死区大小
                        if dead_zone > k:         # 若死区值大于0.15或第二个值，则要传差值
                            Encoder.append(quantizer(difference))       # Encoder加上当前时刻差值                      
                            q += 1
                        else:
                            Encoder.append(quantizer(a))                  # 在死区内，不用传                       
                            n += 1
            
            temp = temp + slope * len(X_value)
            if t == T:
                t = 0
            Encoder_X.append(Encoder)          # 得到最终的编码
            #print("i",i,"Encoder:", Encoder)
            Encoder = []
            X_value = []
    return Encoder_X, p, q

def IRegress(Encoder_X):
    Decoder = []                # 解码器
    Decoder_X = []
    t = 0
    for i in range(len(Encoder_X)):
        value = Encoder_X[i]          # value是第i小段
        #print("value:", value)
        t += 1
        for j in range(len(value)):
            if j == 0:    
                if i == 0 or t == T:
                    Decoder = I_quantizer(value[j])
                    slope = I_quantizer(value[j+1])
                else:
                    slope = I_quantizer(value[j])
                    Decoder = Decoder2 + slope
                Decoder_X.append(Decoder)          # 每小段的第一个值直接填补在Decoder_X上
            elif(j == len(value) - 1):
                Decoder2 = Decoder + slope * j
                Decoder_X.append(Decoder2)
            elif(I_quantizer(value[j]) != a):   # 若当前时刻编码值difference 
                Decoder_X.append(Decoder + slope * (j - 1) + I_quantizer(value[j]))            # 上一个值赋值给当前值
            elif(I_quantizer(value[j]) == a): 
                Decoder_X.append(Decoder + slope * j)            # 上一个值赋值给当前值
        if t == T:
            t = 0
    return Decoder_X

def Intercept(arr):
    in_arr = []
    n = 0
    for i in range(len(arr)):
        n += 1
        if n < (Inte+1):
            in_arr.append(arr[i]) #m为待定量化系数
        if n == length:
            n = 0
    return in_arr

def IIntercept(arr):
    iin_arr = []
    n = 0
    for i in range(len(arr)):
        n += 1
        #print("n:", n)
        if n < Inte:
            iin_arr.append(arr[i]) #m为待定量化系数
        if n == Inte:
            n = 0
            iin_arr.append(arr[i])
            for j in range(length-Inte):
                iin_arr.append(0)
    return iin_arr

def dec2bin(d1_arr):
    bin_arr = []
    for i in range(len(d1_arr)):
        bin_arr.append(dec2bin_(d1_arr[i]))
    return bin_arr

def dec2bin11(x):
    re = []
    x, m = divmod(x, 2)
    re.append(m)
    while x != 0:
        x, m = divmod(x, 2)
        re.append(m)
    while len(re) < 4:
        re.append(0)
    re = list(reversed(re))
    return re

def quantizer(x):
    scaler = Max / 4096  # 量化间隔
    pos = []
    if x > 0:
        pos.append(1)
    else:
        pos.append(0)
    pos_ = abs(x) / scaler  # 确定段落
    if pos_ <= 32:
        pos.append(0)
        pos.append(0)
        pos.append(0)
        interval = (32) / 16  # 确定段内间隔
        in_code = math.floor((pos_ - 0) / interval)  # 确定段内码
    elif 32 < pos_ and pos_ <= 64:
        pos.append(0)
        pos.append(0)
        pos.append(1)
        interval = (64 - 32) / 16
        in_code = math.floor((pos_ - 32) / interval)
    elif 64 < pos_ and pos_ <= 128:
        pos.append(0)
        pos.append(1)
        pos.append(0)
        interval = (128 - 64) / 16
        in_code = math.floor((pos_ - 64) / interval)
    elif 128 < pos_ and pos_ <= 256:
        pos.append(0)
        pos.append(1)
        pos.append(1)
        interval = (256 - 128) / 16
        in_code = math.floor((pos_ - 128) / interval)
    elif 256 < pos_ and pos_ <= 512:
        pos.append(1)
        pos.append(0)
        pos.append(0)
        interval = (512 - 256) / 16
        in_code = math.floor((pos_ - 256) / interval)
    elif 512 < pos_ and pos_ <= 1024:
        pos.append(1)
        pos.append(0)
        pos.append(1)
        interval = (1024 - 512) / 16
        in_code = math.floor((pos_ - 512) / interval)
    elif 1024 < pos_ and pos_ <= 2048:
        pos.append(1)
        pos.append(1)
        pos.append(0)
        interval = (2048 - 1024) / 16
        in_code = math.floor((pos_ - 1024) / interval)
    else:
        pos.append(1)
        pos.append(1)
        pos.append(1)
        interval = (4096 - 2048) / 16
        in_code = math.floor((pos_ - 2048) / interval)
    bin = dec2bin11(in_code)
    for i in range(len(bin)):
        pos.append(bin[i])
    #     matrix = counter(pos)
    return pos

def I_quantizer(x):
    a = x[0]  # 首位判断±            1
    b = x[1] * 4 + x[2] * 2 + x[3] * 1  # 2-4位判断段落             5
    coe = [2,2,4,8,16,32,64,128]  # 各个段落的间隔32/4
    base = [0, 32, 64, 128, 256, 512, 1024, 2048]
    c = x[4] * 8 + x[5] * 4 + x[6] * 2 + x[7] * 1 # 段内码                 4
    reback = c * coe[b] + base[b]  # 段内码还原         4*32+512=640
    fvalue = (reback / 4096) * Max       #              640 / 4096 * 128 = 20
    if a == 1:
        value = fvalue
    elif a == 0:
        value = 0 - fvalue
    return value


def Quantizer(d1_arr):
    q_arr = []
    for i in range(len(d1_arr)):
        q_arr.append(int(d1_arr[i]))
    return q_arr

def IQuantizer(q_arr):
    iq_arr = []
    for i in range(len(q_arr)):
        iq_arr.append(q_arr[i])
    return iq_arr

for w in range(3):

    if w == 0:
        ori = data2_P_x()

    if w == 1:
        ori = data2_P_y()

    if w == 2:
        ori = data2_P_z()

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #分割
    cut_arr = Cut_arr(ori)
    #print("cut_arr:", cut_arr)
    #DCT变换
    dct_arr = DCT(cut_arr)
    #print(dct_arr)
    #转1维
    d1_arr = D2TD1(dct_arr)
    #print("d1_arr:", d1_arr)
    #原始数据放大
    Max1 = d1_arr[0]
    for i in range(len(d1_arr)):
        if abs(d1_arr[i]) > Max1:
            Max1 = abs(d1_arr[i])
    Mp = 128 / Max1
    #print("Mp:", Mp)
    amp_arr = AMP(d1_arr)
    Max = 128
    #print("amp_arr:", amp_arr)
    #截取
    in_arr = Intercept(amp_arr)
    #print("in_arr:", in_arr)
    #回归算法编码+量化
    Encoder_X, p, q = Regress(in_arr)
    #print("Encoder_X:", Encoder_X)


    ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@解码
    #回归算法解码+反量化
    Decoder_X = IRegress(Encoder_X)
    #print("Decoder_X:", Decoder_X)
    #反截取
    iin_arr = IIntercept(Decoder_X)
    #print("iin_arr:", iin_arr)
    #编码数据缩小
    iamp_arr = IAMP(iin_arr)
    #print("iamp_arr:", iamp_arr)
    #打包
    p_arr = P_arr(iamp_arr)
    #print("p_arr:", p_arr)
    #IDCT变换
    idct_arr = IDCT(p_arr)
    #print("idct_arr:", idct_arr)
    #转一维
    iamp_arr = D2TD1(idct_arr)
    #print("len(iamp_arr):", len(iamp_arr))


    #print("length =", length)
    #print("Inte =", Inte)
    #print("CR:", (len(ori * 64)/len(rlc_arr*8)))
    #print("Transmission_Rate:", (len(rlc_arr*8)/len(ori * 64))*100, "%")
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #去掉游程、VLI编码的压缩率
    #print("p:", p)
    #print("q:", q)
    CR = len(ori * 64)/(p*64+q*8)
    #print("CR:", CR)
    #print("Transmission_Rate:", (1 / CR)*100, "%")
    #################
    #print("CR:", (len(ori * 64)/(len(in_arr)*64)))
    #print("Transmission_Rate:", ((len(in_arr)*64)/len(ori * 64))*100, "%")
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    # 计算mse和psnr
    def mse_psnr(pre, lab):
        s = 0
        for i in range(len(pre)):
            s += pow((lab[i] - pre[i]), 2)
        mse = s / (len(lab))
        psnr = 10 * math.log((36 / mse), 10)
        return mse, psnr

    def snr(pre, lab):
        s_lab = 0
        s_err = 0
        for i in range(len(pre)):
            s_lab += pow(lab[i], 2)
            s_err += pow(lab[i] - pre[i], 2)
        snr =  10 * math.log(s_lab / s_err, 10)
        return snr

    mse, psnr = mse_psnr(iamp_arr, ori)

    SNR = snr(iamp_arr, ori)

    #print("MSE:", mse)

    #print("PSNR:", psnr)

    #print("SNR:", SNR)

    #########################计算HSSIM

    #Max = abs(ori[0])

    #for i in range(len(ori)):

    #    if abs(ori[i]) > Max: 

    #        Max = abs(ori[i])

    #def compute_ssim(im1, im2, L, k1=0.01, k2=0.03):

    #    C1 = (k1*L)**2
    #    C2 = (k2*L)**2
    
    #    mu1 = np.mean(im1)          # x的均值
    #    mu2 = np.mean(im2)          # y的均值
    #    mu1_sq = mu1 * mu1                           # x的均值的平方
    #    mu2_sq = mu2 * mu2                           # y的均值的平方
    #    mu1_mu2 = mu1 * mu2                          # x的均值乘以y的均值
    #    sigma1_sq = np.mean(im1*im1) - mu1_sq
    #    sigma2_sq = np.mean(im2*im2) - mu2_sq
    #    sigmal2 = np.mean(im1*im2) - mu1_mu2
 
    #    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    #    return np.mean(np.mean(ssim_map))

    ## HSSIM的Xp = K*X^b的b等于1.8，K是放大系数
    ## r是指该点的范数（三维距离），所以分别算出每个点每个维度的SSIM，然后算该点的SSIM范数，最后再求和取平均
    ## 这里计算的是一维数据，无需算范数
    #SSIM = 0

    #for i in range(len(iamp_arr)):
    #    SSIM = compute_ssim(np.array(iamp_arr[i]),np.array(ori[i]),Max) + SSIM
    #HSSIM = SSIM / (len(iamp_arr))
    #print("HSSIM:", HSSIM)
    #################################
    def HSSIM2(arr1, arr2):
        for i in range(len(arr1)):
            arr1[i] = pow(arr1[i], 2) * 1.4
            arr2[i] = pow(arr2[i], 2) * 1.4
        arr1 = pd.Series(arr1)
        arr2 = pd.Series(arr2)
        arr1_avg = sum(arr1) / len(arr1) #求均值
        arr2_avg = sum(arr2) / len(arr2)
        std_arr1 = np.std(arr1,ddof=1) #求标准差
        std_arr2 = np.std(arr2,ddof=1)
        cov_ab = sum([(x - arr1_avg)*(y - arr2_avg) for x,y in zip(arr1, arr2)])
        sq = math.sqrt(sum([(x - arr1_avg)**2 for x in arr1])*sum([(x - arr2_avg)**2 for x in arr2]))
        corr_factor = cov_ab / sq #求相关系数
        l_ = (2 * (arr1_avg * arr2_avg) + 0.006) / (pow(arr1_avg, 2) + pow(arr2_avg, 2) + 0.006)
        C_ = (2 * (std_arr1 * std_arr2) + 0.6) / (pow(std_arr1, 2) + pow(std_arr2, 2) + 0.6)
        S_ = (corr_factor + 0.06) / (std_arr1 * std_arr2 + 0.06)
        Sp = pow(l_ * C_, 8) 
        return Sp

    arr1 = copy.deepcopy(iamp_arr)
    arr2 = copy.deepcopy(ori)
    Sp = HSSIM2(arr1, arr2)
    #print("HSSIM::", Sp)

    ##绘制对比图像
    #i = 21000
    #k = 0
    #PSPLCKI_value = []
    #dataset_speed_norm_value = []
    #ori = data5_P_z()
    #for j in range(i):
    #    error_value1 = iamp_arr[j + k]
    #    PSPLCKI_value.append(error_value1)
    #    dataset_speed_value = ori[j + k]
    #    dataset_speed_norm_value.append(dataset_speed_value)


    #mp.gcf().set_facecolor(np.ones(3) * 240/255)#设置背景色
    #fig, ax1 = plt.subplots() # 使用subplots()创建窗口
    #ax1.plot(PSPLCKI_value,'-', c='blue',label='predict', linewidth = 1) #绘制折线图像1,圆形点，标签，线宽
    #ax1.plot(dataset_speed_norm_value, '-', c='black',label='true', linewidth = 1)
    #mp.legend(loc='best')

    #ax1.set_xlabel('time', size=18) #与原始matplotlib设置参数略有不同，使用自己下载的中文宋体，参数位置不可改变
    #ax1.set_ylabel('value', size=18)
    #mp.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴
    #mp.legend()#显示折线的意义
    #plt.show()

    ##生成写入文件
    #result2txt = str(iamp_arr)  # data是前面运行出的数据，先将其转为字符串才能写入
    #with open('e_flg.txt', 'w') as file_handle:  # .txt可以不自己新建,代码会自动新建
    #    file_handle.write(result2txt)  # 写入
    #    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
    #result2txt = str(ori)  # data是前面运行出的数据，先将其转为字符串才能写入
    #with open('e_tac.txt', 'w') as file_handle:  # .txt可以不自己新建,代码会自动新建
    #    file_handle.write(result2txt)  # 写入
    #    file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据

    SNR_xyz += SNR
    PSNR_xyz += psnr
    CR_xyz += CR
    HSSIM_xyz += Sp

print("CR_xyz:", CR_xyz / 3)
print("Compression_ratio_xyz:", 100 / (CR_xyz / 3), "%")
print("SNR_xyz:", SNR_xyz / 3)
print("PSNR_xyz:", PSNR_xyz / 3)
print("HSSIM_xyz:", HSSIM_xyz / 3)
