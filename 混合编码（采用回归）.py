#一方面放大+dct+截取+回归+非均匀量化；另一方面0值周围直接不传
#用来预测力信号的高压缩率

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
T = 1

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


def AMP(arr, Mp):
    output = []
    for i in range(len(arr)):
        #arr[i] = int(arr[i] * Mp)
        #print("arr[", i, "]:", arr[i])
        value = arr[i] * Mp
        if value == 32:
            value = 31
        elif value == -32:
            value = -31
        output.append(value)
    return output

def IAMP(arr, Mp):
    output = []
    for i in range(len(arr)):
        value = arr[i] / Mp
        output.append(value)
    return output

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
        if ((i+1) % L != 0) and i != len(ori)-1: 
            X_value.append(ori[i])          # X_value加上每一个时刻的值
        elif ((i+1) % L == 0) or i == len(ori)-1:
            X_value.append(ori[i])          # 同上，X_value加上每一个时刻的值
            X.append(X_value)                  # 把数据集分成9个一小段
            t += 1
            for j in range(len(X_value)):
                if (i+1) == len(X_value) or t == T:
                    if j == 0:
                        #print("X_value[j]:", X_value[j])
                        #print("I_quantizer(quantizer(X_value[j])):", I_quantizer(quantizer(X_value[j])))
                        Encoder.append(int(X_value[j]))
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
                        difference = X_value[j] - (temp + slope * j)           # 计算当前时刻差值
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
                slope = 0
                if i == 0 or t == T:
                    Decoder = (value[j])
                    if len(value) >= 2:
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

def Split(arr):
    sp_arr1 = []
    sp_arr2 = []
    for i in range(len(arr)):
        if (i % 2 == 0):
            sp_arr1.append(arr[i])
        else:
            sp_arr2.append(arr[i])
    return sp_arr1, sp_arr2

def ISplit(sp_arr1, sp_arr2):
    isp_arr = []
    for i in range(len(sp_arr1) + len(sp_arr2)):
        value = (int)(i / 2)
        if (i % 2 == 0):
            isp_arr.append(sp_arr1[value])
        else:
            isp_arr.append(sp_arr2[value])
    return isp_arr

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
        ori = data1_F_x()

    if w == 1:
        ori = data1_F_y()

    if w == 2:
        ori = data1_F_z()

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #原始数据放大
    Max1 = ori[0]
    for i in range(len(ori)):
        if abs(ori[i]) > Max1:
            Max1 = abs(ori[i])
    Mp = 32 / Max1
    #print("Mp:", Mp)
    amp_arr = AMP(ori, Mp)
    Max = 32
    Encoder = []
    b = 40
    c = 60
    m = 0
    input = []
    sum_p = 0
    sum_q = 0
    temp1 = []
    #分割
    cut_arr = Cut_arr(amp_arr)
    #print("cut_arr:", cut_arr)
    for i in range(len(cut_arr)):
        value = cut_arr[i]
        if value[0] == 0 or i == len(cut_arr)-1:
            if i == len(cut_arr)-1:
                input.append(value)
                m == 0
            if m == 0 and i != 0:
                if len(input) >= L:
                    #print("input:", input)
                    #DCT变换
                    dct_arr = DCT(input)
                    #print("dct_arr:", dct_arr)
                    #转1维
                    d1_arr = D2TD1(dct_arr)
                    #print("d1_arr:", d1_arr)
                    #截取
                    in_arr = Intercept(d1_arr)
                    #print("in_arr:", in_arr)
                    #回归算法编码+量化
                    output, p, q = Regress(in_arr)
                    #print("output:", output)
                    Encoder.append(output)
                    sum_p += p
                    sum_q += q
                    
                else:
                    #print("input:", input)
                    temp1.append([c])
                    temp2 = []
                    for z in range(len(input)):
                        input_temp = input[z]
                        for l in range(len(input_temp)):
                            temp2.append(quantizer(input_temp[l]))
                        temp1.append(temp2)
                        sum_q += len(input[z])
                        temp2 = []
                    Encoder.append(temp1)
                    temp1 = []
                    
                input = []
            Encoder.append([[b]])
            m += 1
        else:
            input.append(value)
            m = 0
    
    #print("Encoder:", Encoder)

    ######@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@解码
    Decoder_X = []
    value2 = []
    for i in range(len(Encoder)):
        value = Encoder[i]
        #print("value:", i, value)
        value2 = value[0]
        #print("value2:", value2)
        if value2[0] == b:
            for j in range(length):
                Decoder_X.append(0)
        elif value2[0] == c:
            for j in range(len(value)-1):
                value3 = value[j+1]
                #print("value3", value3)
                for z in range(len(value3)):
                     Decoder_X.append(I_quantizer(value3[z]))
        else:
            ##回归算法解码+反量化
            output = IRegress(value)
            #print("Decoder_X:", Decoder_X)
            #反截取
            iin_arr = IIntercept(output)
            #print("iin_arr:", iin_arr)
            #打包
            p_arr = P_arr(iin_arr)
            #print("p_arr:", p_arr)
            #IDCT变换
            idct_arr = IDCT(p_arr)
            #print("idct_arr:", idct_arr)
            for j in range(len(idct_arr)):
                value3 = idct_arr[j]
                for z in range(len(value3)):
                    Decoder_X.append(value3[z])
    #print("Decoder_X:", Decoder_X)
    #编码数据缩小
    Decoder_X = IAMP(Decoder_X, Mp)
    

    #print("length =", length)
    #print("Inte =", Inte)
    #print("CR:", (len(ori * 64)/len(rlc_arr*8)))
    #print("Transmission_Rate:", (len(rlc_arr*8)/len(ori * 64))*100, "%")
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #去掉游程、VLI编码的压缩率
    p = sum_p
    q = sum_q
    #print("p:", p)
    #print("q:", q)
    CR = len(ori * 64)/(p*64+q*8)
    #print("CR:", CR)
    #print("Transmission_Rate:", (1 / CR)*100, "%")
    ##################
    #CR = (len(ori * 64)/(len(in_arr)*64))
    #print("CR:",CR )
    #print("Transmission_Rate:", (1 / CR)*100, "%")
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



    # 计算mse和psnr
    def mse_psnr(pre, lab):
        s = 0
        for i in range(min(len(pre), len(lab))):
            s += pow((lab[i] - pre[i]), 2)
        mse = s / (len(lab))
        psnr = 10 * math.log((36 / mse), 10)
        return mse, psnr

    def snr(pre, lab):
        s_lab = 0
        s_err = 0
        for i in range(min(len(pre), len(lab))):
            s_lab += pow(lab[i], 2)
            s_err += pow(lab[i] - pre[i], 2)
        snr =  10 * math.log(s_lab / s_err, 10)
        return snr

    mse, psnr = mse_psnr(Decoder_X, ori)

    SNR = snr(Decoder_X, ori)

    #print("MSE:", mse)

    #print("PSNR:", psnr)

    #print("SNR:", SNR)


    def HSSIM2(arr1, arr2):
        for i in range(min(len(arr1), len(arr2))):
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

    arr1 = copy.deepcopy(Decoder_X)
    arr2 = copy.deepcopy(ori)
    Sp = HSSIM2(arr1, arr2)
    #print("HSSIM::", Sp)

    ##绘制对比图像
    #i = 200
    #k = 3450
    #PSPLCKI_value = []
    #dataset_speed_norm_value = []
    #for j in range(i):
    #    error_value1 = Decoder_X[j + k]
    #    PSPLCKI_value.append(error_value1)
    #    dataset_speed_value = ori[j + k]
    #    dataset_speed_norm_value.append(dataset_speed_value)
    #    print("error_value1:", error_value1)

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

    SNR_xyz += SNR
    PSNR_xyz += psnr
    CR_xyz += CR
    HSSIM_xyz += Sp

print("CR_xyz:", CR_xyz / 3)
print("Compression_ratio_xyz:", 100 / (CR_xyz / 3), "%")
print("SNR_xyz:", SNR_xyz / 3)
print("PSNR_xyz:", PSNR_xyz / 3)
print("HSSIM_xyz:", HSSIM_xyz / 3)
