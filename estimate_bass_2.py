# -*- coding: utf-8 -*-
from scipy.optimize import minimize
import numpy as np
import time

def f(params, d):  # 如果要使用其它模型，可以重新定义
    '''
    :param params: p, q, m
    :param d: 时间
    :return: 非累积扩散数据
    '''
    p, q, m = params
    t_list = np.arange(1, d + 1)
    a = 1 - np.exp(-(p + q) * t_list)
    b = 1 + q / p * np.exp(-(p + q) * t_list)
    diff_cont = m * a / b
    adopt_cont = np.zeros_like(diff_cont)
    adopt_cont[0] = diff_cont[0]
    for t in range(1, d):
        adopt_cont[t] = diff_cont[t] - diff_cont[t - 1]
    return adopt_cont


def fitness(params, s=np.ones(25)):
    '''
    :param x: 拟合数据
    :param s: 实际数据
    :return: 均方误
    '''
    d = len(s)
    x = f(params, d=d)
    sse = np.sum(np.square(s - x))
    return sse / d  # 均方误


def mse(x, s):  # 定义适应度函数（mse）
    sse = np.sum(np.square(s - x))
    return np.sqrt(sse) / len(s)  # 均方误


def r2(x, s):  # 求R2
    tse = np.sum(np.square(s - x))
    mean_y = np.mean(s)
    ssl = np.sum(np.square(s - mean_y))
    r_2 = (ssl - tse) / ssl
    return r_2


if __name__=='__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24, 4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40, 6.43, 6.15, 1.15])}
    china_set = {'color tv': (np.arange(1997, 2013),[2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57, 5.49, 6.48, 5.42, 10.72,
                               5.15]),
                 'mobile phone': (np.arange(1997, 2013),[1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88, 12.3, 6.84, 9.02,
                                   7.82, 16.39, 7.39])}

    s = data_set['room air conditioners'][1]
    t1 = time.clock()
    res = minimize(fitness, np.array([0.001, 0.7, 1.5*sum(s)]), args=(s, ), method='Nelder-mead', tol=1e-8,
                   options={ 'disp': True})
    params = res.x
    x = f(params, len(s))
    r_2 = r2(x, s)
    print('p:%.4f   q:%.4f   m:%d' % tuple(params))
    print('r^2: %.4f' % r_2)
    print('Time elapsed: %.2fs' % (time.clock() - t1))
