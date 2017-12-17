# -*- coding: utf-8 -*-
from scipy.optimize import minimize
from math import e
import numpy as np
import time
import sympy as sy

class Bass_Estimate:
    def __init__(self, s):  # 初始化实例参数
        self.s, self.s_len = np.array(s), len(s)
        self.p, self.q, self.m = sy.symbols('p q m')
        sse = self.m * (1 - 1 / (e ** (self.p + self.q))) / (1 + self.q / (self.p * e ** (self.p + self.q))) - self.s[0]
        for i in range(1, self.s_len):
            sse += self.m * (1 - 1 / (e ** ((self.p + self.q) * (i + 1)))) / (1 + self.q / (self.p * e ** ((self.p + self.q) * (i + 1)))) - \
                   self.m * (1 - 1 / (e ** ((self.p + self.q) * i))) / (1 + self.q / (self.p * e ** ((self.p + self.q) * i))) - self.s[i]

        self.der_p = sy.diff(sse, self.p)
        self.der_q = sy.diff(sse, self.q)
        self.der_m = sy.diff(sse, self.m)

        self.der_p_2 = sy.diff(sse, self.p, 2)
        self.der_q_2 = sy.diff(sse, self.q, 2)
        self.der_m_2 = sy.diff(sse, self.m, 2)

        self.der_p_q = sy.diff(sse, self.p, self.q)
        self.der_p_m = sy.diff(sse, self.p, self.m)

        self.der_q_m = sy.diff(sse, self.q, self.m)

    def f(self, params):  # 如果要使用其它模型，可以重新定义
        p, q, m = params
        t_list = np.arange(1, self.s_len + 1)
        a = np.array([1 - e ** (- (p + q) * t) for t in t_list])
        b = np.array([1 + q / p * e ** (- (p + q) * t) for t in t_list])
        diffu_cont = m * a / b
        adopt_cont = np.zeros_like(diffu_cont)
        adopt_cont[0] = diffu_cont[0]
        for t in range(1, self.s_len):
            adopt_cont[t] = diffu_cont[t] - diffu_cont[t - 1]
        return adopt_cont

    def jacob(self, params):
        p_1 = self.der_p.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        q_1 = self.der_q.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        m_1 = self.der_m.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        return np.mat([[p_1], [q_1], [m_1]])

    def hessian(self, params):
        p_2 = self.der_p_2.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        q_2 = self.der_q_2.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        m_2 = self.der_m_2.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])

        p_q = self.der_p_q.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        p_m = self.der_p_m.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])

        q_m = self.der_q_m.subs([(self.p, params[0]), (self.q, params[1]), (self.m, params[2])])
        return np.mat([[p_2, p_q, p_m],
                          [p_q, q_2, q_m],
                          [p_m, q_m, m_2]])

    def sse(self, params):  # 定义适应度函数（mse）
        a = self.f(params)
        v = np.sum(np.square(self.s - a))
        return v / 2  # 均方误

    def r2(self, params):  # 求R2
        f_act = self.f(params)
        tse = np.sum(np.square(self.s - f_act))
        mean_y = np.mean(self.s)
        ssl = np.sum(np.square(self.s - mean_y))
        R_2 = (ssl - tse) / ssl
        return R_2

    def optima_search(self, ini_values, tol=1e-6):
        v_sse = self.sse(ini_values)
        i = 0
        while True:
            jac = self.jacob(ini_values)
            hes = self.hessian(ini_values)
            upd = hes.getI() * jac  # hes.getI()有问题
            ini_values = ini_values - np.array(upd.getT())
            v_sse2 = self.sse(ini_values)
            i += 1
            if abs(v_sse2 - v_sse) <= tol:
                break
        return {'sse':v_sse2, 'estimates':ini_values, 'steps':i}  # p,q,m,r2


if __name__=='__main__':
    data_set = {'room air conditioners': (np.arange(1949, 1962), [96, 195, 238, 380, 1045, 1230, 1267, 1828, 1586, 1673, 1800, 1580, 1500]),
                'color televisions': (np.arange(1963, 1971), [747, 1480, 2646, 5118, 5777, 5982, 5962, 4631]),
                'clothers dryers': (np.arange(1949, 1962), [106, 319, 492, 635, 737, 890, 1397, 1523, 1294, 1240, 1425, 1260, 1236]),
                'ultrasound': (np.arange(1965, 1979), [5, 3, 2, 5, 7, 12, 6, 16, 16, 28, 28, 21, 13, 6]),
                'mammography': (np.arange(1965, 1979), [2, 2, 2, 3, 4, 9, 7, 16, 23, 24, 15, 6, 5, 1]),
                'foreign language': (np.arange(1952, 1964), [1.25, 0.77, 0.86, 0.48, 1.34, 3.56, 3.36, 6.24, 5.95, 6.24, 4.89, 0.25]),
                'accelerated program': (np.arange(1952, 1964), [0.67, 0.48, 2.11, 0.29, 2.59, 2.21, 16.80, 11.04, 14.40, 6.43, 6.15, 1.15])}
    china_set = {'color tv': (np.arange(1997, 2013),[2.6, 1.2, 2.11, 3.79, 3.6, 7.33, 7.18, 5.29, 8.42, 5.68, 6.57, 5.49, 6.48, 5.42, 10.72, 5.15]),
                 'mobile phone': (np.arange(1997, 2013),[1.7, 1.6, 3.84, 12.36, 14.5, 28.89, 27.18, 21.33, 25.6, 15.88, 12.3, 6.84, 9.02, 7.82, 16.39, 7.39])}

    s = data_set['color televisions'][1]
    t1 = time.clock()
    bas_est = Bass_Estimate(s)
    res = bas_est.optima_search(np.array([0.001, 0.1, 1.5*sum(s)]), tol=1e-6)
    params = res['estimates']
    x = bas_est.f(params)
    r_2 = bas_est.r2(params)
    print('p:%.4f   q:%.4f   m:%d' % tuple(params))
    print('r^2: %.4f' % r_2)
    print('Time elapsed: %.2fs' % (time.clock() - t1))







