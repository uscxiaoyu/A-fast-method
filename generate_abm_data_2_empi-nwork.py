# coding=utf-8
from estimate_bass import *
import numpy as np
import networkx as nx
import time
import random
import pickle


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, g=nx.gnm_random_graph(10000, 30000), num_runs=30):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.num_runs = num_runs

    def decision(self, i):  # 线性决策规则
        dose = sum([self.g.node[k]['state'] for k in list(self.g.predecessors(i))])
        prob = self.p + self.q*dose
        return True if random.random() <= prob else False

    def single_diffuse(self):  # 单次扩散
        for i in self.g:
            self.g.node[i]['state'] = False

        non_adopt_set = [i for i in self.g if not self.g.node[i]['state']]
        num_of_adopt = []
        for j in range(self.num_runs):
            x = 0
            random.shuffle(non_adopt_set)
            for i in non_adopt_set:
                if self.decision(i):
                    self.g.node[i]['state'] = True
                    non_adopt_set.remove(i)
                    x += 1
            num_of_adopt.append(x)
        return num_of_adopt

    def repete_diffuse(self, repetes=10):  # 多次扩散
        return [self.single_diffuse() for i in range(repetes)]


class Gen_para:
    def __init__(self, g, p_cont=(0.001, 0.02), q_cont=(0.08, 0.1), delta=(0.0005, 0.01)):
        self.p_cont = p_cont
        self.q_cont = q_cont
        self.d_p, self.d_q = delta
        self.g = g

    def add_data(self, p, q):
        diff = Diffuse(p, q, g=self.g)
        x = np.mean(diff.repete_diffuse(), axis=0)
        max_idx = np.argmax(x)
        s = x[: (max_idx + 2)]
        para_range = [[1e-6, 0.1], [1e-5, 0.8], [2000, 20000]]
        bassest = Bass_Estimate(s, para_range)
        bassest.t_n = 1000
        res = bassest.optima_search(c_n=200, threshold=10e-8)
        return res[:2]

    def identify_range(self):
        min_p, max_p = self.p_cont
        min_q, max_q = self.q_cont
        est_cont = [self.add_data(p, q) for p, q in ((min_p, min_q), (max_p, max_q))]
        i = 1
        while True:
            min_P, min_Q = est_cont[0]
            max_P, max_Q = est_cont[1]
            print(i, ' P:%.4f~%.4f' % (min_P, max_P), ' Q:%.4f~%.4f' % (min_Q, max_Q))
            c1, c2 = 0, 0
            if min_P > 0.0007 or min_p > 0.0005:  # in case of min_p < 0
                if min_p - self.d_p > 0:
                    min_p -= self.d_p
                else:
                    min_p *= 0.8
                c1 += 1
            if min_Q > 0.38:
                min_q -= self.d_q
                c1 += 1
            if max_P < 0.03:
                max_p += self.d_p
                c2 += 1
            if max_Q < 0.53:
                max_q += self.d_q
                c2 += 1
            i += 1
            if c1 + c2 == 0 or i == 25:
                break
            else:
                if c1 != 0:
                    est_cont[0] = self.add_data(min_p, min_q)
                if c2 != 0:
                    est_cont[1] = self.add_data(max_p, max_q)
        return [(min_p, max_p), (min_q, max_q)], [(min_P, max_P), (min_Q, max_Q)]

    def generate_sample(self, n_p=10, n_q=20):
        rg_p, rg_q = self.identify_range()
        sp_cont = [(p, q) for p in np.linspace(rg_p[0], rg_p[1], n_p) for q in np.linspace(rg_q[0], rg_q[1], n_q)]
        return sp_cont


def func(p, q, g):
    diff = Diffuse(p, q, g=g, num_runs=40)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    bound_dict = {}
    g_cont = []

    txt_cont = ['watts_strogatz_graph(10000,6,0)', 'watts_strogatz_graph(10000,6,0.1)',
                'watts_strogatz_graph(10000,6,0.3)']

    for j, g in enumerate(g_cont):
        t1 = time.process_time()
        print(j + 1, txt_cont[j])
        if j == 0:
            p_cont = (0.001, 0.015)
            q_cont = (0.15, 0.2)
            delta = (0.00031, 0.008)
        if j == 1:
            p_cont = (0.001, 0.015)
            q_cont = (0.11, 0.15)
            delta = (0.00031, 0.008)
        if j== 2:
            p_cont = (0.001, 0.015)
            q_cont = (0.1, 0.12)
            delta = (0.00031, 0.008)

        ger_samp = Gen_para(g=g, p_cont=p_cont, q_cont=q_cont, delta=delta)
        bound_dict[txt_cont[j]] = ger_samp.identify_range()
        print('  time: %.2f s' % (time.process_time() - t1))

    f = open('auto_data/bound.pkl', 'wb')
    pickle.dump(bound_dict, f)
    f.close()
    # f = open('auto_data/bound.pkl'); pickle.load(f)