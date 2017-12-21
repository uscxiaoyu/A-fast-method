#coding=utf-8
from estimate_bass import *
import numpy as np
import networkx as nx
import time
import random
import multiprocessing


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, g=nx.gnm_random_graph(10000, 30000), num_runs=30):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.num_runs = num_runs

    def decision(self, i):  # 线性决策规则
        dose = sum([self.g.node[k]['state'] for k in self.g.predecessors(i)])
        prob = self.p + self.q * dose
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


class Diffuse_gmm(Diffuse):  # social influence
    def __init__(self, p, q, alpha, g=nx.gnm_random_graph(10000, 30000), num_runs=30):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.alpha = alpha
        self.num_runs = num_runs

    def decision(self, i):  # gmm决策规则
        dose = sum([self.g.node[k]['state'] for k in self.g.predecessors(i)])
        prob = 1 - (1 - self.p) * (1 - self.q) ** (dose / self.g.in_degree(i) ** self.alpha) if self.g.in_degree(i) else self.p
        return True if random.random() <= prob else False


def generate_random_graph(degre_sequance):
    G = nx.configuration_model(degre_sequance, create_using=None, seed=None)
    G = nx.Graph(G)
    G.remove_edges_from(G.selfloop_edges())

    num_of_edges = nx.number_of_edges(G)
    edges_list = G.edges()

    if num_of_edges > 30000:
        edges_to_drop = num_of_edges - 30000
        x = np.random.choice(num_of_edges, edges_to_drop, replace=False)
        for i in x:
            a, b = edges_list[i]
            G.remove_edge(a, b)
    elif num_of_edges < 30000:
        edges_to_add = 30000 - num_of_edges
        x = np.random.choice(10000, edges_to_add * 2, replace=False)
        to_add_list = zip(x[:edges_to_add], x[edges_to_add:])
        G.add_edges_from(to_add_list)
    else:
        pass

    return G


def add_data(p, q, diff_data, est_data, g):
    diff = Diffuse(p, q, g)
    x = np.mean(diff.repete_diffuse(), axis=0)
    d = np.array(np.concatenate(([p, q], x)))
    n_diff_data = np.concatenate((diff_data, d), axis=0)

    para_range = [[1e-6, 0.1], [1e-5, 0.8], [0.5 * sum(x), 5 * sum(x)]]
    bassest = Bass_Estimate(x, para_range)
    res = bassest.optima_search(c_n=100, threshold=10e-8)
    y = [[p, q] + list(res)]
    n_est_data = np.concatenate((est_data, y), axis=0)
    return n_diff_data, n_est_data


def adjust_range(diff_data, est_data, g):
    delta_p = 0.0025
    delta_q = abs(diff_data[1][1] - diff_data[1][0])
    c = 0
    while True:
        min_est = np.min(est_data, axis=0)
        max_est = np.max(est_data, axis=0)
        p_range = [min_est[0], max_est[0]]
        q_range = [min_est[1], max_est[1]]
        P_range = [min_est[2], max_est[2]]
        Q_range = [min_est[3], max_est[3]]
        flag = True  # 检验范围是否变化
        if P_range[0] > 0.0007 or Q_range[0] > 0.38:  # 高于下限
            if p_range[0] - delta_p > 0:  # 防止p小于0的情况
                p_range[0] -= delta_p

            q_range[0] -= delta_q
            p, q = p_range[0], q_range[0]
            diff_data, est_data = add_data(p, q, diff_data, est_data, g)
            c += 1
            flag = False

        if P_range[1] < 0.03 or Q_range[1] < 0.58:  # 低于上限
            p_range[0] += delta_p
            q_range[0] += delta_q
            p, q = p_range[0], q_range[0]
            diff_data, est_data = add_data(p, q, diff_data, est_data, g)
            c += 1
            flag = False

        peak_idx = [np.argmax(x[2:]) for x in diff_data]
        peak_range = [min(peak_idx), max(peak_idx)]
        if peak_range[1] < 25:  # 低于上限
            q_range[0] -= delta_q
            p, q = p_range[0], q_range[0]
            diff_data, est_data = add_data(p, q, diff_data, est_data, g)
            c += 1
            flag = False

        if flag:
            break

    return diff_data, est_data


if __name__ == '__main__':
    new_path = 'new-data/'
    g = nx.gnm_random_graph(1000, 30000)
    diff_data = np.load(path + )
