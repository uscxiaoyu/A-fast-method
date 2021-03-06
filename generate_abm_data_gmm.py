#coding=utf-8
import numpy as np
import networkx as nx
import time
import random
import os
import pickle
import multiprocessing


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, alpha=0, g=nx.gnm_random_graph(10000, 30000), num_runs=35):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.alpha = alpha
        self.num_runs = num_runs

    def decision(self, i):  # 线性决策规则
        dose = sum([self.g.node[k]['state'] for k in self.g.predecessors(i)])
        prob = 1 - (1 - self.p) * (1 - self.q) ** dose
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
    return G


def func(p, q, g):
    diff = Diffuse(p, q, g=g, num_runs=35)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    g_cont = [nx.gnm_random_graph(10000, 40000), nx.gnm_random_graph(10000, 50000), nx.gnm_random_graph(10000, 60000),
              nx.gnm_random_graph(10000, 70000), nx.gnm_random_graph(10000, 80000), nx.gnm_random_graph(10000, 90000),
              nx.gnm_random_graph(10000, 100000)]

    txt_cont = ['gnm_random_graph(10000,40000)', 'gnm_random_graph(10000,50000)', 'gnm_random_graph(10000,60000)',
                'gnm_random_graph(10000,70000)', 'gnm_random_graph(10000,80000)', 'gnm_random_graph(10000,90000)',
                'gnm_random_graph(10000,100000)']

    f = open('auto_data/bound(gmm).pkl')
    bound = pickle.load(f)

    for i, key in enumerate(txt_cont):
        r_p, r_q = bound[key][0]
        pq_cont = [(p, q) for p in np.linspace(r_p[0], r_p[1], num=10) for q in np.linspace(r_q[0], r_q[1], num=15)]
        g = g_cont[i]
        t1 = time.clock()
        pool = multiprocessing.Pool(processes=6)
        result = []
        for p, q in pq_cont:
            result.append(pool.apply_async(func, (p, q, g)))
        pool.close()
        pool.join()
        data = []
        for res in result:
            data.append(res.get())
        print(i + 1,  key, 'Time: %.2f s' % (time.clock() - t1))
        np.save('auto_data/' + key + '-gmm', data)
