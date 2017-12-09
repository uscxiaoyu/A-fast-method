#coding=utf-8
from __future__ import division
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

def func(p, q, alpha, g):
    diff = Diffuse_gmm(p, q, alpha, g, num_runs=30)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    path = 'C:/Users/XIAOYU/Desktop/data/4 social influence rule/'
    txt_cont = ['gnm_random_graph(10000,30000),0.7', 'gnm_random_graph(10000,30000),0.9',
                'gnm_random_graph(10000,30000),1']
    alpha_cont = [0.7, 0.9, 1]
    for i, txt in enumerate(txt_cont):
        d = np.load(path + txt + '.npy')
        g = nx.gnm_random_graph(10000, 30000)
        t1 = time.clock()
        pool = multiprocessing.Pool(processes=6)
        result = []
        for p, q in d[:, :2]:
            result.append(pool.apply_async(func, (p, q, alpha_cont[i], g)))

        pool.close()
        pool.join()

        data = []
        for res in result:
            data.append(res.get())

        print i, txt, 'Time: %.2f s' % (time.clock() - t1)
        np.save('data\%s-gmm' % txt, data)
