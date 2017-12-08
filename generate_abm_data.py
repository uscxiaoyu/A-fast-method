#coding=utf-8
import numpy as np
import networkx as nx
import time
import random


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


class Diffuse_gmm(Diffuse):
    def decision(self, i):  # gmm决策规则
        dose = sum([self.g.node[k]['state'] for k in self.predecessors(i)])
        prob = 1 - (1 - self.p) * (1 - self.q) ** dose
        return True if random.random() <= prob else False


if __name__ == '__main__':
    pq_range = [(i, j) for i in np.linspace(0.001, 0.021, num=5) for j in np.linspace(0.05, 0.145, num=20)]
    result = []
    u = 1
    g = nx.gnm_random_graph(10000, 30000)
    for p, q in pq_range:
        t1 = time.clock()
        diff = Diffuse_gmm(p, q, g=g, num_runs=30)
        x = np.mean(diff.repete_diffuse(), axis=0)
        result.append(np.concatenate(([p, q], x)))
        print u, 'Time: %.2f s' % (time.clock() - t1)
        u += 1

    np.save('gnm_random_graph(10000, 30000)-gmm', result)