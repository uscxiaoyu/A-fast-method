# coding=utf-8
import numpy as np
import networkx as nx
import time
import random
import multiprocessing


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, alpha=0, g=nx.gnm_random_graph(10000, 30000), num_runs=35):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.alpha = alpha
        self.num_runs = num_runs

    def decision(self, i):  # 线性决策规则
        dose = sum([self.g.node[k]['state'] for k in list(self.g.predecessors(i))])
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
        prob = 1 - (1 - self.p)*(1 - self.q)**(dose / self.g.in_degree(i) ** self.alpha) \
            if self.g.in_degree(i) else self.p
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
    g_1 = nx.read_gpickle('/Users/xiaoyu/IPython notebook/facebook.gpickle')
    g_2 = nx.read_gpickle('/Users/xiaoyu/IPython notebook/epinions.gpickle')
    g_cont = [g_1, g_2]

    txt_cont = ['facebook_network', 'epinions_network']
    bound_dict = {'facebook_network': ([(1.1744051200000018e-05, 0.019),(0.15, 0.2)],
                                    [(0.0005851833355511311, 0.03942370539955275),
                                    (0.08566117951300223, 0.6049901066003283)]),
                  'epinions_network': ([(7.000000000000008e-05, 0.019), (0.11, 0.15)],
                                    [(0.0003985278742605879, 0.03002370660942767),
                                    (0.22768315273114983, 0.5356097641357954)])}

    for i, key in enumerate(sorted(bound_dict.keys())):
        r_p, r_q = bound_dict[key][0]
        pq_cont = [(p, q) for p in np.linspace(r_p[0], r_p[1], num=10) for q in np.linspace(r_q[0], r_q[1], num=15)]
        g = g_cont[i]
        t1 = time.process_time()
        pool = multiprocessing.Pool(processes=6)
        result = []
        for p, q in pq_cont:
            result.append(pool.apply_async(func, (p, q, g)))
        pool.close()
        pool.join()
        data = []
        for res in result:
            data.append(res.get())
        print(i+1,  key, 'Time: %.2f s' % (time.process_time()-t1))
        np.save('auto_data/'+key, data)