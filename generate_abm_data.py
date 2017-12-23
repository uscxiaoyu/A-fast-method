#coding=utf-8
import numpy as np
import networkx as nx
import time
import random
import os
import multiprocessing


class Diffuse:  # 默认网络结构为节点数量为10000，边为30000的随机网络
    def __init__(self, p, q, alpha=0, g=nx.gnm_random_graph(10000, 30000), num_runs=30):
        if not nx.is_directed(g):
            self.g = g.to_directed()
        self.p, self.q = p, q
        self.alpha = alpha
        self.num_runs = num_runs

    def decision(self, i):  # 线性决策规则
        dose = sum([self.g.node[k]['state'] for k in self.g.predecessors(i)])
        prob = self.p + self.q * dose
        #prob = self.p + self.q * (dose / self.g.in_degree(i) ** self.alpha) if self.g.in_degree(i) else self.p
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


file_list = []


def vst_dir(path, exclude='estimate', include='.npy'):
    for x in os.listdir(path):
        sub_path = os.path.join(path, x)
        if os.path.isdir(sub_path):
            vst_dir(sub_path)
        else:
            if include in sub_path.lower() and exclude not in sub_path.lower():
                file_list.append(sub_path)


def func(p, q, g):
    diff = Diffuse(p, q, g=g, num_runs=30)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    path = 'data/'
    new_path = 'new-data/'
    '''
    expon_seq = np.load('exponential_sequance.npy')
    gauss_seq = np.load('gaussian_sequance.npy')
    logno_seq = np.load('lognormal_sequance.npy')
    g_cont = [nx.barabasi_albert_graph(10000, 3), generate_random_graph(expon_seq), generate_random_graph(gauss_seq),
              nx.gnm_random_graph(10000, 100000), nx.gnm_random_graph(10000, 30000),
              nx.gnm_random_graph(10000, 40000), nx.gnm_random_graph(10000, 50000), nx.gnm_random_graph(10000, 60000),
              nx.gnm_random_graph(10000, 70000), nx.gnm_random_graph(10000, 80000), nx.gnm_random_graph(10000, 90000),
              generate_random_graph(logno_seq),
              nx.watts_strogatz_graph(10000, 6, 0), nx.watts_strogatz_graph(10000, 6, 0.1),
              nx.watts_strogatz_graph(10000, 6, 0.3), nx.watts_strogatz_graph(10000, 6, 0.5),
              nx.watts_strogatz_graph(10000, 6, 0.7), nx.watts_strogatz_graph(10000, 6, 0.9),
              nx.watts_strogatz_graph(10000, 6, 1)]
    

    vst_dir(path)
    file_list = sorted(file_list)
    #file_list = file_list[:4] + file_list[10:]
    file_list = file_list[4:10]
    alpha_cont = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    g = nx.gnm_random_graph(10000, 30000)
    for i, txt in enumerate(file_list):
        d = np.load(txt)
        alpha = alpha_cont[i]
        #g = g_cont[i]
        t1 = time.clock()
        pool = multiprocessing.Pool(processes=6)
        result = []
        for p, q in d[:, :2]:
            result.append(pool.apply_async(func, (p, q, alpha, g)))

        pool.close()
        pool.join()

        data = []
        for res in result:
            data.append(res.get())

        print(i, txt[5:-8], 'Time: %.2f s' % (time.clock() - t1))
        np.save('new-data/%s' % txt[5:-8], data)
        '''
    g_cont = [nx.gnm_random_graph(10000, 100000), nx.gnm_random_graph(10000, 30000),
              nx.gnm_random_graph(10000, 40000), nx.gnm_random_graph(10000, 50000), nx.gnm_random_graph(10000, 60000),
              nx.gnm_random_graph(10000, 70000), nx.gnm_random_graph(10000, 80000), nx.gnm_random_graph(10000, 90000)]
    txt_cont = ['gnm_random_graph(10000,100000)', 'gnm_random_graph(10000,30000)',
                'gnm_random_graph(10000,40000)', 'gnm_random_graph(10000,50000)', 'gnm_random_graph(10000,60000)',
                'gnm_random_graph(10000,70000)', 'gnm_random_graph(10000,80000)', 'gnm_random_graph(10000,90000)']
    path = 'new-data/'
    new_path = 'new-data/gnm/'
    for i, txt in enumerate(txt_cont):
        d = np.load(path + txt + '.npy')
        g = g_cont[i]
        t1 = time.clock()
        pool = multiprocessing.Pool(processes=6)
        result = []
        for p, q in d[:, :2]:
            result.append(pool.apply_async(func, (p, q, g)))

        pool.close()
        pool.join()

        data = []
        for res in result:
            data.append(res.get())

        print i,  txt, 'Time: %.2f s' % (time.clock() - t1)
        np.save(new_path + txt, data)
