# for both python 2 and 3
from random import random, shuffle
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt

def _diffuse_(G, p, q, num_of_run=30):
    DG = G.to_directed()
    for i in DG:
        DG.node[i]['state'] = False

    non_adopt_set = list(DG)
    num_of_adopt = []
    for j in range(num_of_run):
        shuffle(non_adopt_set)
        x = 0
        for i in non_adopt_set:
            dose = sum([1 for k in DG.predecessors(i) if DG.node[k]['state']])
            prob = p + q * dose
            if random() <= prob:
                DG.node[i]['state'] = True
                non_adopt_set.remove(i)
                x += 1

        num_of_adopt.append(x)
    return num_of_adopt

def _diffuse_gmm_(G, p, q, num_of_run=30):
    DG = G.to_directed()
    for i in DG:
        DG.node[i]['state'] = False

    non_adopt_set = list(DG)
    num_of_adopt = []
    for j in range(num_of_run):
        shuffle(non_adopt_set)
        x = 0
        for i in non_adopt_set:
            dose = sum([1 for k in DG.predecessors(i) if DG.node[k]['state']])
            prob = 1 - (1 - p) * (1 - q) ** dose
            if random() <= prob:
                DG.node[i]['state'] = True
                non_adopt_set.remove(i)
                x += 1

        num_of_adopt.append(x)
    return num_of_adopt


if __name__ == '__main__':
    p_cont = [0.001 + i * 0.005 for i in range(5)]
    q_cont = [0.05 + i * 0.005 for i in range(20)]
    diff_cont = []
    for p in p_cont:
        for q in q_cont:
            s_estim = []
            t1 = time.clock()
            for i in range(10):
                G = nx.gnm_random_graph(10000, 30000)
                diffuse = _diffuse_gmm_(G, p, q)
                s_estim.append(diffuse)

            print(p, q, 'Time elapsed: %.2f s' % (time.clock() - t1))
            s_estim_avg = np.mean(s_estim, axis=0)
            diff_cont.append(np.concatenate(([p, q], s_estim_avg)))

    np.save('gnm_random_graph(10000,30000)_gmm', diff_cont)


'''
    diff_cont = [np.mean(s_estim[:(5 + i * 5)], axis=0) for i in range(6)]
    fig = plt.figure(figsize=(16, 8))
    for i in range(6):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(diff_cont[i], 'mo-', lw=1, label='repeates: %s' % ((i + 1) * 5))
        ax.legend(loc='best', fontsize=12)
        ax.set_ylim([0, 1000])
        if i == 0 or i == 3:
            ax.set_ylabel('Number of adopters', fontsize=15, style='italic')
        if i >= 3:
            ax.set_xlabel('Steps', fontsize=15, style='italic')
    plt.show()
'''
