#for python3
import estimate_bass as eb
import numpy as np
import time


data = np.load('gnm_random_graph(10000,30000)_new.npy')
params_cont = []
for i, x in enumerate(data):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 2]
    t1 = time.clock()
    para_range = [[1e-6, 0.1], [1e-4, 1], [0, 50000]]
    bassest = eb.Bass_Estimate(s, para_range)
    params = bassest.optima_search(c_n=100, threshold=10e-8)
    print(i, 'p:%.4f   q:%.4f   m:%d   r2:%.4f' % tuple(params), end=' ')
    print('Time elapsed: %.2fs' % (time.clock() - t1))
    params_cont.append([p, q] + list(params))

np.save('estimate gnm_random_graph(10000,30000)_new', params_cont)