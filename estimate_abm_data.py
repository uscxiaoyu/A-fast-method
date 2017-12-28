#for python3
import estimate_bass as eb
import numpy as np
import time
import os
import multiprocessing

file_list = []


def vst_dir(path, exclude='.pkl', include='.npy'):
    for x in os.listdir(path):
        sub_path = os.path.join(path, x)
        if os.path.isdir(sub_path):
            vst_dir(sub_path)
        else:
            if include in sub_path.lower() and exclude not in sub_path.lower():
                file_list.append(sub_path)


def func(x, para_range):
    p, q = x[:2]
    s_full = x[2:]
    max_ix = np.argmax(s_full)
    s = s_full[:max_ix + 2]
    bassest = eb.Bass_Estimate(s, para_range)
    bassest.t_n = 1000
    params = bassest.optima_search(c_n=200, threshold=10e-8)
    return [p, q] + list(params)

if __name__ == '__main__':
    '''
    diff_data = np.load(path + '/complete_graph(10000).npy')
    pool = multiprocessing.Pool(processes=6)
    para_range = [[1e-6, 0.1], [1e-4, 1], [0, 50000]]
    result = []
    t1 = time.clock()
    for x in diff_data:
        result.append(pool.apply_async(func, (x, para_range)))

    pool.close()
    pool.join()
    to_save = []
    for res in result:
        to_save.append(res.get())

    print(': Time elapsed: %.2fs' % (time.clock() - t1))
    np.save(path + '/estimate complete_graph(10000)', to_save)

    path = 'auto_data/'
    vst_dir(path)
    file_list = ['auto_data/watts_strogatz_graph(10000,6,0).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.1).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.3).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.5).npy']
    

    file_list = ['auto_data/watts_strogatz_graph(10000,6,0).npy', 'auto_data/watts_strogatz_graph(10000,6,0.1).npy',
                 'auto_data/gnm_random_graph(10000,30000),0.5.npy', 'auto_data/gnm_random_graph(10000,30000),0.7.npy',
                 'auto_data/gnm_random_graph(10000,30000),0.9.npy', 'auto_data/gnm_random_graph(10000,30000),1.0.npy']
                 
    '''
    file_list = ['auto_data/watts_strogatz_graph(10000,6,0).npy', 'auto_data/watts_strogatz_graph(10000,6,0.1).npy',
                 'auto_data/watts_strogatz_graph(10000,6,0.3).npy']
    for txt in file_list:
        diff_data = np.load(txt)
        pool = multiprocessing.Pool(processes=6)
        para_range = [[0.00002, 0.09], [0.005, 0.9], [0, 30000]]
        result = []
        t1 = time.clock()
        for x in diff_data:
            result.append(pool.apply_async(func, (x, para_range)))

        pool.close()
        pool.join()
        to_save = []
        for res in result:
            to_save.append(res.get())

        print(txt, ': Time elapsed: %.2fs' % (time.clock() - t1))
        np.save(txt[:10] + 'estimate_' + txt[10: -4], to_save)

