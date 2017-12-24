#for python3
import estimate_bass as eb
import numpy as np
import time
import os
import multiprocessing

path = 'data/'
file_list = []


def vst_dir(path, exclude='estimate', include='.npy'):
    for x in os.listdir(path):
        sub_path = os.path.join(path, x)
        if os.path.isdir(sub_path):
            vst_dir(sub_path)
        else:
            if include in sub_path.lower() and exclude not in sub_path.lower():
                file_list.append(sub_path)


vst_dir(path)