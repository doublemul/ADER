#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : grid_search_dig.py
# @Description  :

import subprocess
import time
import argparse

if __name__ == '__main__':

    dataset = 'DIGINETICA'
    batch_size = 256
    test_batch = 32
    device_num = 0

    common_cmd = 'python main.py --dataset=%s --use_exemplar=True --batch_size=%d --test_batch=%d --device_num=%d ' \
                 % (dataset, batch_size, test_batch, device_num)
    common_cmd = common_cmd + '--num_epochs=200 '
    common_cmd = common_cmd + '--desc=Random --select_mode=0 '

    # p = []
    for exemplar_size in [5000, 10000, 20000, 2000]:
        dirs = 'RandomByFrequency%d ' % exemplar_size
        cmd = common_cmd + '--save_dir=%s --exemplar_size=%d ' % (dirs, exemplar_size)

        p = subprocess.Popen(cmd, shell=True)
        p.wait()

        # p.append(subprocess.Popen(cmd, shell=True))
    # for pp in p:
    #     pp.wait()
    print('Grid Search Done.')
