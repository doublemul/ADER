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

    common_cmd = 'python main.py --dataset=%s --use_exemplar=True --batch_size=%d --test_batch=%d --num_epochs=200 ' \
                 % (dataset, batch_size, test_batch)
    common_cmd = common_cmd + '--desc=herding_select_grid_search --select_mode=1 '

    p = []
    for exemplar_sizes in [[5000, 2000]]:
        for i, exemplar_size in enumerate(exemplar_sizes):

            dirs = 'HerdingByFrequencyFullItem%d ' % exemplar_size
            cmd = common_cmd + '--save_dir=%s --device_num=%d --exemplar_size=%d ' % (dirs, i % 2, exemplar_size)

            if i == 0:
                p.append(subprocess.Popen(cmd, shell=True))
            else:
                p.append(subprocess.Popen(cmd, shell=True, stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT))

        for pp in p:
            pp.wait()
    print('Grid Search Done.')
