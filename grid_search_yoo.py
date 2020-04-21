#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : grid_search_yoo.py
# @Description  :

import subprocess
import time
import argparse

if __name__ == '__main__':

    dataset = 'YOOCHOOSE'
    batch_size = 128
    test_batch = 64
    device_num = 1

    common_cmd = 'python main.py --dataset=%s --desc=grid_search --is_joint=True --batch_size=%d --test_batch=%d ' \
          '--device_num=%d ' % (dataset, batch_size, test_batch, device_num)

    for num_heads in [1, 2, 3]:
        for num_blocks in [1, 2, 3]:
            for lr in [0.0001, 0.0005, 0.001]:
                if num_heads == 1 and num_blocks == 1 and lr == 0.0001:
                    continue
                dirs = 'Heads%dBlocks%dLr%f ' % (num_heads, num_blocks, lr)
                cmd = common_cmd + '--save_dir=%s --num_heads=%d --num_blocks=%d --lr=%f' \
                      % (dirs, num_heads, num_blocks, lr)
                p = subprocess.Popen(cmd, shell=True)
                p.wait()
                print(dirs + ' done.')
    print('Grid Search Done.')


