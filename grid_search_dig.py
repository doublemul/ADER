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
    parallel_num = 2
    lr = 0.0001

    common_cmd = 'python main.py --dataset=%s --batch_size=%d --test_batch=%d --num_epochs=200 --lr=%f '\
                 % (dataset, batch_size, test_batch, lr)
    common_cmd = common_cmd + '--desc=modified_model_grid_search --is_joint=True '

    parameter = []
    i = 0
    sub_para = []
    for num_blocks in [1, 2, 3]:
        for num_heads in [1, 2, 3]:
            sub_para.append([num_blocks, num_heads])
            i += 1
            if i % parallel_num == 0:
                parameter.append(sub_para)
                i = 0
                sub_para = []

    p = []
    for sub_para in parameter:
        for i, para in enumerate(sub_para):

            num_blocks, num_heads = para
            dirs = 'ModifiedBlock%dHead%dlr%.4f ' % (num_blocks, num_heads, lr)
            cmd = common_cmd + '--save_dir=%s --device_num=%d --num_heads=%d --num_blocks=%d ' \
                  % (dirs, i % 2, num_heads, num_blocks)

            if i == 0:
                p.append(subprocess.Popen(cmd, shell=True))
            else:
                p.append(subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))

        for pp in p:
            pp.wait()

    num_blocks, num_heads = 3, 3
    dirs = 'ModifiedBlock%dHead%dlr%.4f ' % (num_blocks, num_heads, lr)
    cmd = common_cmd + '--save_dir=%s --device_num=1 --num_heads=%d --num_blocks=%d ' % (dirs, num_heads, num_blocks)
    ppp = subprocess.Popen(cmd, shell=True)
    ppp.wait()

    print('Grid Search Done.')
