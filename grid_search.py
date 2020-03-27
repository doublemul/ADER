#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : grid_search.py
# @Description  :

import subprocess
import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--is_early_stop', default=True, type=bool)
    args = parser.parse_args()

    dataset = args.dataset
    is_early_stop = args.is_early_stop
    cmd = 'python main.py --dataset=%s ' % dataset

    if not is_early_stop:
        for num_heads in [1]:
            for num_blocks in [1]:
                train = cmd + '--mode=train '
                save_dirs = [None] * 3
                train_p = [None] * 3
                for i, lr in enumerate([0.001, 0.0005, 0.0001]):
                    save_dirs[i] = 'Heads%dBlocks%dLr%f ' % (num_heads, num_blocks, lr)
                    train_cmd = train + '--save_dir=%s --num_heads=%d --num_blocks=%d --lr=%f' \
                                % (save_dirs[i], num_heads, num_blocks, lr)
                    train_p[i] = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.DEVNULL,
                                                  stderr=subprocess.STDOUT)
                list(map((lambda p, info: [p.wait(), print('train ' + info + str(p.poll()))]), train_p, save_dirs))
                try:
                    list(map((lambda p, info: [p.wait(), print('test ' + info + str(p.poll()))]), test_p, save_dirs))
                except NameError:
                    pass
                test = cmd + '--mode=test '
                test_p = [None] * 3
                for i, save_dir in enumerate(save_dirs):
                    train_cmd = test + '--save_dir=%s --device_num=1' % save_dir
                    test_p[i] = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        list(map((lambda p, info: [p.wait(), print('test ' + info + str(p.poll()))]), test_p, save_dirs))

    else:
        parallel_num, i = 2, 0
        early_stop = cmd + '--mode=early_stop --is_joint=True --batch_size=2048 --test_batch=128 '
        dirs = [None] * parallel_num
        p = [None] * parallel_num

        for num_heads in [1, 2, 3]:
            for num_blocks in [2, 3]:
                for lr in [0.0001, 0.0005, 0.001]:
                    try:
                        returncodes = list(map(lambda p: p.poll(), p))
                        while 0 not in returncodes:
                            returncodes = list(map(lambda p: p.poll(), p))
                        i = returncodes.index(0)
                        print('early stop ' + dirs[i] + ' done.')
                    except AttributeError:
                        pass
                    dirs[i] = 'Heads%dBlocks%dLr%f ' % (num_heads, num_blocks, lr)
                    early_stop_cmd = early_stop + '--save_dir=%s --num_heads=%d --num_blocks=%d --lr=%f ' \
                                                  '--device_num=%d' % (dirs[i], num_heads, num_blocks, lr, 1)
                    p[i] = subprocess.Popen(early_stop_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    i += 1
                    if i == parallel_num:
                        i = 0
        list(map(lambda p: p.wait() == 0, p))
        list(map(lambda info: print('early stop ' + info + ' done.'), dirs))
    print('Done')
