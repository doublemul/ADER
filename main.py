#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : 
# @File         : main.py
# @Description  :
import argparse
import os
import time
import tensorflow.compat.v1 as tf
from SASRec import SASRec
from tqdm import tqdm
from util import *

if __name__ == '__main__':

    tf.disable_v2_behavior()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    args = parser.parse_args()
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    item_set = set()
    train_dataset = 'train_0'
    train_sess = load_data(train_dataset, item_set,
                           valid_portion=False, is_train=True, remove_item=False)
    valid_dataset = 'valid_0'
    valid_sess = load_data(valid_dataset, item_set,
                           valid_portion=False, is_train=False, remove_item=False)
    test_dataset = 'test_0'
    test_sess = load_data(test_dataset, item_set,
                          valid_portion=False, is_train=False, remove_item=False)

    num_batch = int(len(train_sess) / args.batch_size)
    max_item = max(list(item_set))

    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    model = SASRec(max_item, args)

    with tf.Session(config=config) as sess,\
            open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w') as f:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.global_variables_initializer())
        Time = 0.0
        t0 = time.time()
        for epoch in range(1, args.num_epochs + 1):
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                             desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                seq, pos, neg = sampler(train_sess, max_item, batch_size=args.batch_size, maxlen=args.maxlen)
                auc, loss, _, merged = sess.run([model.auc, model.loss, model.train_op, model.merged],
                                        {model.input_seq: seq, model.pos: pos, model.neg: neg,
                                         model.is_training: True})
            writer.add_summary(merged, epoch)
            if epoch % 1 == 0:
                counter = time.time() - t0
                Time += counter
                print('Evaluating', end='')
                t_test = evaluate(test_sess, max_item, model, args, sess)
                t_valid = evaluate(valid_sess, max_item, model, args, sess)
                print('\repoch:%d, time: %f(s), valid (MRR@20: %.4f, RECALL@20: %.4f), test (MRR@20: %.4f, RECALL@20: '
                      '%.4f)'
                      % (epoch, Time, t_valid[0], t_valid[1], t_test[0], t_test[1]))
                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
    print('Done')
