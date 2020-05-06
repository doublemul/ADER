#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : 
# @File         : joint_learning.py
# @Description  :
from memory_profiler import profile
import argparse
import os
import math
import tensorflow.compat.v1 as tf
from SASRec import SASRec
from tqdm import tqdm
import tracemalloc
from util import *
import gc
import sys
import time
from tfdeterminism import patch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    gc.enable()

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', type=str)
    parser.add_argument('--save_dir', default='compare_with_RepeatNet', type=str)
    parser.add_argument('--is_joint', default=True, type=str2bool)
    # test set
    parser.add_argument('--remove_item', default=True, type=str2bool)
    # early stop parameter
    parser.add_argument('--stop_iter', default=10, type=int)
    parser.add_argument('--display_interval', default=1, type=int)
    # batch size and device
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--test_batch', default=64, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    # hyper-parameters grid search
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    # hyper-parameter fixed
    parser.add_argument('--random_seed', default=555, type=int)
    parser.add_argument('--hidden_units', default=150, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    args = parser.parse_args()

    # set path
    if not os.path.isdir(os.path.join('results', args.dataset + '_joint_' + args.save_dir)):
        os.makedirs(os.path.join('results', args.dataset + '_joint_' + args.save_dir))
    os.chdir(os.path.join('results', args.dataset + '_joint_' + args.save_dir))
    # record logs
    logs = open('Training_logs.txt', mode='a')
    logs.write(' '.join([str(k) + ',' + str(v) + '\n' for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # For reproducibility
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    patch()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # set configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # build model
    if args.dataset == 'DIGINETICA':
        item_num = 43136
    elif args.dataset == 'diginetica':
        item_num = 43097
    else:
        item_num = 30470
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    # Load data
    dataloader = DataLoader(args, logs)
    train_sess = dataloader.train_loader()
    test_sess = dataloader.evaluate_loader()
    max_item = dataloader.max_item()

    # Start of the main algorithm
    stop_counter, best_epoch, best_performance = 0, 0, 0
    start = time.time()
    with tf.Session(config=config) as sess:

        writer = tf.summary.FileWriter('logs', sess.graph)
        saver = tf.train.Saver(max_to_keep=1)

        # initialize variables or reload from previous period
        sess.run(tf.global_variables_initializer())

        # train
        train_sampler = Sampler(args, train_sess, args.batch_size)
        valid_sess = train_sampler.prepare_data(valid_portion=0.1)
        batch_num = train_sampler.batch_num
        for epoch in range(1, args.num_epochs + 1):
            # train each epoch
            for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                          desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                seq, pos = train_sampler.sampler()
                sess.run(model.train_op, {model.input_seq: seq,
                                          model.pos: pos,
                                          model.is_training: True,
                                          model.lr: args.lr})

            # evaluate performance and early stop
            if epoch % args.display_interval == 0:
                # validate performance
                valid_evaluator = Evaluator(args, [], max_item, 'valid', model, sess, logs)
                valid_evaluator.evaluate(epoch, valid_sess)
                performance = valid_evaluator.results()[1]

                # early stop
                best_epoch = epoch
                if best_performance > performance:
                    stop_counter += 1
                    if stop_counter >= args.stop_iter:
                        best_epoch = epoch - args.stop_iter * args.display_interval
                        break
                else:
                    stop_counter = 0
                    best_performance = performance
                    saver.save(sess, 'model/epoch=%d.ckpt' % epoch)

        # record best valid performance
        print('Best valid performance = %f, at epoch %d' % (best_performance, best_epoch))
        logs.write('Best valid Recall@20 = %f, at epoch %d\n' % (best_performance, best_epoch))
        saver.restore(sess, 'model/epoch=%d.ckpt' % best_epoch)

        # test performance
        test_evaluator = Evaluator(args, test_sess, max_item, 'test', model, sess, logs)
        test_evaluator.evaluate(best_epoch)

    logs.write('Total time: %.2f minutes\nDone.' % ((time.time() - start) / 60.0))
    logs.close()
    print('Done')

