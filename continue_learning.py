#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : 
# @File         : main.py
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


def get_periods(args, logs):
    """
    This function returns list of periods for joint learning or continue learning
    :return: [0] for joint learning, [1, 2, ..., period_num] for continue learning
    """
    # if continue learning: periods = [1, 2, ..., period_num]
    datafiles = os.listdir(os.path.join('..', '..', 'data', args.dataset))
    period_num = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles))))
    logs.write('\nContinue Learning: Number of periods is %d.\n' % period_num)
    periods = range(1, period_num + 1)
    for period in periods:
        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
            os.makedirs(os.path.join('model', 'period%d' % period))
    return periods


def load_exemplars(period, use_exemplar):
    """
    This method load exemplar in previous period
    :param period: this period
    :return: exemplar list
    """
    if period > 1 and use_exemplar:
        exemplars = []
        with open('exemplar/Period=%d.pickle' % (period - 1), mode='rb') as file:
            exemplars_item = pickle.load(file)
        for item in exemplars_item.values():
            if isinstance(item, list):
                exemplars.extend([i for i in item if i])
        return exemplars
    else:
        return None


if __name__ == '__main__':

    gc.enable()

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIGINETICA', type=str)
    parser.add_argument('--save_dir', default='try', type=str)
    # continue learning parameter
    parser.add_argument('--use_exemplar', default=False, type=str2bool)
    parser.add_argument('--exemplar_size', default=5000)
    parser.add_argument('--is_herding', default=0, type=int)  # 0 for herding; 1 for random; 2 for loss
    # data parameter
    parser.add_argument('--is_joint', default=False, type=str2bool)
    parser.add_argument('--remove_item', default=True, type=str2bool)
    # early stop parameter
    parser.add_argument('--stop', default=5, type=int)
    # batch size and device
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    # hyper-parameters grid search
    parser.add_argument('--lr', default=0.0005, type=float)
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
    if not os.path.isdir(os.path.join('results', args.dataset + '_' + args.save_dir)):
        os.makedirs(os.path.join('results', args.dataset + '_' + args.save_dir))
    os.chdir(os.path.join('results', args.dataset + '_' + args.save_dir))
    # record logs
    logs = open('Training_logs.txt', mode='a')
    logs.write(' '.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # For reproducibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    item_num = 43136 if args.dataset == 'DIGINETICA' else 27333
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    # Loop each period for continue learning #
    periods = get_periods(args, logs)
    dataloader = DataLoader(args, logs)
    best_epoch = 0
    t_start = time.time()

    for period in periods:

        best_performance, performance = 0, 0
        t = time.time()

        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)

        # Load exemplar
        exemplar_data = load_exemplars(period, args.use_exemplar)

        # Load data
        train_sess = []
        for i in range(1, period + 1):
            train_sess.extend(dataloader.train_loader(i))
        # train_sess = dataloader.train_loader(period)
        train_item_count = dataloader.get_item_counter(exemplar_data)
        if period < periods[-1]:
            test_sess = dataloader.evaluate_loader(period)
        max_item = dataloader.max_item()

        # Start of the main algorithm
        with tf.Session(config=config) as sess:

            writer = tf.summary.FileWriter('logs/period%d' % period, sess.graph)
            saver = tf.train.Saver(max_to_keep=1)

            # initialize variables or reload from previous period
            sess.run(tf.global_variables_initializer())
            # if period <= 1:
            #     sess.run(tf.global_variables_initializer())
            # else:
            #     saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))

            # train
            train_sampler = Sampler(args, train_sess, args.batch_size)
            valid_sess = train_sampler.prepare_data(exemplar=exemplar_data, valid_portion=0.1)
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

                # validate performance
                valid_evaluator = Evaluator(args, [], max_item, 'valid', model, sess, logs)
                valid_evaluator.evaluate(epoch, valid_sess)
                performance = valid_evaluator.results()[1]

                # early stop
                best_epoch = epoch
                if best_performance > performance:
                    stop_counter += 1
                    if stop_counter >= args.stop:
                        best_epoch = epoch - args.stop
                        break
                else:
                    stop_counter = 0
                    best_performance = performance
                    saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))

            # test performance
            if period < periods[-1]:
                saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))
                test_evaluator = Evaluator(args, test_sess, max_item, 'test', model, sess, logs)
                test_evaluator.evaluate(best_epoch)

            # save exemplars
            if args.use_exemplar:
                if isinstance(args.exemplar_size, int):
                    exemplar_size = args.exemplar_size
                else:
                    exemplar_size = int(np.array(train_item_count).sum() / 5)
                exemplar = ExemplarGenerator(args, exemplar_size, train_sess, logs)
                exemplar.prepare_data(exemplar_data)
                if args.is_herding == 1:
                    exemplar.randomly_by_frequency(train_item_count)
                elif args.is_herding == 0:
                    exemplar.herding_by_frequency(train_item_count, sess, model)
                elif args.is_herding == 2:
                    exemplar.loss_by_frequency(train_item_count, sess, model)
                else:
                    raise ValueError('Invalid exemplar select method')
                exemplar.save(period)
                del exemplar

        print('Period %d time: %.2f minutes.' % (period, (time.time() - t) / 60.0))
        logs.write('Period %d time: %.2f minutes\n' % (period, (time.time() - t) / 60.0))

    print('Total time: %.2f minutes.' % ((time.time() - t_start) / 60.0))
    logs.write('Total time: %.2f minutes\nDone' % ((time.time() - t_start) / 60.0))
    logs.close()
    print('Done')
