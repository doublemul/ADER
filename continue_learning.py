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
    # return periods
    return [1, 2, 3, 4, 5, 6, 7, 8, 9]


def load_exemplars(period):
    """
    This method load exemplar in previous period
    :param period: this period
    :return: exemplar list
    """
    exemplars = []
    with open('exemplar/Period=%d.pickle' % period, mode='rb') as file:
        exemplars_item = pickle.load(file)
    for item in exemplars_item.values():
        if isinstance(item, list):
            exemplars.extend([i for i in item if i])
    return exemplars


if __name__ == '__main__':

    gc.enable()

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIGINETICA', type=str)
    parser.add_argument('--save_dir', default='NewMethod5k-oldExample-true', type=str)
    # continue learning parameter
    parser.add_argument('--use_exemplar', default=True, type=str2bool)
    parser.add_argument('--exemplar_size', default=5000, type=int)
    parser.add_argument('--is_herding', default=2, type=int)  # 0 for herding; 1 for random; 2 for loss
    # ewc parameter
    parser.add_argument('--use_ewc', default=False, type=str2bool)
    parser.add_argument('--ewc_lambda', default=10, type=float)
    # data parameter
    parser.add_argument('--is_joint', default=False, type=str2bool)
    parser.add_argument('--remove_item', default=True, type=str2bool)
    # early stop parameter
    parser.add_argument('--stop', default=5, type=int)
    # batch size and device
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--device_num', default=1, type=int)
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
    logs = open('Training_logs.txt', mode='w')
    logs.write(' '.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # For reproducibility
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # build model
    item_num = 27333 if args.dataset == 'YOOCHOOSE' else 43136
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    # Loop each period for continue learning #
    periods = get_periods(args, logs)
    dataloader = DataLoader(args, logs)
    best_epoch = 0
    t_start = time.time()

    for period in periods[:-1]:

        best_performance, performance = 0, 0
        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)

        # Load data
        if args.use_exemplar and period > 1:
            exemplar_data = []
            for p in range(1, period):
                exemplar_data.extend(load_exemplars(p))
        else:
            exemplar_data = None
        # train data
        train_sess = dataloader.train_loader(period)
        train_item_count = dataloader.get_item_counter(exemplar=exemplar_data)
        test_sess = dataloader.evaluate_loader(period)
        max_item = dataloader.max_item()

        # set loss
        if period > 1:
            # model.update_ewc_loss(ewc_lambda=args.ewc_lambda)
            model.update_icarl_loss(train_size=args.batch_size, icarl_lambda=0.2)
            # model.set_vanilla_loss()
        else:
            model.set_vanilla_loss()
            continue

        # Start of the main algorithm
        with tf.Session(config=config) as sess:

            # writer = tf.summary.FileWriter('logs/period%d' % period, sess.graph)
            saver = tf.train.Saver(max_to_keep=1)

            # initialize variables or reload from previous period
            if period > 1:
                if period == 2:
                    saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (1, 40))
                else:
                    saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
                exemplar_samplar = Sampler(args, [], args.batch_size)
                exemplar_samplar.prepare_data()
                exemplar_samplar.add_exemplar(exemplar=exemplar_data)
                valid_exemplar, train_exemplar = exemplar_samplar.split_data(valid_portion=0.1, return_train=True)
                del exemplar_samplar
            else:
                sess.run(tf.global_variables_initializer())

            # train
            train_sampler = Sampler(args, train_sess, args.batch_size)
            train_sampler.prepare_data()
            valid_sess = train_sampler.split_data(valid_portion=0.1)
            # train_sampler.add_exemplar(exemplar=train_exemplar)
            train_sampler.shuffle_data()
            batch_num = train_sampler.batch_num

            if period > 1:
                valid_sess.extend(valid_exemplar)
                exemplar_batch = int(len(train_exemplar)/batch_num)
                exemplar_samplar = Sampler(args, [], exemplar_batch)
                exemplar_samplar.prepare_data()
                exemplar_samplar.add_exemplar(train_exemplar)
                exemplar_samplar.shuffle_data()
            for epoch in range(1, args.num_epochs + 1):

                # train each epoch
                for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                              desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                    seq, pos = train_sampler.sampler()
                    if period > 1:
                        ex_seq, ex_pos = exemplar_samplar.sampler()
                        seq = seq + ex_seq
                        pos = pos + ex_pos
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

                # if period > 1 and args.use_ewc:
                #     loss, ewc_loss = sess.run([model.loss, model.ewc_loss], {model.input_seq: seq,
                #                                                              model.pos: pos,
                #                                                              model.is_training: False})
                #     print('loss=%.12e' % loss)
                #     print('ewc loss=%.12e' % (ewc_loss - loss))
                #     model.variables_prev = sess.run(model.variables)
                #     random_exemplar = random.sample(exemplar_data, min(len(exemplar_data), 1000))
                #     model.compute_fisher(sess, random_exemplar, 50)
                #     # model.compute_fisher(sess, exemplar_data, 50)

            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))
            # test performance
            if period < periods[-1]:
                test_evaluator = Evaluator(args, test_sess, max_item, 'test', model, sess, logs)
                test_evaluator.evaluate(best_epoch)

            # save exemplars
            if args.use_exemplar:
                exemplar = ExemplarGenerator(args, args.exemplar_size, train_sess, logs)
                exemplar.prepare_data(exemplar=exemplar_data)
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

            # if args.use_ewc:
            #     if not exemplar_data:
            #         exemplar_data = []
            #     exemplar_data.extend(load_exemplars(period))
            #     model.variables_prev = sess.run(model.variables)
            #     random_exemplar = random.sample(exemplar_data, min(len(exemplar_data), 1000))
            #     model.compute_fisher(sess, random_exemplar, 50)
            #     # model.compute_fisher(sess, exemplar_data, 50)

    print('Total time: %.2f minutes.' % ((time.time() - t_start) / 60.0))
    logs.write('Total time: %.2f minutes\nDone' % ((time.time() - t_start) / 60.0))
    logs.close()
    print('Done')