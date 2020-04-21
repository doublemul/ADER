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
from SASRec_modify import SASRec
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
    if args.is_joint:
        # if joint learning: periods = [0]
        logs.write('\nJoint Learning\n')
        periods = [0]
    else:
        # if continue learning: periods = [1, 2, ..., period_num]
        datafiles = os.listdir(os.path.join('..', '..', 'data', args.dataset))
        period_num = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles))) / 3 - 1)
        logs.write('\nContinue Learning: Number of periods is %d.\n' % period_num)
        periods = range(1, period_num + 1)
    for period in periods:
        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
            os.makedirs(os.path.join('model', 'period%d' % period))
    return periods


def load_exemplars(period, use_exemplar, mode):
    """
    This method load exemplar in previous period
    :param period: this period
    :return: exemplar list
    """
    if period > 1 and use_exemplar:
        exemplars = []
        with open('exemplar/%sExemplarPeriod=%d.pickle' % (mode, period - 1), mode='rb') as file:
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
    # parser.add_argument('--dataset', required=True)
    # parser.add_argument('--save_dir', required=True)
    # parser.add_argument('--desc', required=True)
    parser.add_argument('--dataset', default='DIGINETICA', type=str)
    parser.add_argument('--save_dir', default='try', type=str)  # ContinueLearning
    parser.add_argument('--desc', default='non_example', type=str)
    # continue learning parameter
    parser.add_argument('--use_exemplar', default=True, type=str2bool)
    parser.add_argument('--exemplar_size', default=5000, type=int)
    parser.add_argument('--select_mode', default=1, type=int)
    parser.add_argument('--is_joint', default=False, type=str2bool)
    parser.add_argument('--remove_item', default=True, type=str2bool)
    # early stop parameter
    parser.add_argument('--stop_iter', default=20, type=int)
    parser.add_argument('--display_interval', default=1, type=int)
    # batch size and device
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
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
    logs = open('Training_logs.txt', mode='a')
    logs.write('Data set: %s Description: %s\nargs:' % (args.dataset, args.desc))
    logs.write(' '.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

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
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # build model
    item_num = 43023 if args.dataset == 'DIGINETICA' or args.dataset == 'DIGINETICA_week' else 29086
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    # Loop each period for continue learning #
    periods = get_periods(args, logs)
    dataloader = DataLoader(args, logs)
    plotter = ContinueLearningPlot(args)
    best_epoch = 0
    t_start = time.time()
    for period in periods:
        t = time.time()
        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)

        # Load exemplar
        train_exemplar_data = load_exemplars(period, args.use_exemplar, 'train')
        valid_exemplar_data = load_exemplars(period, args.use_exemplar, 'valid')

        # Load data
        train_sess = dataloader.train_loader(period)
        train_item_count = dataloader.get_item_counter(train_exemplar_data)
        valid_sess = dataloader.evaluate_loader(period, 'valid')
        valid_item_count = dataloader.get_item_counter(valid_exemplar_data)
        test_sess = dataloader.evaluate_loader(period, 'test')
        max_item = dataloader.max_item()

        # Start of the main algorithm
        Recall20, stop_counter = 0, 0
        with tf.Session(config=config) as sess:

            writer = tf.summary.FileWriter('logs/period%d' % period, sess.graph)
            saver = tf.train.Saver(max_to_keep=1)

            # initialize variables or reload from previous period
            if period <= 1:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
                valid_evaluator = Evaluator(args, valid_sess, max_item, 'valid', model, sess, logs)
                valid_evaluator.evaluate(0, valid_exemplar_data)
                t_valid = valid_evaluator.results()
                plotter.add_valid(period, epoch=0, t_valid=t_valid)
                del valid_evaluator

            # train
            for epoch in range(1, args.num_epochs + 1):
                train_sampler = Sampler(args, train_sess, args.batch_size, True, max_item)
                train_sampler.prepare_data_full_exemplar_seq_for_train_and_eval(train_exemplar_data)
                batch_num = train_sampler.batch_num
                # train each epoch
                for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                              desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                    seq, pos, neg = train_sampler.sampler()
                    sess.run(model.train_op, {model.input_seq: seq,
                                              model.pos: pos,
                                              model.neg: neg,
                                              model.is_training: True})
                del train_sampler

                # evaluate performance and early stop
                if epoch % args.display_interval == 0:
                    # validate performance
                    valid_evaluator = Evaluator(args, valid_sess, max_item, 'valid', model, sess, logs)
                    valid_evaluator.evaluate(epoch, valid_exemplar_data)
                    t_valid = valid_evaluator.results()
                    plotter.add_valid(period, epoch, t_valid)
                    del valid_evaluator

                    # early stop
                    recall_20 = t_valid[1]
                    if Recall20 > recall_20:
                        stop_counter += 1
                        if stop_counter >= args.stop_iter:
                            best_epoch = epoch - args.stop_iter * args.display_interval
                            break
                    else:
                        stop_counter = 0
                        Recall20 = recall_20
                        best_epoch = epoch
                        saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))

            # record best valid performance
            print('Best valid Recall@20 = %f, at epoch %d' % (Recall20, best_epoch))
            logs.write('Best valid Recall@20 = %f, at epoch %d\n' % (Recall20, best_epoch))
            plotter.best_epoch(period, best_epoch)

            # test performance
            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))

            test_evaluator = Evaluator(args, test_sess, max_item, 'test', model, sess, logs)
            test_evaluator.evaluate(best_epoch)
            plotter.add_test(test_evaluator.results())
            del test_evaluator

            # save exemplars
            if args.use_exemplar and not args.is_joint:
                for data, exemplar_data, item_count, mode, size in \
                        [[train_sess, train_exemplar_data, train_item_count, 'train', args.exemplar_size],
                         [valid_sess, valid_exemplar_data, valid_item_count, 'valid',
                          int(1 / 30 * args.exemplar_size)]]:
                    exemplar = ExemplarGenerator(args, size, item_num, data, mode, logs)
                    exemplar.prepare_data(exemplar_data)
                    if args.select_mode == 0:
                        exemplar.randomly_by_frequency(item_count)
                    elif args.select_mode == 1:
                        exemplar.herding_by_frequency(item_count, sess, model)
                    elif args.select_mode == 2:
                        exemplar.randomly_by_period()
                    elif args.select_mode == 3:
                        exemplar.herding_by_period(sess, model)
                    else:
                        raise ValueError('Not defined select mode.')
                    exemplar.save(period)
                    del exemplar

        print('Period %d time: %.2f minutes.' % (period, (time.time() - t) / 60.0))
        logs.write('Period %d time: %.2f minutes\n' % (period, (time.time() - t) / 60.0))

    plotter.plot()
    logs.write('Total time: %.2f minutes\n' % ((time.time() - t_start) / 60.0))
    print('Total time: %.2f minutes.' % ((time.time() - t_start) / 60.0))
    logs.write('Done\n\n')
    logs.close()
    print('Done')
