#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : main.py
# @Description  : main file for training ADER
# @Author       : Xiaoyu Lin, Fei Mi, Boi Faltings
import argparse
import os
import math
import tensorflow.compat.v1 as tf
from ADER import ADER
from tqdm import tqdm
from util import *
import gc
import time
from tfdeterminism import patch


def str2bool(v):
    """
    Convert string to boolean
    :param v: string
    :return: boolean True or False
    """
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
    :return: [0] for joint learning,
             [1, 2, ..., period_num] for continue learning
    """
    # if continue learning: periods = [1, 2, ..., period_num]
    datafiles = os.listdir(os.path.join('..', '..', 'data', args.dataset))
    period_num = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles))))
    logs.write('\nContinue Learning: Number of periods is %d.\n' % period_num)
    periods = range(1, period_num)
    #  create dictionary to save model
    for period in periods:
        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
            os.makedirs(os.path.join('model', 'period%d' % period))
    return periods


def load_exemplars(mode, fast_exemplar=None):
    """
    This method load exemplar in previous period
    :param mode: 'train' or 'valid'
    :param fast_exemplar: read exemplar directly from previous exemplar variable if None read from .pickle file
    :return: exemplar list
    """
    exemplars = []
    if fast_exemplar is None:
        with open('%s_exemplar.pickle' % mode, mode='rb') as file:
            exemplars_item = pickle.load(file)
    else:
        exemplars_item = fast_exemplar

    for item in exemplars_item.values():
        if isinstance(item, list):
            exemplars.extend([i for i in item if i])
    return exemplars


def split_data(data, choice_num):
    """
    Split the data into two parts and the number of data in second part is given by choice_num
    :param data: original data
    :param choice_num: the number of data in second part
    :return: two split data parts
    """
    data_size = len(data)
    sidx = np.arange(data_size, dtype='int32')
    np.random.shuffle(sidx)
    first_part = [data[s] for s in sidx[choice_num:]]
    second_part = [data[s] for s in sidx[:choice_num]]
    return first_part, second_part


if __name__ == '__main__':

    gc.enable()
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIGINETICA', type=str)  # name of dataset 'DIGINETICA' or 'YOOCHOOSE'
    parser.add_argument('--save_dir', default='ADER', type=str)  # name of dictionary save the results
    # exemplar
    parser.add_argument('--exemplar_size', default=30000, type=int)  # size of exemplars
    parser.add_argument('--lambda_', default=0.8, type=float)  # base adaptive weight
    # early stop parameter
    parser.add_argument('--stop', default=5, type=int)  # number of epoch for early stop
    # batch size and device
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--test_batch', default=64, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    # hyper-parameters grid search
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    # hyper-parameter fixed
    parser.add_argument('--random_seed', default=555, type=int)
    parser.add_argument('--hidden_units', default=150, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    args = parser.parse_args()

    # Set path
    if not os.path.isdir(os.path.join('results', args.dataset + '_' + args.save_dir)):
        os.makedirs(os.path.join('results', args.dataset + '_' + args.save_dir))
    os.chdir(os.path.join('results', args.dataset + '_' + args.save_dir))
    # Record logs
    logs = open('Training_logs.txt', mode='w')
    logs.write(' '.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # For reproducibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    patch()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # Set configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Build model
    if args.dataset == 'DIGINETICA':
        item_num = 43136
    elif args.dataset == 'YOOCHOOSE':
        item_num = 25958
    else:
        raise ValueError('Invalid dataset name')
    with tf.device('/gpu:%d' % args.device_num):
        model = ADER(item_num, args)

    # Loop each period for continue learning
    periods = get_periods(args, logs)
    dataloader = DataLoader(args, item_num, logs)
    best_epoch, item_num_prev = 0, 0
    t_start = time.time()

    for period in periods:

        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)
        lr = args.lr
        # record performance for early stop
        best_performance, performance = 0, 0

        # Prepare data
        # load train data
        train_sess, train_item_set = dataloader.train_loader(period - 1)
        train_sampler = Sampler(args, train_sess, args.batch_size)
        train_sampler.prepare_data()
        valid_subseq, train_subseq = train_sampler.split_data(valid_portion=0.1, return_train=True)
        # test data
        test_sess, test_size, test_item_set = dataloader.evaluate_loader(period)
        max_item = dataloader.max_item()
        # exemplar
        if period > 1:
            train_exemplar_data_logits = load_exemplars('train', fast_exemplar)
            train_exemplar_size = len(train_exemplar_data_logits)
            train_exemplar_subseq = np.array(train_exemplar_data_logits)[:, 0].tolist()
        else:
            train_exemplar_subseq = None

        # Set loss
        if period > 1:
            # find lambda for current cycle
            new_item = max_item - item_num_prev
            train_size = train_sampler.data_size()
            lambda_ = args.lambda_ * math.sqrt((item_num_prev / max_item) * (train_exemplar_size / train_size))
            print('lambda=%f' % lambda_)
            logs.write('lambda=%.6f\n' % lambda_)
            model.update_exemplar_loss(lambda_=lambda_)
            # prepare exemplar sampler
            train_sampler.shuffle_data()
            batch_num = train_sampler.batch_num
            exemplar_batch = int(train_exemplar_size / batch_num)
            exemplar_samplar = Sampler(args, [], exemplar_batch)
            exemplar_samplar.prepare_data()
            exemplar_samplar.add_full_exemplar(train_exemplar_data_logits)
            exemplar_samplar.shuffle_data()
        else:
            model.set_vanilla_loss()
            train_sampler.shuffle_data()
            batch_num = train_sampler.batch_num

        # Start of the main algorithm
        with tf.Session(config=config) as sess:

            # initialize variables or reload from previous period
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch)) \
                if period > 1 else sess.run(tf.global_variables_initializer())

            # train
            for epoch in range(1, args.num_epochs + 1):
                # train each epoch
                for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                              desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                    # load train batch
                    seq, pos = train_sampler.sampler()

                    if period > 1:
                        ex_seq, ex_pos, logits = exemplar_samplar.exemplar_sampler()
                        seq = seq + ex_seq
                        # exemplar using logistic-matching label
                        sess.run(model.train_op, {model.input_seq: seq,
                                                  model.pos: pos,
                                                  model.is_training: True,
                                                  model.max_item: max_item,
                                                  model.exemplar_logits: logits,
                                                  model.dropout_rate: args.dropout_rate,
                                                  model.lr: lr})
                    else:
                        # without using exemplar for initial cycle
                        sess.run(model.train_op, {model.input_seq: seq,
                                                  model.pos: pos,
                                                  model.is_training: True,
                                                  model.max_item: max_item,
                                                  model.dropout_rate: args.dropout_rate,
                                                  model.lr: lr})

                # validate performance
                valid_evaluator = Evaluator(args, [], max_item, 'valid', model, sess, logs)
                valid_evaluator.evaluate(epoch, valid_subseq)
                performance = valid_evaluator.results()[1]

                # early stop
                best_epoch = epoch
                if best_performance >= performance:
                    stop_counter += 1
                    if stop_counter >= args.stop:
                        best_epoch = epoch - args.stop
                        break
                else:
                    stop_counter = 0
                    best_performance = performance
                    saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))

            # save current meta-data for next cycle
            item_num_prev = max_item
            prev_item_set = train_item_set
            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))

            # test performance
            test_evaluator = Evaluator(args, test_sess, max_item, 'test', model, sess, logs)
            test_evaluator.evaluate(best_epoch)

            # save exemplars
            train_exemplar = ExemplarGenerator(args, args.exemplar_size, train_sess, max_item, logs)
            train_exemplar.add_exemplar(exemplar=train_exemplar_subseq)
            train_exemplar.herding_selection(sess, model)
            fast_exemplar = train_exemplar.exemplars
            # train_exemplar.save('train')
            del train_exemplar

    print('Total time: %.2f minutes.' % ((time.time() - t_start) / 60.0))
    logs.write('Total time: %.2f minutes\nDone' % ((time.time() - t_start) / 60.0))
    logs.close()
    print('Done')
