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
from util import *
import gc


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


def load_exemplars(period, mode):
    """
    This method load exemplar in previous period
    :param period: this period
    :return: exemplar list
    """
    exemplars = []
    with open('exemplar/%sExemplarPeriod=%d.pickle' % (mode, period - 1), mode='rb') as file:
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
    # parser.add_argument('--dataset', required=True)
    # parser.add_argument('--save_dir', required=True)
    # parser.add_argument('--desc', required=True)
    parser.add_argument('--dataset', default='DIGINETICA', type=str)
    parser.add_argument('--save_dir', default='try', type=str) #ContinueLearning
    parser.add_argument('--desc', default='non_example', type=str)
    parser.add_argument('--use_exemplar', default=True, type=str2bool)
    parser.add_argument('--is_joint', default=False, type=str2bool)
    parser.add_argument('--remove_item', default=True, type=str2bool)

    # early stop parameter
    parser.add_argument('--stop_iter', default=20, type=int)
    parser.add_argument('--display_interval', default=1, type=int)
    # batch size and device
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    # hyper-parameters grid search
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    # hyper-parameter fixed
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

    # set configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # build model
    item_num = 43023 if args.dataset == 'DIGINETICA' else 29086
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    # periods for joint learning or continue learning situation
    periods = get_periods(args, logs)

    # Loop each period for continue learning #
    dataloader = DataLoader(args, logs)
    plotter = ContinueLearningPlot(args)
    best_epoch = 0
    for period in periods:
        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)

        # Load data
        train_sess = dataloader.train_loader(period)
        train_item_counter = dataloader.get_item_counter(load_exemplars(period, 'train')) \
            if args.use_exemplar and period > 1 else dataloader.get_item_counter()
        # print('original total train : %d' % np.array(train_item_counter).sum())
        valid_sess = dataloader.evaluate_loader(period, 'valid')
        valid_item_counter = dataloader.get_item_counter(load_exemplars(period, 'valid')) \
            if args.use_exemplar and period > 1 else dataloader.get_item_counter()
        # print('original total valid : %d' % np.array(valid_item_counter).sum())
        test_sess = dataloader.evaluate_loader(period, 'test')
        max_item = dataloader.max_item()

        # Start of the main algorithm
        num_batch = math.ceil(len(train_sess) / args.batch_size)
        Recall20, stop_counter = 0, 0
        with tf.Session(config=config) as sess:

            writer = tf.summary.FileWriter('logs/period%d' % period, sess.graph)
            saver = tf.train.Saver(max_to_keep=1)

            # initialize variables or reload from previous period
            if period <= 1:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
                valid_evaluator = Evaluator(args, valid_sess, max_item, model, 'valid', sess, logs)
                valid_evaluator.valid(0, load_exemplars(period, 'valid')) \
                    if args.use_exemplar else valid_evaluator.test(0)
                t_valid = valid_evaluator.results()
                plotter.add_valid(period, epoch=0, t_valid=t_valid)
                del valid_evaluator

            # train
            for epoch in range(1, args.num_epochs + 1):
                # train each epoch
                train_sampler = Sampler(args=args, data=train_sess, max_item=max_item, is_train=True)
                for i in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                              desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                    seq, pos, neg = train_sampler.hybrid_sampler(load_exemplars(period, 'train')) \
                        if args.use_exemplar and period > 1 else train_sampler.narm_sampler()
                    auc, loss, _, merged = sess.run([model.auc, model.loss, model.train_op, model.merged],
                                                    {model.input_seq: seq,
                                                     model.pos: pos,
                                                     model.neg: neg,
                                                     model.is_training: True})
                writer.add_summary(merged, epoch)
                del train_sampler

                # evaluate performance and early stop
                if epoch % args.display_interval == 0:
                    # validate performance
                    valid_evaluator = Evaluator(args, valid_sess, max_item, model, 'valid', sess, logs)
                    valid_evaluator.valid(epoch, load_exemplars(period, 'valid')) \
                        if args.use_exemplar and period > 1 else valid_evaluator.test(epoch)
                    t_valid = valid_evaluator.results()
                    plotter.add_valid(period, epoch, t_valid)

                    # early stop
                    recall_20 = valid_evaluator.recall_20
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
                    del valid_evaluator

            # record best valid performance
            print('Best valid Recall@20 = %f, at epoch %d' % (Recall20, best_epoch))
            logs.write('Best valid Recall@20 = %f, at epoch %d\n' % (Recall20, best_epoch))
            plotter.best_epoch(period, best_epoch)

            # test performance
            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))
            test_evaluator = Evaluator(args, test_sess, max_item, model, 'test', sess, logs)
            test_evaluator.test(best_epoch)
            t_test = test_evaluator.results()
            plotter.add_test(t_test)
            del test_evaluator

            # save exemplars
            if args.use_exemplar:
                train_exemplar = ExemplarGenerator(args, int(1.5 * max_item), item_num, train_sess, 'train')
                train_exemplar.sort_by_item(load_exemplars(period, 'train')) if period > 1 \
                    else train_exemplar.sort_by_item()
                train_exemplar.randomly_by_frequency(train_item_counter)
                train_exemplar.save(period)
                del train_exemplar

                valid_exemplar = ExemplarGenerator(args, int(0.1 * max_item), item_num, valid_sess, 'valid')
                valid_exemplar.sort_by_item(load_exemplars(period, 'valid')) if period > 1 \
                    else valid_exemplar.sort_by_item()
                valid_exemplar.randomly_by_frequency(valid_item_counter)
                valid_exemplar.save(period)
                del valid_exemplar

    plotter.plot()
    logs.write('Done\n\n')
    logs.close()
    print('Done')



