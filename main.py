#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : 
# @File         : main.py
# @Description  :
import argparse
import os
import tensorflow.compat.v1 as tf
from SASRec import SASRec
from tqdm import tqdm
from util import *


def save_load_args(args):
    if not os.path.isdir(os.path.join('results', args.dataset + '_' + args.save_dir)):
        os.makedirs(os.path.join('results', args.dataset + '_' + args.save_dir))
    os.chdir(os.path.join('results', args.dataset + '_' + args.save_dir))
    with open('%s_args.txt' % args.mode, 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    if args.mode == 'test':
        with open('train_args.txt', 'r') as f:
            for setting in f:
                setting = setting.replace('\n', '')
                argument = setting.split(',')[0]
                value = setting.split(',')[1]
                if value.isdigit():
                    exec('args.%s = %d' % (argument, int(value)))
                elif value.replace('.', '').isdigit():
                    exec('args.%s = %f' % (argument, float(value)))
                elif value == 'True':
                    exec('args.%s = True' % argument)
                elif value == 'False':
                    exec('args.%s = False' % argument)


def train(config, model, args, train_sess, max_item):
    num_batch = int(len(train_sess) / args.batch_size)
    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, args.num_epochs + 1):
            # for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
            #                  desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
            for step in range(num_batch):
                seq, pos, neg = sampler(train_sess, max_item, batch_size=args.batch_size, maxlen=args.maxlen)
                auc, loss, _, merged = sess.run([model.auc, model.loss, model.train_op, model.merged],
                                                {model.input_seq: seq, model.pos: pos, model.neg: neg,
                                                 model.is_training: True})
            writer.add_summary(merged, epoch)
            if epoch % args.display_interval == 0:
                saver.save(sess, 'model/epoch=%d.ckpt' % epoch)


def test(config, model, args, valid_sess, test_sess, max_item):
    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session(config=config) as sess, open('test_' + '_log.txt', 'w') as f:
        for epoch in range(1, args.num_epochs + 1):
            if epoch % args.display_interval == 0:
                saver.restore(sess, 'model/epoch=%d.ckpt' % epoch)
                t_valid = evaluate(valid_sess, max_item, model, args, sess, 'Validating')
                t_test = evaluate(test_sess, max_item, model, args, sess, 'Testing')
                info = 'epoch:%d, valid (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                       ', test (MRR@20: %.4f, RECALL@20:%.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                       % (epoch, t_valid[0], t_valid[1], t_valid[2], t_valid[3],
                          t_test[0], t_test[1], t_test[2], t_test[3])
                print(info)
                f.write(info + '\n')


def early_stop(config, model, args, train_sess, valid_sess, test_sess, max_item, logs):
    num_batch = int(len(train_sess) / args.batch_size)
    saver = tf.train.Saver(max_to_keep=50)
    t_valid_previous = (0, 0, 0, 0)
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, args.num_epochs + 1):
            for _ in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                             desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
            # for step in range(num_batch):
                seq, pos, neg = sampler(train_sess, max_item, batch_size=args.batch_size, maxlen=args.maxlen)
                auc, loss, _, merged = sess.run([model.auc, model.loss, model.train_op, model.merged],
                                                {model.input_seq: seq, model.pos: pos, model.neg: neg,
                                                 model.is_training: True})
            writer.add_summary(merged, epoch)
            if epoch % args.display_interval == 0:
                saver.save(sess, 'model/epoch=%d.ckpt' % epoch)
                t_valid = evaluate(valid_sess, max_item, model, args, sess, 'Validating')
                t_test = evaluate(test_sess, max_item, model, args, sess, 'Testing')
                info = 'epoch:%d, valid (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                       ', test (MRR@20: %.4f, RECALL@20:%.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                       % (epoch, t_valid[0], t_valid[1], t_valid[2], t_valid[3],
                          t_test[0], t_test[1], t_test[2], t_test[3])
                print(info)
                logs.write(info + '\n')
                if all(j < i for i, j in zip(t_valid_previous, t_valid)):
                    break
                else:
                    t_valid_previous = t_valid


if __name__ == '__main__':

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--joint', default=0, type=int)
    parser.add_argument('--remove_item', default=True, type=bool)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--hidden_units', default=150, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--display_interval', default=5, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    parser.add_argument('--test_batch', default=128, type=int)
    parser.add_argument('--neg_sample', default=None, type=int)
    args = parser.parse_args()
    save_load_args(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    if args.joint:
        logs = open('JointLearning_logs.txt', mode='a')
        logs.write('Using %s method.\n' % args.mode)
        test_dataset = os.path.join('..', '..', 'data', args.dataset, 'test_0')
        train_dataset = os.path.join('..', '..', 'data', args.dataset, 'train_0')
        valid_dataset = os.path.join('..', '..', 'data', args.dataset, 'valid_0')

        item_set = set()
        train_sess = load_data(train_dataset, item_set, is_train=True, remove_item=False)
        logs.write('Training set information: total number of action: %d.\n'
                   % sum(list(map(lambda sess: len(sess), train_sess))))
        logs.write('Validating set information: ')
        valid_sess = load_data(valid_dataset, item_set, is_train=False, remove_item=args.remove_item, logs=logs)
        logs.write('Test set information: ')
        test_sess = load_data(test_dataset, item_set, is_train=False, remove_item=args.remove_item, logs=logs)
        max_item = max(list(item_set))

        with tf.device('/gpu:%d' % args.device_num):
            model = SASRec(max_item, args)

        if args.mode == 'train':
            train(config, model, args, train_sess, max_item)
        if args.mode == 'test':
            test(config, model, args, valid_sess, test_sess, max_item)
        if args.mode == 'early_stop':
            early_stop(config, model, args, train_sess, valid_sess, test_sess, max_item, logs=logs)

    else:
        logs = open('ContinueLearning_logs.txt', mode='a')
        if args.dataset == 'DIGINETICA':
            max_item = 122867
        elif args.dataset == 'YOOCHOOSE':
            max_item = None
        with tf.device('/gpu:%d' % args.device_num):
            model = SASRec(max_item, args)

        datafiles = os.listdir(os.path.join('..', '..', 'data', args.dataset))
        periods = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles)))/3 - 1)
        logs.write('Number of periods is %d.\nUsing %s method.\n' % (periods, args.mode))

        for period in range(1, periods + 1):
            print('Period %d:' % period)
            logs.write('Period %d:\n' % period)
            train_dataset = os.path.join('..', '..', 'data', args.dataset, 'train_%d' % period)
            valid_dataset = os.path.join('..', '..', 'data', args.dataset, 'valid_%d' % period)
            test_dataset = os.path.join('..', '..', 'data', args.dataset, 'test_%d' % period)

            item_set = set()
            train_sess = load_data(train_dataset, item_set, is_train=True, remove_item=False)
            if period > 1:
                previous_valid = os.path.join('..', '..', 'data', args.dataset, 'valid_%d' % (period - 1))
                previous_test = os.path.join('..', '..', 'data', args.dataset, 'test_%d' % (period - 1))
                train_sess.extend(load_data(previous_valid, item_set, is_train=True, remove_item=False))
                train_sess.extend(load_data(previous_test, item_set, is_train=True, remove_item=False))
            logs.write('Training set information: total number of action: %d.\n'
                       % sum(list(map(lambda sess: len(sess), train_sess))))
            logs.write('Validating set information: ')
            valid_sess = load_data(valid_dataset, item_set, is_train=False, remove_item=args.remove_item, logs=logs)
            logs.write('Testing set information: ')
            test_sess = load_data(test_dataset, item_set, is_train=False, remove_item=args.remove_item, logs=logs)
            temp_max = max(list(item_set))

            num_batch = int(len(train_sess) / args.batch_size)
            saver = tf.train.Saver(max_to_keep=20)
            t_valid_previous = (0, 0, 0, 0)

            with tf.Session(config=config) as sess:
                writer = tf.summary.FileWriter('logs/period%d' % period, sess.graph)
                if period == 1:
                    sess.run(tf.global_variables_initializer())
                else:
                    saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
                for epoch in range(1, args.num_epochs + 1):
                    for _ in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                                  desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                        # for step in range(num_batch):
                        seq, pos, neg = sampler(train_sess, temp_max, batch_size=args.batch_size, maxlen=args.maxlen)
                        auc, loss, _, merged = sess.run([model.auc, model.loss, model.train_op, model.merged],
                                                        {model.input_seq: seq, model.pos: pos, model.neg: neg,
                                                         model.is_training: True})
                    writer.add_summary(merged, epoch)
                    if epoch % args.display_interval == 0:
                        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
                            os.makedirs(os.path.join('model', 'period%d' % period))
                        saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))
                        t_valid = evaluate(valid_sess, max_item, model, args, sess, 'Validating')
                        t_test = evaluate(test_sess, max_item, model, args, sess, 'Testing')
                        info = 'epoch:%d, valid (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                               ', test (MRR@20: %.4f, RECALL@20:%.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                               % (epoch, t_valid[0], t_valid[1], t_valid[2], t_valid[3],
                                  t_test[0], t_test[1], t_test[2], t_test[3])
                        print(info)
                        logs.write(info + '\n')
                        if all(j < i for i, j in zip(t_valid_previous, t_valid)):
                            best_epoch = epoch - args.display_interval
                            break
                        else:
                            t_valid_previous = t_valid
            logs.write('\n')

    logs.close()
    print('Done')
