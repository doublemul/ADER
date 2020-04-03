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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def exemplar_generator(args, model, sess, exemplar, period, data_sess, item_list, info):
    sessions_by_item = defaultdict(list)
    for _ in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                  desc='%s exemplar generating 1/2' % info):
        seq, pos, _ = sampler(data_sess, item_list, batch_size=args.batch_size, maxlen=args.maxlen)
        pos = np.array(pos)
        pos = pos[:, -1]
        for session, item in zip(seq, pos):
            sessions_by_item[item].append(session)

    for item in tqdm(sessions_by_item.keys(), total=num_batch, ncols=70, leave=False, unit='b',
                     desc='%s exemplar generating 2/2' % info):
        seq = sessions_by_item[item]
        rep = sess.run(model.rep, {model.input_seq: seq, model.is_training: False})
        exemplar.add(rep=rep, item=item, seq=seq)
    exemplar.save(period, info)


def full_exemplar_generator(args, model, sess, exemplar, period, data_sess, item_list, info):
    sessions_by_item = defaultdict(list)
    for _ in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                  desc='%s exemplar generating 1/2' % info):
        seq, pos, _ = sampler(data_sess, item_list, batch_size=args.batch_size, maxlen=args.maxlen)
        rep_full = sess.run(model.rep_full, {model.input_seq: seq, model.is_training: False})
        for s, p, r in zip(seq, pos, rep_full):
            for i in range(1, len(p)+1):
                item = p[-i]
                session = s[:1-i]
                rep = r[-i]
                if item not in sessions_by_item:
                    sessions_by_item[item] = [[], []]
                sessions_by_item[item][0].append(session)
                sessions_by_item[item][1].append(rep)
    for item in tqdm(sessions_by_item.keys(), total=num_batch, ncols=70, leave=False, unit='b',
                     desc='%s exemplar generating 2/2' % info):
        seq, rep = sessions_by_item[item]
        exemplar.add(rep=rep, item=item, seq=seq)
    exemplar.save(period, info)


if __name__ == '__main__':

    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--desc', required=True)

    parser.add_argument('--remove_item', default=True, type=str2bool)
    parser.add_argument('--is_joint', default=True, type=str2bool)
    parser.add_argument('--is_herding', default=True, type=str2bool)

    parser.add_argument('--mode', default='early_stop', type=str)
    parser.add_argument('--stop_iter', default=20, type=int)
    parser.add_argument('--display_interval', default=1, type=int)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch', default=64, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    # hyper-parameters grid search
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    # hyper-parameter fixed
    parser.add_argument('--hidden_units', default=120, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    args = parser.parse_args()

    # set path
    if not os.path.isdir(os.path.join('results', args.dataset + '_' + args.save_dir)):
        os.makedirs(os.path.join('results', args.dataset + '_' + args.save_dir))
    os.chdir(os.path.join('results', args.dataset + '_' + args.save_dir))
    save_load_args(args)

    # set configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # record logs
    logs = open('Training_logs.txt', mode='a')
    logs.write('Data set: %s Description: %s\nargs:' % (args.dataset, args.desc))
    logs.write(' '.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # build model
    item_num = 0
    if args.dataset == 'DIGINETICA':
        item_num = 43023  # 122867
    elif args.dataset == 'YOOCHOOSE':
        item_num = 29086  # 52597
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    # periods for joint learning or continue learning situation
    if args.is_joint:
        # if joint learning: periods = [0]
        logs.write('\nJoint Learning\nUsing %s mode.\n' % args.mode)
        periods = [0]
    else:
        # if continue learning: periods = [1, 2, ..., period_nmu]
        datafiles = os.listdir(os.path.join('..', '..', 'data', args.dataset))
        period_num = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles))) / 3 - 1)
        logs.write('\nContinue Learning:Number of periods is %d.\nUsing %s mode.\n' % (period_num, args.mode))
        periods = range(1, period_num + 1)

    item_set = set()
    # Loop each period for continue learning #
    for period in periods:
        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)

        # Load data #
        train_dataset = os.path.join('..', '..', 'data', args.dataset, 'train_%d' % period)
        valid_dataset = os.path.join('..', '..', 'data', args.dataset, 'valid_%d' % period)
        test_dataset = os.path.join('..', '..', 'data', args.dataset, 'test_%d' % period)
        # load train data
        train_sess = load_data(train_dataset, item_set, is_train=True, remove_item=False)
        if period > 1:
            previous_valid = os.path.join('..', '..', 'data', args.dataset, 'valid_%d' % (period - 1))
            previous_test = os.path.join('..', '..', 'data', args.dataset, 'test_%d' % (period - 1))
            train_sess.extend(load_data(previous_valid, item_set, is_train=True, remove_item=False))
            train_sess.extend(load_data(previous_test, item_set, is_train=True, remove_item=False))
        train_info = 'Train set information: total number of action: %d.' \
                     % sum(list(map(lambda session: len(session), train_sess)))
        logs.write(train_info + '\n')
        print(train_info)
        # load validation data
        valid_sess = load_data(valid_dataset, item_set,
                               is_train=False, remove_item=args.remove_item, logs=logs, info='Validation')
        # load test data
        test_sess = load_data(test_dataset, item_set,
                              is_train=False, remove_item=args.remove_item, logs=logs, info='Test')
        # generate item list
        item_list = list(item_set)

        # Start of the main algorithm #
        exemplar = ExemplarSet(args, item_list, m=5)
        num_batch = int(len(train_sess) / args.batch_size)
        saver = tf.train.Saver(max_to_keep=20)
        Recall20 = 0
        stopcounter = 0
        best_epoch = 0
        with tf.Session(config=config) as sess:
            writer = tf.summary.FileWriter('logs/period%d' % period, sess.graph)
            # initialize variables for the first period or restore previous model for later periods
            if period <= 1:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
            # saver.restore(sess, 'model/period0/epoch=77.ckpt')
            # train each epoch
            for epoch in range(1, args.num_epochs + 1):
                # train model
                if args.mode == 'train' or args.mode == 'early_stop':
                    for _ in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b',
                                  desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                        seq, pos, neg = sampler(train_sess, item_list, batch_size=args.batch_size, maxlen=args.maxlen)
                        auc, loss, _, merged, rep = sess.run(
                            [model.auc, model.loss, model.train_op, model.merged, model.rep],
                            {model.input_seq: seq, model.pos: pos, model.neg: neg,
                             model.is_training: True})
                    writer.add_summary(merged, epoch)
                # evaluate performance or save model every display_interval epoch
                if epoch % args.display_interval == 0:
                    # save model
                    if args.mode == 'train':
                        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
                            os.makedirs(os.path.join('model', 'period%d' % period))
                        saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))
                    # validate performance
                    if args.mode == 'test' or args.mode == 'early_stop':
                        t_valid = evaluate_all(valid_sess, item_list, model, args, sess,
                                               'Validating epoch %d/%d' % (epoch, args.num_epochs))
                        info = 'epoch:%d, valid (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                               % (epoch, t_valid[0], t_valid[1], t_valid[2], t_valid[3])
                        print(info)
                        logs.write(info + '\n')
                    # early stop
                    if args.mode == 'early_stop':
                        if Recall20 > t_valid[1]:
                            stopcounter += 1
                            if stopcounter >= args.stop_iter:
                                best_epoch = epoch - args.stop_iter * args.display_interval
                                break
                        else:
                            stopcounter = 0
                            Recall20 = t_valid[1]
                            best_epoch = epoch
                            if not os.path.isdir(os.path.join('model', 'period%d' % period)):
                                os.makedirs(os.path.join('model', 'period%d' % period))
                            saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))
            print('Best valid Recall@20 = %f, at epoch %d' % (Recall20, best_epoch))
            logs.write('Best valid Recall@20 = %f, at epoch %d\n' % (Recall20, best_epoch))

            # test performance
            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))
            t_test = evaluate_all(test_sess, item_list, model, args, sess, 'Testing epoch%d' % best_epoch)
            info = 'epoch:%d, test (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
                   % (best_epoch, t_test[0], t_test[1], t_test[2], t_test[3])
            print(info)
            logs.write(info + '\n')

            # if period > 0:
            #     exemplar_generator(args, model, sess, exemplar, period, train_sess, item_list, 'Train')
            #     # full_exemplar_generator(args, model, sess, exemplar, period, valid_sess, item_list, 'Valid')

    logs.write('Done\n\n')
    logs.close()
    print('Done')



