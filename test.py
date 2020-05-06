#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : 
# @File         : test.py
# @Description  :
import argparse
import tensorflow.compat.v1 as tf
from SASRec import SASRec
import gc
from util import *
import matplotlib.pyplot as plt
import pickle


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
    return periods


if __name__ == '__main__':

    gc.enable()
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIGINETICA', type=str)
    parser.add_argument('--save_dir', default='try', type=str)
    parser.add_argument('--remove_item', default=True, type=str2bool)
    parser.add_argument('--is_joint', default=False, type=str2bool)
    # batch size and device
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
    parser.add_argument('--random_seed', default=555, type=int)
    args = parser.parse_args()

    # set configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # build model
    item_num = 43136 if args.dataset == 'DIGINETICA' else 27333
    with tf.device('/gpu:%d' % args.device_num):
        model = SASRec(item_num, args)

    main_dir = os.getcwd()
    overall_logs = open('results_mean.txt', mode='w')
    next_session_recall20_plt = dict()
    next_session_recall10_plt = dict()
    next_session_mrr20_plt = dict()
    next_session_mrr10_plt = dict()
    overall_recall20_plt = dict()
    overall_recall10_plt = dict()
    overall_mrr20_plt = dict()
    overall_mrr10_plt = dict()

    for args.save_dir in ['NoExemplar', 'Herding-5k', 'Herding-10k', 'Random-5k', 'Random-10k', 'Loss-5k', 'Loss-10k', 'Upper-bound']:

        # set path
        if not os.path.isdir(os.path.join('results', args.dataset + '_' + args.save_dir)):
            raise ValueError('Wrong save_dir !')
        os.chdir(os.path.join('results', args.dataset + '_' + args.save_dir))
        # record logs

        if os.path.isfile('results.pickle'):
            with open('results.pickle', mode='rb') as file:
                results = pickle.load(file)
                next_session_recall20 = results[0]
                next_session_recall10 = results[1]
                next_session_mrr20 = results[2]
                next_session_mrr10 = results[3]
                overall_recall20 = results[4]
                overall_recall10 = results[5]
                overall_mrr20 = results[6]
                overall_mrr10 = results[7]
        else:
            logs = open('test.txt', mode='w')
            # Loop each period for continue learning #
            periods = get_periods(args, logs)
            data_loader = DataLoader(args, logs)
            train_sess = dict()
            test_sess = dict()
            next_session_recall20 = []
            next_session_recall10 = []
            next_session_mrr20 = []
            next_session_mrr10 = []
            overall_recall20 = []
            overall_recall10 = []
            overall_mrr20 = []
            overall_mrr10 = []

            for period in periods:
                print('Period %d:' % period)
                logs.write('Period %d:\n' % period)

                # Load data
                train_sess[period] = data_loader.train_loader(period)
                if period < periods[-1]:
                    next_sess = data_loader.evaluate_loader(period)
                max_item = data_loader.max_item()

                # Start of the main algorithm
                with tf.Session(config=config) as sess:

                    saver = tf.train.Saver()
                    epoch = 200
                    while not os.path.isfile('model/period%d/epoch=%d.ckpt.index' % (period, epoch)):
                        epoch -= 1
                        if epoch == 0:
                            raise ValueError('Wrong model direction or no model')
                    saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))

                    # test performance
                    if period < periods[-1]:
                        print('Next period %d performance:')
                        test_evaluator = Evaluator(args, next_sess, max_item, 'test', model, sess, logs)
                        test_evaluator.evaluate(epoch)
                        results = test_evaluator.results()
                        next_session_mrr20.append(results[0])
                        next_session_recall20.append(results[1])
                        next_session_mrr10.append(results[2])
                        next_session_recall10.append(results[3])

                    else:
                        print('Overall performance:')
                        for p in sorted(train_sess.keys()):
                            overall_evaluator = Evaluator(args, train_sess[p], max_item, 'test', model, sess, logs)
                            overall_evaluator.evaluate(epoch)
                            results = overall_evaluator.results()
                            del overall_evaluator
                            overall_mrr20.append(results[0])
                            overall_recall20.append(results[1])
                            overall_mrr10.append(results[2])
                            overall_recall10.append(results[3])

            with open('results.pickle', mode='wb') as file:
                results = pickle.dump([next_session_recall20,
                                       next_session_recall10,
                                       next_session_mrr20,
                                       next_session_mrr10,
                                       overall_recall20,
                                       overall_recall10,
                                       overall_mrr20,
                                       overall_mrr10], file)
            logs.write('Done\n')
            logs.close()

        overall_logs.write('%s:\n' % args.save_dir)
        next_session_recall20_plt[args.save_dir] = np.array(next_session_recall20) * 100
        overall_logs.write('mean next session recall@20: %.2f%%.\n' % (np.array(next_session_recall20) * 100).mean())
        next_session_recall10_plt[args.save_dir] = np.array(next_session_recall10) * 100
        overall_logs.write('mean next session recall@10: %.2f%%.\n' % (np.array(next_session_recall10) * 100).mean())
        next_session_mrr20_plt[args.save_dir] = np.array(next_session_mrr20) * 100
        overall_logs.write('mean next session mrr@20: %.2f%%.\n' % (np.array(next_session_mrr20) * 100).mean())
        next_session_mrr10_plt[args.save_dir] = np.array(next_session_mrr10) * 100
        overall_logs.write('mean next session mrr@10: %.2f%%.\n' % (np.array(next_session_mrr10) * 100).mean())

        overall_recall20_plt[args.save_dir] = np.array(overall_recall20) * 100
        overall_logs.write('mean overall recall@20: %.2f%%.\n' % (np.array(overall_recall20) * 100).mean())
        overall_recall10_plt[args.save_dir] = np.array(overall_recall10) * 100
        overall_logs.write('mean overall recall@10: %.2f%%.\n' % (np.array(overall_recall10) * 100).mean())
        overall_mrr20_plt[args.save_dir] = np.array(overall_mrr20) * 100
        overall_logs.write('mean overall mrr@20: %.2f%%.\n' % (np.array(overall_mrr20) * 100).mean())
        overall_mrr10_plt[args.save_dir] = np.array(overall_mrr10) * 100
        overall_logs.write('mean overall mrr@10: %.2f%%.\n' % (np.array(overall_mrr10) * 100).mean())

        os.chdir(main_dir)

    width = 0.1
    num_bar = len(next_session_mrr10_plt.keys()) / 2
    bars = []
    lines = []
    fig, axs = plt.subplots(4, 2, sharex='col', figsize=[6.4 * 1.5 * 1.5, 4.8 * 1.5])
    axs[0, 0].set_title('next period performance')
    axs[0, 1].set_title('overall performance on final model')
    for s, save_dir in enumerate(next_session_mrr10_plt.keys()):

        next_session_recall20 = next_session_recall20_plt[save_dir]
        overall_recall20 = overall_recall20_plt[save_dir]

        next_session_recall10 = next_session_recall10_plt[save_dir]
        overall_recall10 = overall_recall10_plt[save_dir]

        next_session_mrr20 = next_session_mrr20_plt[save_dir]
        overall_mrr20 = overall_mrr20_plt[save_dir]

        next_session_mrr10 = next_session_mrr10_plt[save_dir]
        overall_mrr10 = overall_mrr10_plt[save_dir]

        for i, data in enumerate([[next_session_recall20, overall_recall20],
                                  [next_session_recall10, overall_recall10],
                                  [next_session_mrr20, overall_mrr20],
                                  [next_session_mrr10, overall_mrr10]]):

            line, = axs[i, 0].plot(np.arange(1, len(data[0]) + 1), data[0], label=save_dir)
            lines.append(line)
            bar = axs[i, 1].bar(np.arange(1, len(data[1]) + 1) + width * (s - num_bar + 0.5), data[1],
                                width=width, label=save_dir)
            if i == 0:
                bars.append(bar)

    axs[0, 0].set_ylabel('Recall@20(%)')
    axs[1, 0].set_ylabel('Recall@10(%)')
    axs[2, 0].set_ylabel('MRR@20(%)')
    axs[3, 0].set_ylabel('MRR@10(%)')
    axs[0, 0].set_xticks(np.arange(1, len(data[0]) + 1))
    axs[0, 1].set_xticks(np.arange(1, len(data[1]) + 1))
    fig.tight_layout()
    fig.legend(handles=bars, loc='upper center', bbox_to_anchor=(0.5, 0.04), fancybox=False, ncol=10)
    plt.savefig('results.pdf', bbox_inches='tight')
    plt.show()
    overall_logs.close()
    print('Done')
