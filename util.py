#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : util.py
# @Description  :
import random
import os
import pickle
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


def load_data(dataset_name, item_set, is_train=True, remove_item=True, logs=None, info=None):
    # Load the dataset
    Sessions = defaultdict(list)
    removed_num = 0
    total_num = 0
    with open('%s.txt' % dataset_name, 'r') as f:
        for line in f:
            total_num += 1
            sessId, itemId = line.rstrip().split(' ')
            sessId = int(sessId)
            itemId = int(itemId)

            # remove new items in test that not appear in train
            if is_train or (not remove_item):
                item_set.add(itemId)
            else:
                if itemId not in item_set:
                    removed_num += 1
                    continue
            Sessions[sessId].append(itemId)

    if not is_train and remove_item:
        delete_keys = []
        for sessId in Sessions:
            if len(Sessions[sessId]) == 1:
                removed_num += 1
                delete_keys.append(sessId)
        for delete_key in delete_keys:
            del Sessions[delete_key]

    if not is_train:
        info = '%s set information: original total number of action: %d, removed number of action: %d.' \
               % (info, total_num, removed_num)
        logs.write(info + '\n')
        print(info)

    sessions = list(Sessions.values())
    del Sessions
    return sessions


def valid_portion(sessions, portion=0.1):
    sess_num = len(sessions)
    indices = np.arange(sess_num)
    np.random.shuffle(indices)
    train_num = int(np.round(sess_num * (1 - portion)))
    train_sess = [sessions[i] for i in indices[:train_num]]
    valid_sess = [sessions[i] for i in indices[train_num:]]
    return train_sess, valid_sess


def random_neg(item_list, ts):
    neg = random.choice(item_list)
    while neg in ts:
        neg = random.choice(item_list)
    return neg


def sample(user_train, item_list, maxlen):
    session = random.choice(user_train)
    while len(session) <= 1:
        session = random.choice(user_train)

    seq = np.zeros([maxlen], dtype=np.int32)
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    nxt = session[-1]
    idx = maxlen - 1

    ts = set(session)
    for itemId in reversed(session[:-1]):
        seq[idx] = itemId
        pos[idx] = nxt
        if nxt != 0:
            neg[idx] = random_neg(item_list, ts)
        nxt = itemId
        idx -= 1
        if idx == -1:
            break

    return seq, pos, neg


def sampler(user_train, item_set, batch_size, maxlen):
    SEED = random.randint(0, 2e9)
    random.seed(SEED)

    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample(user_train, item_set, maxlen))

    return zip(*one_batch)


def evaluate(inputs, item_list, model, args, sess, mode):
    MRR_20 = 0.0
    RECALL_20 = 0.0
    ranks = []
    sess_num = len(inputs)
    batch_num = int(sess_num / args.test_batch)
    test_item = item_list

    for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b', desc=mode):
        sess_indices = random.sample(range(sess_num), args.test_batch)
        ground_truth = []
        seq = []
        for sess_index in sess_indices:
            length = len(inputs[sess_index])
            if length <= 2:
                seq.append([inputs[sess_index][0]])
            elif length > (args.maxlen + 1):
                seq.append(inputs[sess_index][length-args.maxlen-1:-1])
            else:
                seq.append(inputs[sess_index][:-1])
            ground_truth.append(copy.deepcopy(inputs[sess_index][-1]))
            while len(seq[-1]) < args.maxlen:
                seq[-1].insert(0, 0)

        predictions = model.predict(sess, seq, test_item)
        rank = list(map(lambda label, pred: pred[test_item.index(label)], ground_truth, predictions))
        ranks.extend(rank)

    valid_user = len(ranks)
    valid_ranks_20 = list(filter(lambda x: x < 20, ranks))
    valid_ranks_10 = list(filter(lambda x: x < 10, ranks))
    RECALL_20 = len(valid_ranks_20)
    MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
    RECALL_10 = len(valid_ranks_10)
    MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))

    return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user


def evaluate_neg(inputs, item_list, model, args, sess, mode):
    MRR_20 = 0.0
    RECALL_20 = 0.0
    ranks = []
    sess_num = len(inputs)
    batch_num = int(sess_num / args.test_batch)
    test_item = item_list

    for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b', desc=mode):
        ground_truth = []
        seq = []
        for i in range(args.test_batch):
            random_sess = inputs[random.choice(range(sess_num))]
            if random_sess[-1] in ground_truth:
                random_sess = inputs[random.choice(range(sess_num))]

            length = len(random_sess)
            if length <= 2:
                seq.append([random_sess[0]])
            elif length > (args.maxlen + 1):
                seq.append(random_sess[length-args.maxlen-1:-1])
            else:
                seq.append(random_sess[:-1])
            ground_truth.append(copy.deepcopy(random_sess[-1]))
            while len(seq[-1]) < args.maxlen:
                seq[-1].insert(0, 0)

        predictions = model.predict(sess, seq, ground_truth)
        rank = list(map(lambda label, pred: pred[ground_truth.index(label)], ground_truth, predictions))
        ranks.extend(rank)

    valid_user = len(ranks)
    valid_ranks_20 = list(filter(lambda x: x < 20, ranks))
    valid_ranks_10 = list(filter(lambda x: x < 10, ranks))
    RECALL_20 = len(valid_ranks_20)
    MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
    RECALL_10 = len(valid_ranks_10)
    MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))

    return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user


def evaluate_all(inputs, item_list, model, args, sess, mode):
    ranks = []
    sess_num = len(inputs)
    batch_num = int(sess_num / args.test_batch)
    test_item = item_list

    for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b', desc=mode):
        sess_indices = random.sample(range(sess_num), args.test_batch)
        ground_truth = []
        seq = []

        for sess_index in sess_indices:
            length = len(inputs[sess_index])
            if length > (args.maxlen + 1):
                seq.append(inputs[sess_index][length-args.maxlen-1:-1])
                ground_truth.append(inputs[sess_index][length-args.maxlen:])
            else:
                seq.append(inputs[sess_index][:-1])
                ground_truth.append(inputs[sess_index][1:])
            while len(seq[-1]) < args.maxlen:
                seq[-1].insert(0, 0)
            while len(ground_truth[-1]) < args.maxlen:
                ground_truth[-1].insert(0, 0)

        predictions = model.predict_all(sess, seq, test_item)
        ground_truth = np.array(ground_truth).flatten()
        predictions = predictions[np.where(ground_truth != 0)]
        ground_truth = ground_truth[np.where(ground_truth != 0)]
        # rank = list(map(lambda label, pred: pred[test_item.index(label)], ground_truth, predictions))
        rank = [pred[test_item.index(label)] for pred, label in zip(predictions, ground_truth)]
        ranks.extend(rank)

    valid_user = len(ranks)
    valid_ranks_20 = list(filter(lambda x: x < 20, ranks))
    valid_ranks_10 = list(filter(lambda x: x < 10, ranks))
    RECALL_20 = len(valid_ranks_20)
    MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
    RECALL_10 = len(valid_ranks_10)
    MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))

    return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user


class ContinueLearningPlot:
    def __init__(self, args):
        self.args = args
        self.periods = []
        self.epochs = []
        self.MRR20 = []
        self.RECALL20 = []
        self.MRR10 = []
        self.RECALL10 = []

    def add(self, period, epoch, t_test):
        if period > 1 and epoch < self.epochs[-1]:
            del self.periods[-1]
            del self.epochs[-1]
            del self.MRR20[-1]
            del self.RECALL20[-1]
            del self.MRR10[-1]
            del self.RECALL10[-1]

        self.periods.append(period)
        self.epochs.append(epoch)
        self.MRR20.append(t_test[0])
        self.RECALL20.append(t_test[1])
        self.MRR10.append(t_test[2])
        self.RECALL10.append(t_test[3])

    def plot(self, logs):
        x_label = list(map(lambda period, epoch: 'P%dE%d' % (period, epoch), self.periods, self.epochs))

        plt.figure()
        plt.plot(range(len(self.MRR20)), self.MRR20, label='MRR@20', color='b')
        plt.plot(range(len(self.MRR20)), self.MRR10, label='MRR@10', color='g')
        plt.plot(range(len(self.MRR20)), self.RECALL20, label='RECALL20', color='c')
        plt.plot(range(len(self.MRR20)), self.RECALL10, label='RECALL10', color='y')

        if self.args.dataset == 'DIGINETICA':
            NARM_RECALL20 = 0.4970
            NARM_RECALL10 = 0.3362
        elif self.args.dataset == 'YOOCHOOSE':
            NARM_RECALL20 = 0.6973
            NARM_RECALL10 = 0.5870
        plt.hlines(NARM_RECALL20, 0, len(self.MRR20)-1, label='NARM_RECALL20', color='r')
        plt.hlines(NARM_RECALL10, 0, len(self.MRR20)-1, label='NARM_RECALL10', color='m')

        plt.xticks(range(len(self.MRR20)), x_label, rotation=90, size='small')
        plt.title('Continue learning test results')
        plt.legend()

        i = 0
        while os.path.isfile('Coutinue_Learning_result%d.pdf' % i):
            i += 1
        plt.savefig('Coutinue_Learning_result%d.pdf' % i)
        plt.close()
        logs.write('Coutinue_Learning_result%d.pdf\n' % i)


class ExemplarSet:

    def __init__(self, args, item_list, m):
        # self.exemplars = dict.fromkeys(item_list, [None] * m)
        self.exemplars = {item: [] for item in item_list}
        self.m = m
        self.args = args

    def load(self, period):
        exemplars = []
        with open('ExemplarSetPeriod=%d.pickle' % (period - 1), mode='rb') as file:
            exemplars_item = pickle.load(file)
        for item in exemplars_item.values():
            if isinstance(item, list):
                exemplars.extend([i for i in item if i])
        return exemplars

    def add(self, rep, item, seq):
        if self.args.is_herding:
            # Initialize mean and selected ids
            D = rep.T / np.linalg.norm(rep.T, axis=0)
            seq = np.array(seq)
            mu = D.mean(axis=1)
            w_t = mu
            step_t = 0
            while not(len(self.exemplars[item]) == self.m) and step_t < 1.1*self.m:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                w_t = w_t + mu - D[:, ind_max]
                tmp_exemplar = np.append(seq[ind_max], item).tolist()
                step_t += 1
                if tmp_exemplar not in self.exemplars[item]:
                    self.exemplars[item].append(tmp_exemplar)
        else:
            seq_num = len(seq)
            if seq_num > self.m:
                selected_ids = np.random.choice(seq_num, self.m, replace=False)
            else:
                selected_ids = list(range(seq_num))
            self.exemplars[item] = list(map(lambda i: np.append(seq[i], item), selected_ids))


    def save(self, period, info):
        if not os.path.isdir('exemplar'):
            os.makedirs(os.path.join('exemplar'))
        with open('exemplar/%sExemplarSetPeriod=%d.pickle' % (info, period), mode='wb') as file:
            pickle.dump(self.exemplars, file)


def save_load_args(args):
    if args.mode == 'train':
        with open('train_args.txt', 'w') as f:
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

