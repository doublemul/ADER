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
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import time


class DataLoader:
    def __init__(self, args, logs):
        self.item_set = set()
        self.path = os.path.join('..', '..', 'data', args.dataset)
        self.is_remove_item = args.remove_item
        self.logs = logs

    def train_loader(self, period):
        Sessions = defaultdict(list)
        with open(self.path + '/train_%d.txt' % period, 'r') as f:
            for line in f:
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                self.item_set.add(itemId)
                Sessions[sessId].append(itemId)
        sessions = list(Sessions.values())
        if period > 1:
            sessions.extend(self.load_previous_evaluate_data(period))
        info = 'Train set information: total number of action: %d.' \
               % sum(list(map(lambda session: len(session), sessions)))
        self.logs.write(info + '\n')
        print(info)
        return sessions

    def load_previous_evaluate_data(self, period):
        period = period - 1
        Sessions = defaultdict(list)
        for name in ['valid', 'test']:
            with open(self.path + '/%s_%d.txt' % (name, period), 'r') as f:
                for line in f:
                    sessId, itemId = line.rstrip().split(' ')
                    sessId = int(sessId)
                    itemId = int(itemId)
                    self.item_set.add(itemId)
                    Sessions[sessId].append(itemId)
        sessions = list(Sessions.values())
        return sessions

    def evaluate_loader(self, period, mode):
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        with open(self.path + '/%s_%d.txt' % (mode, period), 'r') as f:
            for line in f:
                total_num += 1
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                # remove new items in test or validation set that not appear in train set
                if self.is_remove_item and (itemId not in self.item_set):
                    removed_num += 1
                    continue
                else:
                    self.item_set.add(itemId)
                Sessions[sessId].append(itemId)
        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    delete_keys.append(sessId)
            for delete_key in delete_keys:
                del Sessions[delete_key]
        if mode == 'test':
            info = 'Test'
        else:
            info = 'Validation'
        info = '%s set information: original total number of action: %d, removed number of action: %d.' \
               % (info, total_num, removed_num)
        self.logs.write(info + '\n')
        print(info)
        sessions = list(Sessions.values())
        return sessions

    def max_item(self, period):
        if (period <= 1) and (len(self.item_set) != max(self.item_set)):
            print('Item index error!')
            self.logs.write('Item index error!')
        return max(self.item_set)


class Sampler:
    def __init__(self, args, item_num, is_train):
        self.item_num = item_num
        self.maxlen = args.maxlen
        self.is_train = is_train
        if is_train:
            self.batch_size = args.batch_size
        else:
            self.batch_size = args.test_batch

    def label_generator(self, session):
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        nxt = session[-1]
        idx = self.maxlen - 1
        if self.is_train:
            neg = np.zeros([self.maxlen], dtype=np.int32)
            ts = set(session)

        for itemId in reversed(session[:-1]):
            seq[idx] = itemId
            pos[idx] = nxt
            if nxt != 0 and self.is_train:
                neg[idx] = self.negative_generator(ts)
            nxt = itemId
            idx -= 1
            if idx == -1:
                break

        if self.is_train:
            return seq, pos, neg
        else:
            return seq, pos

    def negative_generator(self, ts):
        neg = random.randint(1, self.item_num)
        while neg in ts:
            neg = random.randint(1, self.item_num)
        return neg

    def sampler(self, data):
        SEED = random.randint(0, 2e9)
        random.seed(SEED)

        one_batch = []
        for i in range(self.batch_size):
            session = random.choice(data)
            while len(session) <= 1:
                session = random.choice(data)
            one_batch.append(self.label_generator(session))

        return zip(*one_batch)


class Evaluator:
    def __init__(self):






def evaluate_last(inputs, item_list, model, args, sess, mode):
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
                seq.append(inputs[sess_index][length - args.maxlen - 1:-1])
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
    ranks = []
    sess_num = len(inputs)
    batch_num = int(sess_num / args.test_batch)

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
                seq.append(random_sess[length - args.maxlen - 1:-1])
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


def evaluate_all(inputs, item_list, model, args, sess, info):
    ranks = []
    sess_num = len(inputs)
    batch_num = int(sess_num / args.test_batch)
    test_item = item_list

    for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b', desc=info):
        sess_indices = random.sample(range(sess_num), args.test_batch)
        ground_truth = []
        seq = []
        for sess_index in sess_indices:
            length = len(inputs[sess_index])
            if length > (args.maxlen + 1):
                seq.append(inputs[sess_index][length - args.maxlen - 1:-1])
                ground_truth.append(inputs[sess_index][length - args.maxlen:])
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
        rank = [pred[index-1] for pred, index in zip(predictions, ground_truth)]
        ranks.extend(rank)

    valid_user = len(ranks)
    valid_ranks_20 = list(filter(lambda x: x < 20, ranks))
    valid_ranks_10 = list(filter(lambda x: x < 10, ranks))
    RECALL_20 = len(valid_ranks_20)
    MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
    RECALL_10 = len(valid_ranks_10)
    MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))

    return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user


class ExemplarGenerator:

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
            while not (len(self.exemplars[item]) == self.m) and step_t < 1.1 * self.m:
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
