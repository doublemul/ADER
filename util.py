#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : util.py
# @Description  :
import random
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


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
    def __init__(self, args, max_item, is_train):
        self.max_item = max_item
        self.maxlen = args.maxlen
        self.is_train = is_train
        if is_train:
            self.batch_size = args.batch_size
        else:
            self.batch_size = args.test_batch
        SEED = random.randint(0, 2e9)
        random.seed(SEED)

    def label_generator(self, session):
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        nxt = session[-1]
        idx = self.maxlen - 1
        if self.is_train:
            neg = np.zeros([self.maxlen], dtype=np.int32)

        for itemId in reversed(session[:-1]):
            seq[idx] = itemId
            pos[idx] = nxt
            if nxt != 0 and self.is_train:
                neg[idx] = self.negative_generator(session)
            nxt = itemId
            idx -= 1
            if idx == -1:
                break

        if self.is_train:
            return seq, pos, neg
        else:
            return seq, pos

    def negative_generator(self, session):
        neg = random.randint(1, self.max_item)
        while neg in session:
            neg = random.randint(1, self.max_item)
        return neg

    def sampler(self, data):
        one_batch = []
        for i in range(self.batch_size):
            session = random.choice(data)
            while len(session) <= 1:
                session = random.choice(data)
            one_batch.append(self.label_generator(session))
        return zip(*one_batch)

    def evaluate_negative_sampler(self, data):
        one_batch = []
        ground_truth = []
        for i in range(self.batch_size):
            session = random.choice(data)
            while (len(session) <= 1) or session[-1] in ground_truth:
                session = random.choice(data)
            ground_truth.append(session[-1])
            one_batch.append(self.label_generator(session))
        return zip(*one_batch)


class Evaluator:
    def __init__(self, args, data, max_item, model, mode, sess, logs):

        self.data = data
        self.max_item = max_item
        self.mode = mode
        self.sess = sess
        self.model = model
        self.logs = logs

        self.ranks = []
        self.recall_20 = 0
        self.batch_num = int(len(data) / args.test_batch)
        self.evaluate_sampler = Sampler(args=args, max_item=max_item, is_train=False)
        self.desc = 'Validating epoch ' if mode == 'valid' else 'Testing epoch '

    def last_item(self, epoch):
        self.ranks = []
        for _ in tqdm(range(self.batch_num), total=self.batch_num, ncols=70, leave=False, unit='b',
                      desc=self.desc+str(epoch)):
            seq, pos = self.evaluate_sampler.sampler(self.data)

            predictions = self.model.predict(self.sess, seq, list(range(1, self.max_item)))
            ground_truth = [sess[-1] for sess in pos]
            rank = [pred[index - 1] for pred, index in zip(predictions, ground_truth)]
            self.ranks.extend(rank)
        self.display(epoch)

    def full_item(self, epoch):
        self.ranks = []
        for _ in tqdm(range(self.batch_num), total=self.batch_num, ncols=70, leave=False, unit='b',
                      desc=self.desc + str(epoch)):
            seq, pos = self.evaluate_sampler.sampler(self.data)

            predictions = self.model.predict_all(self.sess, seq, list(range(1, self.max_item)))
            ground_truth = np.array(pos).flatten()

            predictions = predictions[np.where(ground_truth != 0)]
            ground_truth = ground_truth[np.where(ground_truth != 0)]

            rank = [pred[index - 1] for pred, index in zip(predictions, ground_truth)]
            self.ranks.extend(rank)
        self.display(epoch)

    def neg_sample(self, epoch):
        self.ranks = []
        for _ in tqdm(range(self.batch_num), total=self.batch_num, ncols=70, leave=False, unit='b',
                      desc=self.desc + str(epoch)):
            seq, pos = self.evaluate_sampler.evaluate_negative_sampler(self.data)

            ground_truth = [item[-1] for item in pos]
            predictions = self.model.predict_all(self.sess, seq, ground_truth)

            rank = [pred[ground_truth.index(label)] for pred, label in zip(ground_truth, predictions)]
            self.ranks.extend(rank)
        self.display(epoch)

    def results(self):
        valid_user = len(self.ranks)
        valid_ranks_20 = list(filter(lambda x: x < 20, self.ranks))
        valid_ranks_10 = list(filter(lambda x: x < 10, self.ranks))
        RECALL_20 = len(valid_ranks_20)
        MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
        RECALL_10 = len(valid_ranks_10)
        MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))
        self.recall_20 = RECALL_20 / valid_user
        return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user

    def display(self, epoch):
        results = self.results()
        info = 'epoch:%d, %s (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
               % (epoch, self.mode, results[0], results[1], results[2], results[3])
        print(info)
        self.logs.write(info + '\n')


class ExemplarGenerator:

    def __init__(self, args, max_item, m):
        # self.exemplars = dict.fromkeys(item_list, [None] * m)
        self.exemplars = {item: [] for item in list(range(1, max_item + 1))}
        self.m = m
        self.args = args

    def load(self, period):
        """
        This function load exemplar in previous period
        :param period: this period
        :return:
        """
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


    def sorted_by_last_item:





