#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : util.py
# @Description  : Some class to read data, evaluate results, select exemplar
import random
import os
import pickle
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm


class DataLoader:
    """
    DataLoader object to load train, valid and test data from dataset.
    """
    def __init__(self, args, item_num, logs):
        """
        :param args: args
        :param item_num: all item number of entire dataset
        :param logs: logs
        """
        self.logs = logs
        self.args = args
        self.item_set = set()
        if args.is_joint:
            self.path = os.path.join('..', '..', 'data', '%s_joint' % args.dataset)
        else:
            self.path = os.path.join('..', '..', 'data', args.dataset)
        self.is_remove_item = args.remove_item
        self.item_counter = np.zeros(item_num)

    def train_loader(self, period=None):
        """
        This method return train data of specific period
        :param period: current period
        :return: train data of current period
        """
        Sessions = defaultdict(list)
        train_item_set = set()
        if self.args.is_joint:
            file_name = '/train.txt'
        else:
            file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
            for line in f:
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                self.item_set.add(itemId)
                Sessions[sessId].append(itemId)
                self.item_counter[itemId - 1] += 1
                train_item_set.add(itemId)
        sessions = list(Sessions.values())
        del Sessions
        info = 'Train set information: total number of action: %d.' \
               % sum(list(map(lambda session: len(session), sessions)))
        self.logs.write(info + '\n')
        print(info)

        for sess in sessions:
            self.item_counter[sess[0] - 1] -= 1

        return sessions, train_item_set

    def get_item_counter(self):
        """
        This method return numbers of
        :return: list of cumulative numbers with length of maximum item
        """
        return self.item_counter

    def evaluate_loader(self, period=None):
        """
        This method load and return test or valid data according to mode of specific period
        :param period: current period
        :return: test data
        """
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        test_item_set = set()
        if self.args.is_joint:
            file_name = '/test.txt'
        else:
            file_name = '/period_%d.txt' % period
        with open(self.path + file_name, 'r') as f:
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
                    test_item_set.add(itemId)
                Sessions[sessId].append(itemId)

        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    delete_keys.append(sessId)
            for delete_key in delete_keys:
                del Sessions[delete_key]

        info = 'Test set information: original total number of action: %d, removed number of action: %d.' \
               % (total_num, removed_num)
        self.logs.write(info + '\n')
        print(info)
        sessions = list(Sessions.values())
        del Sessions

        return sessions, total_num - len(sessions), test_item_set

    def max_item(self):
        """
        This method returns the maximum item in item set.
        """
        return max(self.item_set)


class Sampler:
    """
    This object samples data and generates positive labels for train, valid and test, as well as negative sample for
    train.
    """

    def __init__(self, args, data, batch_size):
        """
        :param args: args
        :param data: original data for sampling
        :param batch_size: size of one batch
        """
        self.maxlen = args.maxlen
        self.data = data
        self.batch_size = batch_size

        self.prepared_data = []
        self.dataset_size = 0
        self.batch_num = 0

        self.batch_counter = 0
        self.data_indices = []

        self.SEED = args.random_seed
        random.seed(args.random_seed)

    def label_generator(self, session, return_pos=True):
        """
        This method return input sequence as well as positive and negative sample
        :param return_pos: if True, return processed session and label, else only return processed session
        :param session: a item sequence
        :return: input sequence, [label]
        """
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.array(session[-1], dtype=np.int32)
        idx = self.maxlen - 1

        for itemId in reversed(session[:-1]):
            seq[idx] = itemId
            idx -= 1
            if idx == -1:
                break
        if return_pos:
            return seq, pos
        else:
            return seq

    def prepare_data(self):
        """
        Split original session into sub-sessions
        """
        self.prepared_data = []
        for session in self.data:
            self.prepared_data.append(session)
            length = len(session)
            if length > 2:
                for t in range(1, length - 1):
                    self.prepared_data.append(session[:-t])

    def add_exemplar(self, exemplar):
        """
        Only add exemplar data
        :param exemplar: exemplar
        """
        if exemplar:
            for session in exemplar:
                self.prepared_data.append(session)

    def add_full_exemplar(self, exemplar):
        """
        Add exemplar data and logits
        :param exemplar: exemplar data and logits
        """
        self.logits = []
        for session, logits in exemplar:
            self.prepared_data.append(session)
            self.logits.append(logits)

    def split_data(self, valid_portion, return_train=False):
        """
        Split data into valid and train dataset
        :param valid_portion: the portion of validation dataset w.r.t entire dataset
        :param return_train: if True, return validation data and train data, else only return validation data
        :return:
        """

        data_size = len(self.prepared_data)
        sidx = np.arange(data_size, dtype='int32')
        np.random.shuffle(sidx)
        n_train = int(np.round(data_size * (1. - valid_portion)))
        valid_data = [self.prepared_data[s] for s in sidx[n_train:]]
        train_data = [self.prepared_data[s] for s in sidx[:n_train]]
        self.prepared_data = train_data

        if return_train:
            return valid_data, train_data
        else:
            return valid_data

    def shuffle_data(self):
        """
        Shuffle data
        :return: batch number
        """
        self.dataset_size = len(self.prepared_data)
        self.batch_num = math.ceil(self.dataset_size * 1.0 / self.batch_size)
        self.batch_counter = 0
        self.data_indices = list(range(self.dataset_size))
        random.Random(self.SEED).shuffle(self.data_indices)

    def sampler(self):
        """
        This method returns a batch of sample: (seq, pos (,neg))
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < self.dataset_size:
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                one_batch.append(self.label_generator(session))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num:
            self.batch_counter = 0
            random.Random(self.SEED).shuffle(self.data_indices)

        return zip(*one_batch)

    def exemplar_sampler(self):
        """
        This method returns a batch of sample: (seq, pos (,neg))
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < self.dataset_size:
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.prepared_data[index]
                if len(session) <= 1:
                    continue
                seq, pos = self.label_generator(session)
                one_batch.append((seq, pos, self.logits[index]))
            else:
                break

        self.batch_counter += 1
        if self.batch_counter == self.batch_num:
            self.batch_counter = 0
            random.Random(self.SEED).shuffle(self.data_indices)

        return zip(*one_batch)

    def data_size(self):
        return len(self.prepared_data)


class Evaluator:
    """
    This object evaluates performance on valid or test data.
    """

    def __init__(self, args, data, max_item, mode, model, sess, logs):
        """
        :param args: args
        :param data: data to evaluate, valid data or test data
        :param max_item: maximum item at current period
        :param model: model
        :param mode: 'valid' or 'test'
        :param sess: tf session
        :param logs: logs
        """
        self.args = args
        self.data = data
        self.max_item = max_item
        self.mode = mode
        self.model = model
        self.sess = sess

        self.logs = logs
        self.ranks = []
        self.recall_20 = 0
        self.desc = 'Validating epoch ' if mode == 'valid' else 'Testing epoch '

    def evaluate(self, epoch, exemplar=None):
        """
        This method only evaluate performance of predicted last item among all existing item.
        :param exemplar: valid exemplar from previous period
        :param epoch: current epoch
        """
        self.ranks = []

        evaluate_sampler = Sampler(self.args, self.data, self.args.test_batch)
        evaluate_sampler.prepare_data()
        evaluate_sampler.add_exemplar(exemplar=exemplar)
        evaluate_sampler.shuffle_data()
        batch_num = evaluate_sampler.batch_num
        for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                      desc=self.desc + str(epoch)):
            seq, pos = evaluate_sampler.sampler()
            predictions = self.model.predict(self.sess, seq, list(range(1, self.max_item + 1)))
            ground_truth = pos
            rank = [pred[index - 1] for pred, index in zip(predictions, ground_truth)]
            self.ranks.extend(rank)
        self.display(epoch)

    def results(self):
        """
        This method returns evaluation metrics(MRR@20, RECALL@20, MRR@10, RECALL@10)
        """
        valid_user = len(self.ranks)
        valid_ranks_20 = list(filter(lambda x: x < 20, self.ranks))
        valid_ranks_10 = list(filter(lambda x: x < 10, self.ranks))
        RECALL_20 = len(valid_ranks_20)
        MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
        RECALL_10 = len(valid_ranks_10)
        MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))
        return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user

    def display(self, epoch):
        """
        This method display and save evaluation metrics(MRR@20, RECALL@20, MRR@10, RECALL@10)
        """
        results = self.results()
        info = 'epoch:%d, %s (MRR@20: %.4f, RECALL@20: %.4f, MRR@10: %.4f, RECALL@10: %.4f)' \
               % (epoch, self.mode, results[0], results[1], results[2], results[3])
        print(info)
        self.logs.write(info + '\n')


class ExemplarGenerator:
    """
    This object select exemplars from given dataset
    """

    def __init__(self, args, m, data, max_item, logs):
        """
        :param args: args
        :param m: number of exemplars per item
        :param data: dataset, train data or valid data
        :param max_item: accumulative number of item
        :param logs: logs
        """
        self.sess_by_item = defaultdict(list)
        self.exemplars = dict()
        self.m = m
        self.args = args
        self.data = data
        self.max_item = max_item
        self.logs = logs
        self.item_count = np.zeros(max_item)

    def add_exemplar(self, exemplar=None, item_set=None):
        """
        This method sorts sub-sessions by their last item (label).
        """
        self.sess_by_item = defaultdict(list)
        exemplar_sampler = Sampler(self.args, self.data, self.args.batch_size)
        exemplar_sampler.prepare_data()
        exemplar_sampler.add_exemplar(exemplar=exemplar)
        exemplar_sampler.shuffle_data()
        batch_num = exemplar_sampler.batch_num

        for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                      desc='Sorting exemplars'):
            seq, pos = exemplar_sampler.sampler()
            pos = np.array(pos)
            for s, item in zip(seq, pos):
                if item_set is not None and item not in item_set:
                    continue
                session = np.append(s, item)
                self.sess_by_item[item].append(session)
                self.item_count[item - 1] += 1

    def herding(self, rep, logits, item, seq, m):
        """
        Herding algorithm for exemplar selection
        :param rep: representations
        :param logits: logits
        :param item: label
        :param seq: input session (item sequence)
        :param m: number of exemplar per label
        """
        # Initialize mean and selected ids
        D = rep.T / np.linalg.norm(rep.T, axis=0)
        mu = D.mean(axis=1)
        w_t = mu
        step_t = 0
        selected_ids = []
        counter = 0
        while not (len(selected_ids) == m) and step_t < 1.1 * m:  # 2
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            w_t = w_t + mu - D[:, ind_max]
            step_t += 1
            if ind_max not in selected_ids:
                selected_ids.append(ind_max)
                counter += 1
        self.exemplars[item] = [[seq[i][seq[i] != 0].tolist(), logits[i].tolist()] for i in selected_ids]
        return counter

    def herding_selection(self, sess, model, history_item_count=None):
        """
        This method selects exemplars using herding and selects exemplars, the number of exemplars is proportional to
        item frequency.
        """
        self.exemplars = defaultdict(list)
        if history_item_count is not None:
            self.item_count = history_item_count
        if self.args.disable_m:
            self.item_count = np.ones_like(self.item_count)
        item_prob = self.item_count / self.item_count.sum()
        item_count = np.random.multinomial(n=self.m, pvals=item_prob, size=1)[0]
        item_count = np.int32(item_count)

        saved_num = 0
        for item in tqdm(self.sess_by_item, ncols=70, leave=False, unit='b', desc='Selecting exemplar'):
            m = item_count[item - 1]
            seq = self.sess_by_item[item]
            seq = np.array(seq)
            input_seq = seq[:, :-1]
            rep, logits = sess.run([model.rep, model.logits], {model.input_seq: input_seq,
                                                               model.dropout_rate: self.args.dropout_rate,
                                                               model.max_item: self.max_item,
                                                               model.is_training: False})
            rep = np.array(rep)
            logits = np.array(logits)
            saved = self.herding(rep, logits, item, seq, min(m, len(seq)))
            saved_num += saved
        print('Total saved exemplar: %d' % saved_num)
        self.logs.write('Total saved exemplar: %d\n' % saved_num)

    def loss_selection(self, sess, model, history_item_count=None):
        """
        This method selects exemplars by ranking loss, the number of exemplars is proportional to
        item frequency.
        """
        self.exemplars = defaultdict(list)
        if history_item_count is not None:
            self.item_count = history_item_count
        item_prob = self.item_count / self.item_count.sum()
        item_count = np.random.multinomial(n=self.m, pvals=item_prob, size=1)[0]
        item_count = np.int32(item_count)

        saved_num = 0
        for item in tqdm(self.sess_by_item, ncols=70, leave=False, unit='b', desc='Selecting exemplar'):
            m = item_count[item - 1]
            if m < 0.5:
                continue
            seq = self.sess_by_item[item]
            seq_num = len(seq)
            seq = np.array(seq)
            loss, logits = sess.run([model.loss, model.logits], {model.input_seq: seq[:, :-1],
                                                                 model.pos: seq[:, -1],
                                                                 model.dropout_rate: self.args.dropout_rate,
                                                                 model.max_item: self.max_item,
                                                                 model.is_training: False})
            loss = np.array(loss)
            logits = np.array(logits)
            selected_ids = loss.argsort()[:int(min(m, seq_num))]
            self.exemplars[item] = [[seq[i][seq[i] != 0].tolist(), logits[i].tolist()] for i in selected_ids]
            saved_num += len(selected_ids)
        print('Total saved exemplar: %d' % saved_num)
        self.logs.write('Total saved exemplar: %d\n' % saved_num)

    def randomly_selection(self, sess, model, history_item_count=None):
        """
        This method randomly selects exemplars, and selects equivalent number of exemplar for each label.
        """
        self.exemplars = defaultdict(list)
        if history_item_count is not None:
            self.item_count = history_item_count
        item_prob = self.item_count / self.item_count.sum()
        item_count = np.random.multinomial(n=self.m, pvals=item_prob, size=1)[0]
        item_count = np.int32(item_count)
        self.exemplars = defaultdict(list)

        saved_num = 0
        for item in tqdm(self.sess_by_item, ncols=70, leave=False, unit='b', desc='Selecting exemplar'):
            seq = self.sess_by_item[item]
            seq = np.array(seq)
            seq_num = len(seq)
            m = item_count[item - 1]
            if m > 0:
                selected_ids = np.random.choice(seq_num, min(m, seq_num), replace=False)
                selected_seq = seq[selected_ids]
                logits = sess.run(model.logits, {model.input_seq: selected_seq[:, :-1],
                                                 model.dropout_rate: self.args.dropout_rate,
                                                 model.max_item: self.max_item,
                                                 model.is_training: False})
                logits = np.array(logits)
                for s, l in zip(selected_seq, logits):
                    self.exemplars[item].append([s[s != 0].tolist(), l.tolist()])
                    saved_num += 1
        print('Total saved exemplar: %d' % saved_num)
        self.logs.write('Total saved exemplar: %d\n' % saved_num)

    def save(self, mode):
        """
        This method save the generated exemplars
        """
        with open('%s_exemplar.pickle' % mode, mode='wb') as file:
            pickle.dump(self.exemplars, file)
        del self.exemplars
