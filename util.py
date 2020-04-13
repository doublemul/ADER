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
import math
from collections import defaultdict
from matplotlib import pyplot as plt
from tqdm import tqdm


class DataLoader:
    """
    DataLoader object to load train, valid and test data from dataset.
    """

    def __init__(self, args, logs):
        """
        :param args: args
        :param logs: logs
        """
        self.item_set = set()
        self.path = os.path.join('..', '..', 'data', args.dataset)
        self.is_remove_item = args.remove_item
        self.logs = logs
        self.item_counter = defaultdict(lambda: 0)

    def train_loader(self, period):
        """
        This method return train data of specific period
        :param period: current period
        :return: train data of current period
        """
        Sessions = defaultdict(list)
        self.item_counter = defaultdict(lambda: 0)
        with open(self.path + '/train_%d.txt' % period, 'r') as f:
            for line in f:
                sessId, itemId = line.rstrip().split(' ')
                sessId = int(sessId)
                itemId = int(itemId)
                self.item_set.add(itemId)
                Sessions[sessId].append(itemId)
                self.item_counter[itemId] += 1
        sessions = list(Sessions.values())
        del Sessions
        # if period > 1:
        #     sessions.extend(self.load_previous_evaluate_data(period))
        info = 'Train set information: total number of action: %d.' \
               % sum(list(map(lambda session: len(session), sessions)))
        self.logs.write(info + '\n')
        print(info)

        for sess in sessions:
            self.item_counter[sess[0]] -= 1

        return sessions

    # def load_previous_evaluate_data(self, period):
    #     """
    #     This method return valid and test data from previous period to be extended behind current train data
    #     :param period: current period
    #     :return valid and test data from previous period
    #     """
    #     period = period - 1
    #     Sessions = defaultdict(list)
    #     for name in ['valid', 'test']:
    #         with open(self.path + '/%s_%d.txt' % (name, period), 'r') as f:
    #             for line in f:
    #                 sessId, itemId = line.rstrip().split(' ')
    #                 sessId = int(sessId)
    #                 itemId = int(itemId)
    #                 self.item_set.add(itemId)
    #                 self.item_counter[itemId] += 1
    #                 Sessions[sessId].append(itemId)
    #     sessions = list(Sessions.values())
    #     del Sessions
    #     return sessions

    def get_item_counter(self, exemplar=None):
        """
        This method return numbers of
        :return: list of cumulative numbers with length of maximum item
        """
        max_item = self.max_item()
        item_count = np.zeros(max_item)
        for item in self.item_counter.keys():
            item_count[item - 1] = self.item_counter[item]

        if exemplar:
            for sess in exemplar:
                item_count[sess[-1]-1] += 1
        return item_count

    def evaluate_loader(self, period, mode):
        """
        This method load and return test or valid data according to mode of specific period
        :param period: current period
        :param mode: 'test' or 'valid'
        :return: test or valid data according to mode
        """
        Sessions = defaultdict(list)
        removed_num = 0
        total_num = 0
        self.item_counter = defaultdict(lambda: 0)
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
                self.item_set.add(itemId)
                self.item_counter[itemId] += 1
                Sessions[sessId].append(itemId)

        if self.is_remove_item:
            delete_keys = []
            for sessId in Sessions:
                if len(Sessions[sessId]) == 1:
                    removed_num += 1
                    self.item_counter[Sessions[sessId][0]] -= 1
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
        del Sessions

        for sess in sessions:
            self.item_counter[sess[0]] -= 1

        return sessions

    def max_item(self):
        """
        This method returns the maximum item in item set.
        """
        # if len(self.item_set) != max(self.item_set):
        #     print('Item index error!')
        #     self.logs.write('Item index error!')
        return max(self.item_set)


class Sampler:
    """
    This object samples data and generates positive labels for train, valid and test, as well as negative sample for
    train.
    """

    def __init__(self, args, data, is_train, max_item=None, cumulative=None):
        """
        :param args: args
        :param data: original data for sampling
        :param is_train: boolean: train or evaluation (valid/test) sampler, for train, it also return negative sample
        :param max_item: maximum item in train dataset, to generate negative sample randomly
        :param cumulative: cumulative list according to item frequency, to generate negative sample by frequency
        """
        self.data = data
        self.max_item = max_item
        self.maxlen = args.maxlen
        self.is_train = is_train
        self.cumulative = cumulative

        if is_train:
            self.batch_size = args.batch_size
        else:
            self.batch_size = args.test_batch
        self.batch_num = math.ceil(len(self.data) / self.batch_size)
        self.batch_counter = 0
        SEED = random.randint(0, 2e9)
        random.seed(SEED)
        self.dataset_size = len(data)
        self.data_indices = list(range(self.dataset_size))
        random.shuffle(self.data_indices)

        self.exemplar_initialized = False
        self.exemplar_indices = []

    def label_generator(self, session):
        """
        This method return input sequence as well as positive and negative sample
        :param session: a item sequence
        :return: train: input sequence, positive sample (label sequence), negative sample
                 valid/test: input sequence, positive sample (label sequence)
        """
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
        """
        This method generate negative sample. If cumulative is given, it generate by item frequency
        :param session: the generated negative sample should not in the original session
        """
        # if self.cumulative is None:
        neg = random.randint(1, self.max_item)
        while neg in session:
            neg = random.randint(1, self.max_item)
        # else:
        #     neg = self.cumulative.searchsorted(np.random.randint(self.cumulative[-1])) + 1
        #     while neg in session:
        #         neg = self.cumulative.searchsorted(np.random.randint(self.cumulative[-1])) + 1
        return neg

    def narm_sampler(self, reuse=None):
        """
        This method returns a batch of sample: (seq, pos (,neg))
        """
        one_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_counter * self.batch_size) < self.dataset_size:
                index = self.data_indices[i + self.batch_counter * self.batch_size]
                session = self.data[index]
            else:
                break
            if len(session) <= 1:
                continue
            length = len(session)
            one_batch.append(self.label_generator(session))
            if length > 2:
                for t in range(1, length - 1):
                    one_batch.append(self.label_generator(session[:-t]))
        self.batch_counter += 1
        if self.batch_counter == self.batch_num:
            self.batch_counter = 0
            random.shuffle(self.data_indices)
        if not reuse:
            return zip(*one_batch)
        else:
            return one_batch

    def hybrid_sampler(self, exemplar):
        one_batch = []
        if not self.exemplar_initialized:
            self.exemplar_indices = list(range(len(exemplar)))
            random.shuffle(self.exemplar_indices)
            self.exemplar_initialized = True

        exemplar_batch_size = math.ceil(len(exemplar) / self.batch_num)
        one_batch.extend(self.exemplar_sampler(exemplar_batch_size, exemplar))
        one_batch.extend(self.narm_sampler(reuse=True))
        return zip(*one_batch)

    def exemplar_sampler(self, batch_size, exemplar):
        """
        This method returns a batch of sample: (seq, pos (,neg))
        """
        one_batch = []
        for i in range(batch_size):
            if (i + self.batch_counter * batch_size) < len(exemplar):
                index = self.exemplar_indices[i + self.batch_counter * batch_size]
                session = exemplar[index]
            else:
                break
            if len(session) <= 1:
                continue
            one_batch.append(self.label_generator(session))
        return one_batch


class Evaluator:
    """
    This object evaluates performance on valid or test data.
    """

    def __init__(self, args, data, max_item, model, mode, sess, logs):
        """
        :param args: args
        :param data: data to evaluate, valid data or test data
        :param max_item: maximum item at current period
        :param model: model
        :param mode: 'valid' or 'test'
        :param sess: tf session
        :param logs: logs
        """
        self.data = data
        self.max_item = max_item
        self.mode = mode
        self.sess = sess
        self.model = model
        self.logs = logs
        self.args = args
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
        batch_num = math.ceil(len(self.data) / self.args.test_batch)
        sampler = Sampler(args=self.args, data=self.data, max_item=self.max_item, is_train=False)
        for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                      desc=self.desc + str(epoch)):
            seq, pos = sampler.hybrid_sampler(exemplar) if exemplar else sampler.narm_sampler()
            predictions = self.model.predict(self.sess, seq, list(range(1, self.max_item + 1)))
            ground_truth = [sess[-1] for sess in pos]
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
        self.recall_20 = RECALL_20 / valid_user
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

    def __init__(self, args, max_item, m, data, mode):
        """
        :param args: args
        :param max_item: number of existing items
        :param m: number of exemplars per item
        :param data: dataset, train data or valid data
        :param mode: 'train' or 'valid'
        """
        self.sess_rep_by_item = defaultdict(list)
        self.exemplars = dict()
        self.m = m
        self.mode = mode
        self.args = args
        self.max_item = max_item
        self.data = data

    def sort_by_item(self, exemplar=None):
        """
        This method sorts sessions by their last item.
        """
        self.sess_rep_by_item = defaultdict(list)
        sampler = Sampler(args=self.args, data=self.data, max_item=self.max_item, is_train=False)
        batch_num = math.ceil(len(self.data) / self.args.test_batch)
        for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                      desc='Sorting %s exemplars' % self.mode):
            seq, pos = sampler.hybrid_sampler(exemplar) if exemplar else sampler.narm_sampler()
            pos = np.array(pos)[:, -1]
            for session, item in zip(seq, pos):
                session = np.append(session, item)
                self.sess_rep_by_item[item].append(session)
        del sampler

    def herding(self, rep, item, seq, m):
        """
        Herding algorithm for exempler selection
        :param rep: representations
        :param item: label
        :param seq: input session (item sequence)
        :param m: number of exemplar per label
        """
        # Initialize mean and selected ids
        D = rep.T / np.linalg.norm(rep.T, axis=0)
        mu = D.mean(axis=1)
        w_t = mu
        step_t = 0
        while not (len(self.exemplars[item]) == m) and step_t < 1.5 * m:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            w_t = w_t + mu - D[:, ind_max]
            tmp_exemplar = seq[ind_max].tolist()
            step_t += 1
            if tmp_exemplar not in self.exemplars[item]:
                self.exemplars[item].append(tmp_exemplar)

    def herding_by_frequency(self, item_count, sess, model):
        """
        This method selects exemplars using herding and selects exemplars, the number of exemplars is proportional to
        item frequency.
        """
        self.exemplars = defaultdict(list)
        item_count = np.array(item_count)
        item_count = item_count * (self.m / item_count.sum())
        item_count = np.floor(item_count)
        item_count = np.int32(item_count)

        for item in tqdm(self.sess_rep_by_item,  ncols=70, leave=False, unit='b', desc='Selecting exemplar'):
            m = item_count[item - 1]
            if m < 0.5:
                continue
            seq = self.sess_rep_by_item[item]
            seq = np.array(seq)
            input_seq = seq[:, :-1]
            rep = sess.run(model.rep_last, {model.input_seq: input_seq, model.is_training: False})
            rep = np.array(rep)
            self.herding(rep, item, seq, min(m, len(seq)))
        del self.sess_rep_by_item

    # def herding_by_period(self):
    #     """
    #     This method selects exemplars using herding algorithm among all labels.
    #     """
    #     self.exemplars = dict()
    #     self.exemplars = {'herding_period': []}
    #     full_data = []
    #     for item in self.sess_rep_by_item:
    #         full_data.extend(self.sess_rep_by_item[item])
    #     seq, rep = zip(*full_data)
    #     seq = np.array(seq)
    #     rep = np.array(rep)
    #     self.herding(rep, 'herding_period', seq, self.m)

    def randomly_by_frequency(self, item_count):
        """
        This method randomly selects exemplars, and selects equivalent number of exemplar for each label.
        """
        self.exemplars = defaultdict(list)
        item_count = np.array(item_count)
        item_count = item_count * (self.m / item_count.sum())
        item_count = np.floor(item_count)
        item_count = np.int32(item_count)

        # total_num = 0
        # for item in sorted(self.sess_rep_by_item):
        #     seq = self.sess_rep_by_item[item]
        #     total_num += len(seq)
        # print('after sort number%d' % total_num)

        for item in self.sess_rep_by_item:
            seq = self.sess_rep_by_item[item]
            seq = np.array(seq)
            seq_num = len(seq)
            m = item_count[item - 1]
            if m > 0:
                selected_ids = np.random.choice(seq_num, min(m, seq_num), replace=False)
                self.exemplars[item] = [seq[i].tolist() for i in selected_ids]
        del self.sess_rep_by_item

    # def randomly_by_period(self):
    #     """
    #     This method randomly selects exemplars among all labels.
    #     """
    #     self.exemplars = dict()
    #     self.exemplars = {'random_period': []}
    #     for _ in range(self.m):
    #         random_item = random.choice(list(self.sess_rep_by_item.keys()))
    #         seq, _ = random.choice(self.sess_rep_by_item[random_item])
    #         seq = seq.tolist()
    #         self.exemplars['random_period'].append(seq)

    def save(self, period):
        """
        This method save the generated exemplars
        """
        if not os.path.isdir('exemplar'):
            os.makedirs(os.path.join('exemplar'))
        with open('exemplar/%sExemplarPeriod=%d.pickle' % (self.mode, period), mode='wb') as file:
            pickle.dump(self.exemplars, file)
        del self.exemplars


class ContinueLearningPlot:

    def __init__(self, args):
        self.args = args
        self.epochs = defaultdict(list)
        self.MRR20 = defaultdict(list)
        self.RECALL20 = defaultdict(list)
        self.MRR10 = defaultdict(list)
        self.RECALL10 = defaultdict(list)

        self.MRR20_test = []
        self.RECALL20_test = []
        self.MRR10_test = []
        self.RECALL10_test = []

    def add_valid(self, period, epoch, t_valid):

        self.epochs[period].append(epoch)
        self.MRR20[period].append(t_valid[0])
        self.RECALL20[period].append(t_valid[1])
        self.MRR10[period].append(t_valid[2])
        self.RECALL10[period].append(t_valid[3])

    def best_epoch(self, period, best_epoch):

        best_epoch_idx = self.epochs[period].index(best_epoch)
        self.epochs[period] = self.epochs[period][:best_epoch_idx + 1]
        self.MRR20[period] = self.MRR20[period][:best_epoch_idx + 1]
        self.RECALL20[period] = self.RECALL20[period][:best_epoch_idx + 1]
        self.MRR10[period] = self.MRR10[period][:best_epoch_idx + 1]
        self.RECALL10[period] = self.RECALL10[period][:best_epoch_idx + 1]

    def add_test(self, t_test):
        self.MRR20_test.append(t_test[0])
        self.RECALL20_test.append(t_test[1])
        self.MRR10_test.append(t_test[2])
        self.RECALL10_test.append(t_test[3])

    def plot(self):

        x_counter = defaultdict(list)
        last_max = [0]
        for period in sorted(self.epochs.keys()):
            new_seq = [epoch + last_max[-1] for epoch in self.epochs[period]]
            x_counter[period] = new_seq
            last_max.append(new_seq[-1])

        for i, period in enumerate(self.epochs.keys()):
            if i == 0:
                plt.plot(x_counter[period], self.MRR20[period], color='r', label='MRR@20')
                plt.plot(x_counter[period], self.MRR10[period], color='g', label='MRR@10')
                plt.plot(x_counter[period], self.RECALL20[period], color='b', label='RECALL@20')
                plt.plot(x_counter[period], self.RECALL10[period], color='y', label='RECALL@10')
            else:
                plt.plot(x_counter[period], self.MRR20[period], color='r')
                plt.plot(x_counter[period], self.MRR10[period], color='g')
                plt.plot(x_counter[period], self.RECALL20[period], color='b')
                plt.plot(x_counter[period], self.RECALL10[period], color='y')

        test_idx = last_max[1:]
        plt.scatter(test_idx, self.MRR20_test, color='r')
        for x, y in zip(test_idx, self.MRR20_test):
            plt.text(x, y, '%.4f' % y, ha='center', va='bottom', size='x-small')
        plt.scatter(test_idx, self.MRR10_test, color='g')
        for x, y in zip(test_idx, self.MRR10_test):
            plt.text(x, y, '%.4f' % y, ha='center', va='top', size='x-small')
        plt.scatter(test_idx, self.RECALL20_test, color='b')
        for x, y in zip(test_idx, self.RECALL20_test):
            plt.text(x, y, '%.4f' % y, ha='center', va='bottom', size='x-small')
        plt.scatter(test_idx, self.RECALL10_test, color='y')
        for x, y in zip(test_idx, self.RECALL10_test):
            plt.text(x, y, '%.4f' % y, ha='center', va='bottom', size='x-small')

        x_max = last_max[-1]
        last_max = [i + 1 for i in last_max]
        last_max = last_max[:-1]
        x_label = []
        p = 1
        l = 1
        for x in range(1, x_max + 1):
            if x not in last_max:
                x_label.append(None)
                # x_label.append(str(x - l + 1))
            else:
                l = x
                # x_label.append('%d\nperiod %d' % (x - l + 1, p))
                x_label.append('period %d' % p)
                p += 1

        plt.xticks(range(1, x_max + 1), x_label)
        plt.title('%s\ncontinue learning results\n%s exemplar_size_%d'
                  % (self.args.dataset, self.args.desc, self.args.exemplar_size))
        plt.legend()
        plt.show()

        i = 0
        while os.path.isfile('Coutinue_Learning_results%d.pdf' % i):
            i += 1
        plt.savefig('Coutinue_Learning_results%d.pdf' % i)
        plt.close()
