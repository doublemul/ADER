#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : util.py
# @Description  :
import random
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def load_data(dataset_name, item_set, is_train=True, remove_item=False, logs=None):
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
    if not is_train:
        info = 'original total number of action: %d, removed number of action: %d.' % (total_num, removed_num)
        logs.write(info + '\n')
    # info = 'Total number of action: %d, removed number of action: %d' % (total_num, removed_num)
    # with open('data_args.txt', 'w') as f:
    #     f.write(info + '\n')
    sessions = list(Sessions.values())
    del Sessions
    return sessions


def random_neg(max_item, ts):
    neg = np.random.randint(1, max_item)
    while neg in ts:
        neg = np.random.randint(1, max_item)
    return neg


def sample(user_train, max_item, maxlen):
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
            neg[idx] = random_neg(max_item, ts)
        nxt = itemId
        idx -= 1
        if idx == -1:
            break

    return seq, pos, neg


def sampler(user_train, max_item, batch_size, maxlen):
    SEED = random.randint(0, 2e9)
    random.seed(SEED)

    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample(user_train, max_item, maxlen))

    return zip(*one_batch)


def evaluate(inputs, max_item, model, args, sess, mode):
    MRR_20 = 0.0
    RECALL_20 = 0.0
    ranks = []

    sess_num = len(inputs)

    batch_num = int(sess_num / args.test_batch)

    for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b', desc=mode):
    # for _ in range(batch_num):
        sess_indices = random.sample(range(sess_num), args.test_batch)
        ground_truth = []
        seq = []
        truth = []
        for i, sess_index in enumerate(sess_indices):
            length = len(inputs[sess_index])
            if length <= 2:
                seq.append([inputs[sess_index][0]])
            elif length > (args.maxlen + 1):
                seq.append(inputs[sess_index][length-args.maxlen-1:-1])
            else:
                seq.append(inputs[sess_index][:-1])
            ground_truth.append(copy.deepcopy(inputs[sess_index][-1]))
            while len(seq[i]) < args.maxlen:
                seq[i].insert(0, 0)
            pos = copy.deepcopy(seq[i])
            pos.append(copy.deepcopy(inputs[sess_index][-1]))
            pos = pos[1:]
            truth.append(pos)

        if args.neg_sample:
            for i in range(len(seq)):
                item_set = set(seq[i])
                item_set.add(0)
                item_idx = [ground_truth[i]]
                for _ in range(args.neg_sample):
                    t = np.random.randint(1, max_item + 1)
                    while t in item_set:
                        t = np.random.randint(1, max_item + 1)
                    item_idx.append(t)
                predictions = -model.predict_neg(sess, [seq[i]], item_idx)
                predictions = predictions[0]
                ranks.append(predictions.argsort().argsort()[0])
        else:
            predictions = -model.predict(sess, seq, range(1, max_item+1))
            for i, prediction in enumerate(predictions):
                ranks.append(prediction.argsort().argsort()[ground_truth[i]-1])

            # predictions = -model.predict_seq(sess, seq, range(1, max_item+1))
            # predictions = predictions.argsort(axis=-1).argsort(axis=-1)
            # predictions = predictions.reshape(-1, np.shape(predictions)[-1])
            # truth = np.array(truth).reshape(-1)
            # ranks = predictions[np.arange(np.shape(predictions)[0]), truth-1]
            # ranks = ranks[np.where(truth != 0)]

    valid_user = len(ranks)
    valid_ranks_20 = list(filter(lambda x: x < 20, ranks))
    valid_ranks_10 = list(filter(lambda x: x < 10, ranks))
    RECALL_20 = len(valid_ranks_20)
    MRR_20 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_20))
    RECALL_10 = len(valid_ranks_10)
    MRR_10 = sum(map(lambda x: 1.0 / (x + 1), valid_ranks_10))

    return MRR_20 / valid_user, RECALL_20 / valid_user, MRR_10 / valid_user, RECALL_10 / valid_user
