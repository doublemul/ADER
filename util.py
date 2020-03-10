#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : util.py
# @Description  :
import random
import copy
import numpy as np
from collections import defaultdict


def load_data(dataset_name, item_set, is_train=True, remove_item=False):
    # Load the dataset
    Sessions = defaultdict(list)
    with open('data/%s.txt' % dataset_name, 'r') as f:
        for line in f:
            sessId, itemId = line.rstrip().split(' ')
            sessId = int(sessId)
            itemId = int(itemId)

            # remove new items in test that not appear in train
            if is_train or (not remove_item):
                item_set.add(itemId)
            else:
                if itemId not in item_set:
                    continue
            Sessions[sessId].append(itemId)
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


def evaluate(sessions, max_item, model, args, sess):

    MRR = 0.0
    RECALL = 0.0
    valid_user = 0.0

    sess_num = len(sessions)

    if sess_num > 10000:
        sess_indices = random.sample(range(sess_num), 10000)
    else:
        sess_indices = range(sess_num)

    for sess_index in sess_indices:
        session = copy.deepcopy(sessions[sess_index])
        while len(session) <= args.maxlen:
            session.insert(0, 0)

        ground_truth = session[-1]
        seq = session[:args.maxlen]

        predictions = -model.predict(sess, [seq], range(1, max_item + 1))
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[ground_truth - 1]

        valid_user += 1

        if rank < 20:
            MRR += 1.0 / (rank + 1)
            RECALL += 1
        if valid_user % 1000 == 0:
            print('.', end='')

    return MRR / valid_user, RECALL / valid_user
