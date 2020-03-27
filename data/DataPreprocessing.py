#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : DataPreprocessing.py
# @Description  :
import argparse
import os
from datetime import datetime
from data.util import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_data(dataset_path):
    """
    Load data from raw dataset.
    :param dataset_path: the full name of dataset including extension name
    :return sess_map: map from raw data session name to session Id, a dictionary sess_map[sess_name]=sessId
    :return item_map: map from raw data item name to item Id, a dictionary item_map[item_name]=itemId
    :return reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    # load data according to file extension name
    filename_extension = dataset_path.split('/')[-1].split('.')[-1]
    if filename_extension == 'gz':
        sess_map, item_map, reformed_data = read_gz(dataset_path)
    elif filename_extension == 'dat':
        sess_map, item_map, reformed_data = read_dat(dataset_path)
    elif filename_extension == 'csv':
        sess_map, item_map, reformed_data = read_csv(dataset_path)
    else:
        print("Error: new data file type !!!")

    # print raw dataset information
    print('Total number of users in dataset:', len(sess_map.keys()))
    print('Total number of items in dataset:', len(item_map.keys()))
    print('Total number of actions in dataset:', len(reformed_data))
    print('Average number of actions per user:', len(reformed_data) / len(sess_map.keys()))
    print('Average number of actions per item:', len(reformed_data) / len(item_map.keys()))

    return sess_map, item_map, reformed_data


def short_remove(reformed_data, args):
    """
    Remove data according to threshold
    :param reformed_data: loaded data, a list: each element is a action, which is a list of [sessId, itemId, time]
    :param args.threshold_item: minimum number of appearance time of item -1
    :param args.threshold_sess: minimum length of session -1
    :return removed_data: result data after removing
    :return sess_end: a map recording session end time, a dictionary sess_end[sessId]=end_time
    """
    # remove session whose length is 1
    sess_counter = defaultdict(lambda: 0)
    for [userId, _, _] in reformed_data:
        sess_counter[userId] += 1
    removed_data = list(filter(lambda x: sess_counter[x[0]] > 1, reformed_data))

    # remove item which appear less or equal to threshold_item
    item_counter = defaultdict(lambda: 0)
    for [_, itemId, _] in removed_data:
        item_counter[itemId] += 1
    removed_data = list(filter(lambda x: item_counter[x[1]] > args.threshold_item, removed_data))

    # remove session whose length less or equal to threshold_sess
    sess_counter = defaultdict(lambda: 0)
    for [userId, _, _] in removed_data:
        sess_counter[userId] += 1
    removed_data = list(filter(lambda x: sess_counter[x[0]] > args.threshold_sess, removed_data))

    # print information of removed data
    print('Number of users after pre-processing:', len(set(map(lambda x: x[0], removed_data))))
    print('Number of items after pre-processing:', len(set(map(lambda x: x[1], removed_data))))
    print('Number of actions after pre-processing:', len(removed_data))
    print('Average number of actions per user:', len(removed_data) / len(set(map(lambda x: x[0], removed_data))))
    print('Average number of actions per item:', len(removed_data) / len(set(map(lambda x: x[1], removed_data))))

    # record session end time
    sess_end = dict()
    for [userId, _, time] in removed_data:
        sess_end = generate_sess_end_map(sess_end, userId, time)

    return removed_data, sess_end


def time_partition(removed_data, session_end, args):
    """
    Partition data according to time periods
    :param removed_data: input data, a list: each element is a action, which is a list of [sessId, itemId, time]
    :param session_end: a dictionary recording session end time, session_end[sessId]=end_time
    :param args.time_fraction: time interval for partition
    :param args.is_time_fraction: boolean, weather partition or not
    :return: time_fraction: a dictionary, the keys are different time periods,
    value is a list of actions in that time period
    """
    time_fraction = dict()

    if args.is_time_fraction:
        for [sessId, itemId, time] in removed_data:
            date = datetime.datetime.fromtimestamp(session_end[sessId]).isoformat().split('T')[0]
            # find period of each action
            if args.time_fraction == 'day':
                period = int(date.split('-')[0] + date.split('-')[1] + date.split('-')[2])
            elif args.time_fraction == 'month':
                period = int(date.split('-')[0] + date.split('-')[1])
            elif args.time_fraction == 'year':
                period = int(date.split('-')[0])
            else:
                print('invalid time interval')
            # generate time period for dictionary keys
            if period not in time_fraction:
                time_fraction[period] = []
            # partition data according to period
            time_fraction[period].append([sessId, itemId, time])

        # combine first and second time fraction in DIGINETICA and remove the last one
        if os.getcwd().split('/')[-1] == 'DIGINETICA' and args.time_fraction == 'month':
            periods = sorted(time_fraction.keys())
            time_fraction[periods[1]].extend(time_fraction[periods[0]])
            del time_fraction[periods[-1]]
            del time_fraction[periods[0]]

    else:
        if args.yoochoose_select and os.getcwd().split('/')[-1] == 'YOOCHOOSE':
            threshold = np.percentile(list(map(lambda x: x[2], removed_data)), (1-args.yoochoose_select)*100)
            print(threshold)
            removed_data = list(filter(lambda x: x[2] > threshold, removed_data))
        # if not partition, put all actions in the last period
        max_time = max(map(lambda x: x[2], removed_data))
        date = datetime.datetime.fromtimestamp(max_time).isoformat().split('T')[0]
        period = int(date.split('-')[0] + date.split('-')[1] + date.split('-')[2])
        time_fraction[period] = removed_data

    return time_fraction


def generating_txt(time_fraction, sess_end, args):
    """
    Generate final txt file
    :param time_fraction: input data, a dictionary, the keys are different time periods,
    value is a list of actions in that time period
    :param sess_end: session end time map, sess_map[sessId]=end_time
    :param args.test_fraction: time interval for test and valid sets
    """
    # sort action according to time sequence and session
    for period in sorted(time_fraction.keys()):
        time_fraction[period].sort(key=lambda x: x[2])
        time_fraction[period].sort(key=lambda x: x[0])

    if args.test_fraction == 'day':
        test_threshold = 86400
    elif args.test_fraction == 'week':
        test_threshold = 86400 * 7
    elif args.test_fraction == 'month':
        test_threshold = 86400 * 31
    # get the last unix time of each period
    last_time = get_last_time(list(sorted(time_fraction.keys())))

    if args.is_time_fraction:
        for i, period in enumerate(sorted(time_fraction.keys()), start=1):
            with open('train_' + str(i) + '.txt', 'w') as file_train, \
                    open('test_' + str(i) + '.txt', 'w') as file_test, \
                    open('valid_' + str(i) + '.txt', 'w') as file_valid:
                for [userId, itemId, time] in time_fraction[period]:
                    if sess_end[userId] <= last_time[i - 1] - test_threshold * 2:
                        file_train.write('%d %d\n' % (userId, itemId))
                    elif sess_end[userId] <= last_time[i - 1] - test_threshold * 1:
                        file_valid.write('%d %d\n' % (userId, itemId))
                    elif sess_end[userId] > last_time[i - 1] - test_threshold:
                        file_test.write('%d %d\n' % (userId, itemId))
    else:
        with open('train_0.txt', 'w') as file_train, \
                open('test_0.txt', 'w') as file_test, \
                open('valid_0.txt', 'w') as file_valid:
            for [userId, itemId, time] in time_fraction[period]:
                if sess_end[userId] <= last_time[0] - test_threshold * 2:
                    file_train.write('%d %d\n' % (userId, itemId))
                elif sess_end[userId] <= last_time[0] - test_threshold * 1:
                    file_valid.write('%d %d\n' % (userId, itemId))
                elif sess_end[userId] > last_time[0] - test_threshold:
                    file_test.write('%d %d\n' % (userId, itemId))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='train-item-views.csv', type=str)  # 'yoochoose-clicks.dat'
    parser.add_argument('--time_fraction', default='month', type=str)
    parser.add_argument('--test_fraction', default='day', type=str)
    parser.add_argument('--threshold_sess', default=1, type=int)
    parser.add_argument('--threshold_item', default=4, type=int)
    parser.add_argument('--is_time_fraction', default=False, type=str2bool)
    parser.add_argument('--yoochoose_select', default=0.25, type=float)
    args = parser.parse_args()

    print('Start preprocess ' + args.dataset + ':')

    # load data and get the session and item lookup table
    os.chdir('dataset')
    sess_map, item_map, reformed_data = read_data(args.dataset)

    if args.dataset.split('.')[0] == 'yoochoose-clicks':
        if not os.path.isdir(os.path.join('..', 'YOOCHOOSE')):
            os.makedirs(os.path.join('..', 'YOOCHOOSE'))
        os.chdir(os.path.join('..', 'YOOCHOOSE'))
    elif args.dataset.split('.')[0] == 'train-item-views':
        if not os.path.isdir(os.path.join('..', 'DIGINETICA')):
            os.makedirs(os.path.join('..', 'DIGINETICA'))
        os.chdir(os.path.join('..', 'DIGINETICA'))

    # remove data according to occurrences time
    removed_data, sess_end = short_remove(reformed_data, args)

    # partition data according to time periods
    time_fraction = time_partition(removed_data, sess_end, args)
    # generate final txt file
    generating_txt(time_fraction, sess_end, args)

    if args.is_time_fraction: plot_stat(time_fraction)

    print(args.dataset + ' finish!')
