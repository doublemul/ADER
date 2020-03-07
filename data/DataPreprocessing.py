#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       :
# @File         : DataPreprocessing.py
# @Discription  :
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from data.util import *


def read_data(dataset_path):
    filename_extension = dataset_path.split('/')[-1].split('.')[-1]

    if filename_extension == 'gz':
        user_map, item_map, reformed_data = read_gz(dataset_path)
    elif filename_extension == 'dat':
        user_map, item_map, reformed_data = read_dat(dataset_path)
    elif filename_extension == 'csv':
        user_map, item_map, reformed_data = read_csv(dataset_path)
    else:
        print("Error: new data file type !!!")

    print('Total number of users in dataset:', len(user_map.keys()))
    print('Total number of items in dataset:', len(item_map.keys()))
    print('Total number of actions in dataset:', len(reformed_data))
    print('Average number of actions per user:', len(reformed_data) / len(user_map.keys()))
    print('Average number of actions per item:', len(reformed_data) / len(item_map.keys()))

    return user_map, item_map, reformed_data


def short_remove(reformed_data, threshold_item, threshold_user):

    user_counter = defaultdict(lambda: 0)
    for [userId, _, _] in reformed_data:
        user_counter[userId] += 1
    removed_data = list(filter(lambda x: user_counter[x[0]] > 1, reformed_data))

    item_counter = defaultdict(lambda: 0)
    for [_, itemId, _] in removed_data:
        item_counter[itemId] += 1
    removed_data = list(filter(lambda x: item_counter[x[1]] > threshold_item, removed_data))

    user_counter = defaultdict(lambda: 0)
    for [userId, _, _] in removed_data:
        user_counter[userId] += 1
    removed_data = list(filter(lambda x: user_counter[x[0]] > threshold_user, removed_data))

    print('\n')
    print('Number of users after pre-processing:', len(set(map(lambda x: x[0], removed_data))))
    print('Number of items after pre-processing:', len(set(map(lambda x: x[1], removed_data))))
    print('Number of actions after pre-processing:', len(removed_data))
    print('Average number of actions per user:', len(removed_data) / len(set(map(lambda x: x[0], removed_data))))
    print('Average number of actions per item:', len(removed_data) / len(set(map(lambda x: x[1], removed_data))))

    user_start = dict()
    for [userId, _, time] in removed_data:
        user_start = generate_user_start_map(user_start, userId, time)

    return removed_data, user_start


def time_partition(reformed_data, user_start, interval='month', is_time_fraction = True):
    time_fraction = dict()
    if is_time_fraction:
        for [userId, itemId, time] in reformed_data:
            date = datetime.datetime.fromtimestamp(user_start[userId]).isoformat().split('T')[0]

            if interval == 'day':
                period = int(date.split('-')[0] + date.split('-')[1] + date.split('-')[2])
            elif interval == 'month':
                period = int(date.split('-')[0] + date.split('-')[1])
            elif interval == 'year':
                period = int(date.split('-')[0])
            else:
                print('invalid time interval')

            if period not in time_fraction:
                time_fraction[period] = []
            time_fraction[period].append([userId, itemId, time])
    else:
        max_time = max(map(lambda x: x[2], removed_data))
        date = datetime.datetime.fromtimestamp(max_time).isoformat().split('T')[0]
        period = int(date.split('-')[0] + date.split('-')[1] + date.split('-')[2])
        time_fraction[period] = removed_data

    return time_fraction


def generating_txt(time_part, user_start, test_interval='day'):
    for period in sorted(time_part.keys()):
        time_part[period].sort(key=lambda x: x[2])
        time_part[period].sort(key=lambda x: x[0])

    if test_interval == 'day':
        test_threshold = 86400
    elif test_interval == 'week':
        test_threshold = 86400 * 7
    elif test_interval == 'month':
        test_threshold = 86400 * 30

    last_time = get_last_time(list(sorted(time_part.keys())))
    for i, period in enumerate(sorted(time_part.keys()), start=1):
        with open('train_' + str(i) + '.txt', 'w') as file_train, \
                open('test_' + str(i) + '.txt', 'w') as file_test, \
                open('valid_' + str(i) + '.txt', 'w') as file_valid:
            for [userId, itemId, time] in time_part[period]:
                if user_start[userId] <= last_time[i-1] - test_threshold * 2:
                    file_train.write('%d %d\n' % (userId, itemId))
                elif user_start[userId] <= last_time[i-1] - test_threshold:
                    file_valid.write('%d %d\n' % (userId, itemId))
                elif user_start[userId] > last_time[i-1] - test_threshold:
                    file_test.write('%d %d\n' % (userId, itemId))


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', required=True)
    # parser.add_argument('--threshold', required=True)
    # args = parser.parse_args()
    # dataset_full_name = args.dataset
    # threshold = int(args.threshold)

    dataset_name = 'yoochoose-clicks.dat'
    dataset_name = 'train-item-views.csv'
    time_fraction = 'month'
    test_fraction = 'day'
    threshold_user = 1
    threshold_item = 4
    is_time_fraction = True

    print('Start preprocss ' + dataset_name + ':')

    os.chdir('datasets')
    user_map, item_map, reformed_data = read_data(dataset_name)

    removed_data, user_start = short_remove(reformed_data, threshold_item, threshold_user)

    time_part = time_partition(removed_data, user_start, interval=time_fraction, is_time_fraction=is_time_fraction)

    os.chdir(os.path.join('..', dataset_name.split('.')[0] + '_results'))
    generating_txt(time_part, user_start, test_interval=test_fraction)

    if is_time_fraction: plot_stat(time_part)

    print(dataset_name + ' finish!')
