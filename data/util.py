#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : SASRec
# @Author       : 
# @File         : util.py
# @Discription  :
import gzip
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import tqdm
import datetime
import numpy as np
import copy


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def generate_name_Id_map(name, map):
    """
    Given a name and map, return corresponding Id. If name not in map, generate a new Id.
    :param name: session or item name in dataset
    :param map: existing map, a dictionary: map[name]=Id
    :return: Id: allocated Id of the corresponding name
    """
    if name in map:
        Id = map[name]
    else:
        Id = len(map.keys())+1
        map[name] = Id
    return Id


def generate_sess_end_map(sess_end, sessId, time):
    """
    Generate map recording the session end time.
    :param sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    :param sessId:session Id of new action
    :param time:time of new action
    :return: sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    """
    if sessId in sess_end:
        sess_end[sessId] = max(time, sess_end[sessId])
    else:
        sess_end[sessId] = time
    return sess_end


def read_gz(dataset_path):
    """
    Read .gz type dataset file including Amazon reviews dataset and Steam dataset
    :param dataset_path: dataset path
    :return: sess_map: map[session name in row dataset]=session Id in system
    :return: item_map: map[item name in row dataset]=item Id in system
    :return: reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    sess_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with gzip.open(dataset_path, 'r') as f:

        if dataset_name.split('_')[0] == 'reviews':
            # Amazon reviews dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                sample = eval(sample)
                user = sample['reviewerID']
                item = sample['asin']
                time = sample['unixReviewTime']
                sessId = generate_name_Id_map(user, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])

        elif dataset_name.split('_')[0] == 'steam':
            # Steam dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                sample = eval(sample)
                user = sample['username']
                item = sample['product_id']
                time = sample['date']
                time = int(datetime.datetime.strptime(time, "%Y-%m-%d").timestamp())
                sessId = generate_name_Id_map(user, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
        else:
            print("Error: new gz data file!")
    return sess_map, item_map, reformed_data


def read_dat(dataset_path):
    """
    Read .dat type dataset file including MovieLens 1M dataset and Yoochoose dataset
    :param dataset_path: dataset path
    :return: sess_map: map[session name in row dataset]=session Id in system
    :return: item_map: map[item name in row dataset]=item Id in system
    :return: reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    sess_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path, 'r') as f:

        if dataset_name.split('-')[0] == 'ml':
            # MovieLens 1M dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                sess = sample.split('::')[0]
                item = sample.split('::')[1]
                time = int(sample.split('::')[3])

                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])

        elif dataset_name.split('-')[0] == 'yoochoose':
            # YOOCHOOSE dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                sess = sample.split(',')[0]
                item = sample.split(',')[2]
                time = sample.split(',')[1]
                time = int(datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
        else:
            print("Error: new dat data file!")
    return sess_map, item_map, reformed_data


def read_csv(dataset_path):
    """
    Read .csv type dataset file including MovieLens 20M dataset and DIGINETICA dataset
    :param dataset_path: dataset path
    :return: sess_map: map[session name in row dataset]=session Id in system
    :return: item_map: map[item name in row dataset]=item Id in system
    :return: reformed_data: a list: each element is a action, which is a list of [sessId, itemId, time]
    """
    sess_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path) as f:

        if dataset_name.split('-')[0] == 'ml':
            # MovieLens 20M dataset
            reader = csv.DictReader(f)
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['userId']
                item = sample['movieId']
                time = sample['timestamp']

                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])

        elif dataset_name.split('-')[0] == 'train':
            # DIGINETICA dataset
            ###############################
            # with sequence information
            reader = csv.DictReader(f, delimiter=';')
            timeframes = []
            for sample in reader:
                timeframes.append(int(sample['timeframe']))
            converter = 86400.00 / max(timeframes)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=';')
            for sample in tqdm.tqdm(reader, desc='Reformatting data'):
                sess = sample['sessionId']
                item = sample['itemId']
                date = sample['eventdate']
                timeframe = int(sample['timeframe'])
                time = int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()) + timeframe * converter
            ##############################
            # # without sequence information
            # reader = csv.DictReader(f, delimiter=';')
            # for sample in tqdm.tqdm(reader, desc='Loading data'):
            #     sess = sample['sessionId']
            #     item = sample['itemId']
            #     time = sample['eventdate']
            #     time = int(datetime.datetime.strptime(time, "%Y-%m-%d").timestamp())
            ##########################
                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
        else:
            print("Error: new csv data file!")
    return sess_map, item_map, reformed_data


def plot_stat(time_fraction):

    total_item, item_num, user_num, action_num, new_num, old_num = [], [], [], [], [], []
    total_item_set, user_num_set, item_num_set, old_item_set = set(), set(), set(), set()

    length_counter = defaultdict(lambda: 0)
    session_length = []
    for peroid in sorted(time_fraction.keys()):
        new, old = 0, 0
        for [userId, itemId, _] in time_fraction[peroid]:
            total_item_set.add(itemId)
            item_num_set.add(itemId)
            user_num_set.add(userId)
            length_counter[userId] += 1
            if itemId not in old_item_set:
                new += 1
            else:
                old += 1

        action_num.append(len(time_fraction[peroid]))
        user_num.append(len(user_num_set))
        item_num.append(len(item_num_set))
        total_item.append(len(total_item_set))
        item_num_set = set()
        user_num_set = set()

        session_length.extend(length_counter.values())
        length_counter = defaultdict(lambda: 0)

        old_item_set = copy.deepcopy(total_item_set)
        new_num.append(new)
        old_num.append(old)

    plt.figure()
    plt.title('Number of item vs time')
    width = 0.3
    bar1 = plt.bar(np.arange(len(time_fraction.keys())), item_num,
                   width, label='# item')
    for x, y in zip(np.arange(len(time_fraction.keys())), item_num):
        plt.text(x, y, '%d' % y, ha='center', va='bottom', rotation=30, size='x-small')
    bar2 = plt.bar(np.arange(len(time_fraction.keys())) + width, np.array(total_item)-np.array([0]+total_item[:-1]),
                   width, label='# new item')
    for x, y in zip(np.arange(len(time_fraction.keys())), np.array(total_item)-np.array([0]+total_item[:-1])):
        plt.text(x + width, y, '%d' % y, ha='center', va='bottom', rotation=30, size='x-small')
    plt.ylabel('# item or # new item')
    axes2 = plt.twinx()
    line1 = axes2.plot(range(len(time_fraction.keys())), total_item, color='green', marker='o',
                       label='# accumulate item')
    figures = [bar1] + [bar2] + line1
    labs = [l.get_label() for l in figures]
    axes2.legend(figures, labs, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    axes2.set_ylabel('# accumulate item')
    plt.xticks(range(len(time_fraction.keys())), sorted(time_fraction.keys()))
    plt.savefig('sta_item.pdf')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('Average session length vs time')
    plt.bar(np.arange(len(time_fraction.keys())), np.array(action_num)/np.array(user_num))
    for x, y in zip(np.arange(len(time_fraction.keys())), np.array(action_num)/np.array(user_num)):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    plt.xticks(range(len(time_fraction.keys())), sorted(time_fraction.keys()))
    plt.savefig('sta_length.pdf')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('Session length histogram')
    plt.hist(session_length, bins=29)
    plt.xlim((0, 30))
    plt.xticks(np.arange(30), rotation=30)
    plt.ylabel('# session')
    plt.savefig('sta_hist.pdf')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('Number of actions vs time')
    width = 0.2
    bar1 = plt.bar(np.arange(len(time_fraction.keys())), user_num, width, label='# session')
    for x, y in zip(np.arange(len(time_fraction.keys())), user_num):
        plt.text(x, y, '%d' % y, ha='center', va='center', rotation=90, size='x-small')
    bar2 = plt.bar(np.arange(len(time_fraction.keys())) - width, item_num, width, label='# class')
    for x, y in zip(np.arange(len(time_fraction.keys())), item_num):
        plt.text(x - width, y, '%d' % y, ha='center', va='center', rotation=90, size='x-small')
    plt.ylabel('# session or # class')
    axes2 = plt.twinx()
    bar3 = axes2.bar(np.arange(len(time_fraction.keys())) + width, action_num, width, label='# action',
                     color='green')
    for x, y in zip(np.arange(len(time_fraction.keys())), action_num):
        axes2.text(x + width, y, '%d' % y, ha='center', va='center', rotation=90, size='x-small')
    figures = [bar1] + [bar2] + [bar3]
    labs = [l.get_label() for l in figures]
    axes2.legend(figures, labs)
    axes2.set_ylabel('# action')
    plt.xticks(range(len(time_fraction.keys())), sorted(time_fraction.keys()))
    plt.savefig('sta_action.pdf')
    plt.show()
    plt.close()

    plt.figure()
    plt.title('Number of actions on new item vs old item')
    width = 0.3
    plt.bar(np.arange(len(time_fraction.keys())), old_num, width, label='old item')
    for x, y in zip(np.arange(len(time_fraction.keys())), old_num):
        plt.text(x, y, '%d' % y, ha='center', va='center', rotation=90, size='x-small')
    plt.bar(np.arange(len(time_fraction.keys())) + width, new_num, width, label='new item')
    for x, y in zip(np.arange(len(time_fraction.keys())), new_num):
        plt.text(x + width, y, '%d' % y, ha='center', va='center', rotation=90, size='x-small')
    plt.ylabel('# actions')
    plt.xticks(range(len(time_fraction.keys())), sorted(time_fraction.keys()))
    plt.legend()
    plt.savefig('sta_new_old.pdf')
    plt.show()
    plt.close()


def get_last_time(periods):
    """
    Find the last second of the period in unix form
    :param periods: YYYY if year, YYYYMM if month, YYYYMMDD if day
    :return: the last second of the period in unix form
    """
    if periods[0] < 9999:
        # periods in year
        last_time = map(lambda x: int(datetime.datetime.strptime(str(x + 1), "%Y").timestamp()) - 1, periods)
    elif periods[0] < 999999:
        # periods in month
        new_periods = []
        for period in periods:
            year = period // 100
            month = period % 100
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            new_periods.append(str(year) + '-' + str(month))
        last_time = map(lambda x: int(datetime.datetime.strptime(x, "%Y-%m").timestamp()) - 1, new_periods)
    else:
        # periods in day
        new_periods = []
        for period in periods:
            day = period % 100
            yearmonth = period // 100
            month = yearmonth % 100
            year = yearmonth // 100
            new_periods.append(str(year) + '-' + str(month) + '-' + str(day))
        last_time = map(lambda x: int(datetime.datetime.strptime(x, "%Y-%m-%d").timestamp()) + 86400-1, new_periods)
    return list(last_time)


