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


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def generate_name_Id_map(name, map):
    if name in map:
        Id = map[name]
    else:
        Id = len(map.keys())+1
        map[name] = Id
    return Id


def generate_user_start_map(user_start, userId, time):
    if userId in user_start:
        user_start[userId] = min(time, user_start[userId])
    else:
        user_start[userId] = time
    return user_start


def read_gz(dataset_path):
    user_map = {}
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
                userId = generate_name_Id_map(user, user_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([userId, itemId, time])

        elif dataset_name.split('_')[0] == 'steam':
            # Steam dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                sample = eval(sample)
                user = sample['username']
                item = sample['product_id']
                time = sample['date']
                time = int(datetime.datetime.strptime(time, "%Y-%m-%d").timestamp())
                userId = generate_name_Id_map(user, user_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([userId, itemId, time])
        else:
            print("Error: new gz data file!")
    return user_map, item_map, reformed_data


def read_dat(dataset_path):
    user_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path, 'r') as f:

        if dataset_name.split('-')[0] == 'ml':
            # MovieLens 1M dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                user = sample.split('::')[0]
                item = sample.split('::')[1]
                time = int(sample.split('::')[3])

                userId = generate_name_Id_map(user, user_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([userId, itemId, time])

        elif dataset_name.split('-')[0] == 'yoochoose':
            # YOOCHOOSE dataset
            for sample in tqdm.tqdm(f, desc='Loading data'):
                user = sample.split(',')[0]
                item = sample.split(',')[2]
                time = sample.split(',')[1]
                time = int(datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

                userId = generate_name_Id_map(user, user_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([userId, itemId, time])
        else:
            print("Error: new dat data file!")
    return user_map, item_map, reformed_data


def read_csv(dataset_path):
    user_map = {}
    item_map = {}
    reformed_data = []

    dataset_name = dataset_path.split('/')[-1]
    with open(dataset_path) as f:

        if dataset_name.split('-')[0] == 'ml':
            # MovieLens 20M dataset
            reader = csv.DictReader(f)
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                user = sample['userId']
                item = sample['movieId']
                time = sample['timestamp']

                userId = generate_name_Id_map(user, user_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([userId, itemId, time])

        elif dataset_name.split('-')[0] == 'train':
            # DIGINETICA dataset
            reader = csv.DictReader(f, delimiter=';')
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                user = sample['sessionId']
                item = sample['itemId']
                time = sample['eventdate']
                time = int(datetime.datetime.strptime(time, "%Y-%m-%d").timestamp())

                userId = generate_name_Id_map(user, user_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([userId, itemId, time])
        else:
            print("Error: new csv data file!")
    return user_map, item_map, reformed_data


def plot_stat(time_fraction):

    total_item, item_num, user_num, action_num = [], [], [], []
    total_item_set, user_num_set, item_num_set = set(), set(), set()

    length_counter = defaultdict(lambda: 0)
    session_length = []

    for peroid in sorted(time_fraction.keys()):
        for [userId, itemId, _] in time_fraction[peroid]:
            total_item_set.add(itemId)
            item_num_set.add(itemId)
            user_num_set.add(userId)
            length_counter[userId] += 1

        action_num.append(len(time_fraction[peroid]))
        user_num.append(len(user_num_set))
        item_num.append(len(item_num_set))
        total_item.append(len(total_item_set))
        item_num_set = set()
        user_num_set = set()

        session_length.extend(length_counter.values())
        length_counter = defaultdict(lambda: 0)

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


def get_last_time(peroids):
    if peroids[0] < 9999:
        # periods in year
        last_time = map(lambda x: int(datetime.datetime.strptime(str(x + 1), "%Y").timestamp()) - 1, peroids)
    elif peroids[0] < 999999:
        # periods in month
        new_peroids = []
        for peroid in peroids:
            year = peroid // 100
            month = peroid % 100
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            new_peroids.append(str(year) + '-' + str(month))
        last_time = map(lambda x: int(datetime.datetime.strptime(x, "%Y-%m").timestamp()) - 1, new_peroids)
    else:
        # periods in day
        new_peroids = []
        for peroid in peroids:
            day = peroid % 100
            yearmonth = peroid // 100
            month = yearmonth % 100
            year = yearmonth // 100
            new_peroids.append(str(year) + '-' + str(month) + '-' + str(day))
        last_time = map(lambda x: int(datetime.datetime.strptime(x, "%Y-%m-%d").timestamp()) + 86400-1, new_peroids)
    return list(last_time)


