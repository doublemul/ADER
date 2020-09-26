#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
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
    :return: Id: allocated new Id of the corresponding name
    """
    if name in map:
        Id = map[name]
    else:
        Id = len(map.keys()) + 1
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


def generate_sess_strat_map(sess_end, sessId, time):
    """
    Generate map recording the session end time.
    :param sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    :param sessId:session Id of new action
    :param time:time of new action
    :return: sess_end: the map recording session end time, a dictionary see_end[sessId]=end_time
    """
    if sessId in sess_end:
        sess_end[sessId] = min(time, sess_end[sessId])
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
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['sessionId']
                item = sample['itemId']
                date = sample['eventdate']
                timeframe = int(sample['timeframe'])
                if date:
                    time = int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()) + timeframe * converter
                else:
                    continue
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
