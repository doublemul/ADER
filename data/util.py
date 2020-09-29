#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : util.py

import csv
import tqdm
import datetime


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
        """ YOOCHOOSE
        """
        for sample in tqdm.tqdm(f, desc='Loading data'):
            sess = sample.split(',')[0]
            item = sample.split(',')[2]
            time = sample.split(',')[1]
            time = int(datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp())

            sessId = generate_name_Id_map(sess, sess_map)
            itemId = generate_name_Id_map(item, item_map)
            reformed_data.append([sessId, itemId, time])

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

        if dataset_name.split('-')[0] == 'train':
            """ DIGINETICA
            """
            # with sequence information
            reader = csv.DictReader(f, delimiter=';')
            timeframes = []
            for sample in reader:
                timeframes.append(int(sample['timeframe']))
            converter = 86400.00 / max(timeframes)
            f.seek(0)
            reader = csv.DictReader(f, delimiter=';')
            # load data
            for sample in tqdm.tqdm(reader, desc='Loading data'):
                sess = sample['sessionId']
                item = sample['itemId']
                date = sample['eventdate']
                timeframe = int(sample['timeframe'])
                if date:
                    time = int(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp()) + timeframe * converter
                else:
                    continue
                sessId = generate_name_Id_map(sess, sess_map)
                itemId = generate_name_Id_map(item, item_map)
                reformed_data.append([sessId, itemId, time])
        else:
            print("Error: new csv data file!")
    return sess_map, item_map, reformed_data
