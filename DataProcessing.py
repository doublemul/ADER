#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : 
# @File         : utility.py
# @Discription  :

def load_data(dataset_name):
    with open(dataset_name, 'r') as file:
        Session = {}
        for sample in file:
            sessId, itemId = sample.rstrip().split(' ')
            sessId = int(sessId)
            itemId = int(itemId)
            if sessId not in Session:
                Session[itemId] = []
            Session[itemId].append(itemId)
    splitted_sess, labels = session_splitting(list(Session.values()))
    return splitted_sess, labels


def session_splitting(sessions):
    '''
    Split sessions [[item0, item1, item2,...,item8, item9],[],[]]
    into two list:
    splitted_sess: [[item0, ..., tiem8], [item0,... ,item7]]
    labels: [item9, item8, ...]
    :param sessions:
    :return:
    '''
    splitted_sess = []
    labels = []
    for session in sessions:
        for i in range(1, len(session)):
            labels.append(session[-i])
            splitted_sess.append(session[:-i])
    return splitted_sess, labels
