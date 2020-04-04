#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : temp.py
# @Description  :


class ContinueLearningPlot:
    def __init__(self, args):
        self.args = args
        self.periods = []
        self.epochs = []
        self.MRR20 = []
        self.RECALL20 = []
        self.MRR10 = []
        self.RECALL10 = []

    def add(self, period, epoch, t_test):
        if period > 1 and epoch < self.epochs[-1]:
            del self.periods[-1]
            del self.epochs[-1]
            del self.MRR20[-1]
            del self.RECALL20[-1]
            del self.MRR10[-1]
            del self.RECALL10[-1]

        self.periods.append(period)
        self.epochs.append(epoch)
        self.MRR20.append(t_test[0])
        self.RECALL20.append(t_test[1])
        self.MRR10.append(t_test[2])
        self.RECALL10.append(t_test[3])

    def plot(self, logs):
        x_label = list(map(lambda period, epoch: 'P%dE%d' % (period, epoch), self.periods, self.epochs))

        plt.figure()
        plt.plot(range(len(self.MRR20)), self.MRR20, label='MRR@20', color='b')
        plt.plot(range(len(self.MRR20)), self.MRR10, label='MRR@10', color='g')
        plt.plot(range(len(self.MRR20)), self.RECALL20, label='RECALL20', color='c')
        plt.plot(range(len(self.MRR20)), self.RECALL10, label='RECALL10', color='y')

        if self.args.dataset == 'DIGINETICA':
            NARM_RECALL20 = 0.4970
            NARM_RECALL10 = 0.3362
        elif self.args.dataset == 'YOOCHOOSE':
            NARM_RECALL20 = 0.6973
            NARM_RECALL10 = 0.5870
        plt.hlines(NARM_RECALL20, 0, len(self.MRR20)-1, label='NARM_RECALL20', color='r')
        plt.hlines(NARM_RECALL10, 0, len(self.MRR20)-1, label='NARM_RECALL10', color='m')

        plt.xticks(range(len(self.MRR20)), x_label, rotation=90, size='small')
        plt.title('Continue learning test results')
        plt.legend()

        i = 0
        while os.path.isfile('Coutinue_Learning_result%d.pdf' % i):
            i += 1
        plt.savefig('Coutinue_Learning_result%d.pdf' % i)
        plt.close()
        logs.write('Coutinue_Learning_result%d.pdf\n' % i)
