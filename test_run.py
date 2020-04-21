#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : Project
# @Author       : Xiaoyu LIN
# @File         : grid_search_dig.py
# @Description  :

import subprocess
import time
import argparse

if __name__ == '__main__':

    for i in ['ContinueLearningNonexemplar',
              'HerdingByFrequency2000',
              'RandomByFrequency2000',
              'HerdingByFrequency5000',
              'RandomByFrequency5000',
              'HerdingByFrequency10000',
              'RandomByFrequency10000',
              'HerdingByFrequency20000',
              'RandomByFrequency20000'
              ]:
    # for i in ['RandomByFrequency10000']:
        cmd = 'python test.py --save_dir=%s ' % i

        p = subprocess.Popen(cmd, shell=True)
        p.wait()

    print('Run Done.')
