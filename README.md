# ADER: Adaptively Distilled Exemplar Replay towards Continual Learning for Session-based Recommendation

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Dataset and Pre-processing](#dataset)
- [License](#license)

## Background

## Requirements
Python 3.7, TensorFlow 2.1.0, and other common packages listed in `requirements.txt` or `requirements.yaml`.

Install required environment: `conda create env -f requirement.yaml`

Activate required environment `conda activate ader`

## Dataset and Pre-processing
Two widely used dataset are adopted:
### Dataset
- [DIGINETICA](http://cikm2016.cs.iupui.edu/cikm-cup): This dataset contains click-streams data on a e-commerce
site over a 5 months, and it is used for CIKM Cup 2016.
- [YOOCHOOSE](http://2015.recsyschallenge.com/challenge.html) :It is another dataset used by RecSys Challenge 2015  for predicting
click-streams on another e-commerce site over 6 months.
### Pre-process
The pre-processed data is uploaded in `data/DIGINETICA` and `data/YOOCHOOSE` folder. *Note:* the name of each sub-dataset is from 1 to 17, however, we name them from 0 to 16 in our paper; although the sub-dataset of YOOCHOOSE is named as week, it is actually splited by day. 

## License 
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
