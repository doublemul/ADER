# ADER: Adaptively Distilled Exemplar Replay towards Continual Learning for Session-based Recommendation

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Dataset and Pre-processing](#dataset)
- [License](#license)

## Background
The implemention of self-attenvie recommender is modified based on [SASRec](https://github.com/kang205/SASRec)

## Requirements
Python 3.7, TensorFlow 2.1.0, and other common packages listed in `requirements.txt` or `requirements.yaml`.<br/>
Install required environment: `conda create env -f requirement.yaml`<br/>
Activate required environment `conda activate ader`

## Dataset and Pre-processing
Two widely used dataset are adopted:
### Dataset
- [DIGINETICA](http://cikm2016.cs.iupui.edu/cikm-cup): This dataset contains click-streams data on a e-commerce
site over a 5 months, and it is used for CIKM Cup 2016.
- [YOOCHOOSE](http://2015.recsyschallenge.com/challenge.html) :It is another dataset used by RecSys Challenge 2015  for predicting
click-streams on another e-commerce site over 6 months.
### Pre-process
The pre-processed data used in our paper is uploaded in `data/DIGINETICA` and `data/YOOCHOOSE` folder.<br/>
**Note:** The name of each sub-dataset is from 1 to 17, however, we name them from 0 to 16 in our paper.<br/>
**Note:** Although the sub-dataset of YOOCHOOSE is named as week, it is actually splited by day.
### Run data pre-process
Download `train-item-views.csv` or `yoochoose-clicks.dat` into folder `data\dataset`.<br/>
For DIGINETICA: run `python DataPreprocessing.py`<br/>
For YOOCHOOSE: run `python DataPreprocessing.py --dataset=yoochoose-clicks.dat --test_fraction=day`<br/>

## Model Training
To train our model on DIGINETICA: `python main.py`<br/>
To train our model on YOOCHOOSE: `python main.py --dataset=YOOCHOOSE --lambda_=1.0`

## Results
ADER significantly outperforms other methods. More importantly, it even outperforms Joint. This result empirically
reveals that ADER is a promising solution for the continual recommendation setting by effectively preserving user
preference patterns learned before.
![results](results.png)


## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
