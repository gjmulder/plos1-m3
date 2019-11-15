#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import numpy as np
#import pandas as pd
from datetime import date

from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from os import environ
import traceback
from math import log
#from itertools import repeat
#from datetime import timedelta
from json import loads

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)
    
########################################################################################################

if "VERSION" in environ:    
    version = environ.get("VERSION")
    logger.info("Using version : %s" % version)
    
    use_cluster = True
else:
    version = "test"
    logger.warning("VERSION not set, using: %s" % version)
    
    use_cluster = False

if "DATASET" in environ:    
    dataset_name = environ.get("DATASET")
    logger.info("Using dataset : %s" % dataset_name)
    
    use_cluster = True
else:
    dataset_name = "test"
    logger.warning("DATASET not set, using: %s" % version)
    
num_eval_samples = 1
freq="M"
prediction_length = 18
    
########################################################################################################

def load_plos_m3_data():
    data = {}
    for dataset in ["train", "test"]:
        data[dataset] = []
        data["%s-nocat" % dataset] = []
        with open("./m3_monthly/%s/data.json" % dataset) as fp:
            for line in fp:
               ts_data = loads(line)
               data[dataset].append(ts_data)
               ts_data_copy = ts_data.copy()
               del(ts_data_copy['feat_static_cat'])
               data["%s-nocat" % dataset].append(ts_data_copy)
    return data

def forecast(data, cfg):
    logger.info("Params: %s " % cfg)
    
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        gluon_train = ListDataset(data['train-nocat'], freq=freq)
    else:
        gluon_train = ListDataset(data['train'], freq=freq)

#    trainer=Trainer(
#        mx.Context("gpu"),
#        epochs=5,
#    )
    
    trainer=Trainer(
        mx.Context("gpu"),
        epochs=cfg['trainer']['max_epochs'],
        num_batches_per_epoch=cfg['trainer']['num_batches_per_epoch'],
        batch_size=cfg['trainer']['batch_size'],
        patience=cfg['trainer']['patience'],
        
        learning_rate=cfg['trainer']['learning_rate'],
        learning_rate_decay_factor=cfg['trainer']['learning_rate_decay_factor'],
        minimum_learning_rate=cfg['trainer']['minimum_learning_rate'],
        weight_decay=cfg['trainer']['weight_decay'],
    )
    
    if cfg['model']['type'] == 'SimpleFeedForwardEstimator':
        estimator = SimpleFeedForwardEstimator(
            freq=freq,
            prediction_length=prediction_length, 
            num_hidden_dimensions = cfg['model']['num_hidden_dimensions'],
            num_parallel_samples=1,
            trainer=trainer)

    if cfg['model']['type'] == 'DeepFactorEstimator': 
         estimator = DeepFactorEstimator(
            freq=freq,
            prediction_length=prediction_length,
            num_hidden_global=cfg['model']['num_hidden_global'], 
            num_layers_global=cfg['model']['num_layers_global'], 
            num_factors=cfg['model']['num_factors'], 
            num_hidden_local=cfg['model']['num_hidden_local'], 
            num_layers_local=cfg['model']['num_layers_local'],
            trainer=trainer)
         
    if cfg['model']['type'] == 'DeepAREstimator':            
        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,        
            num_cells=cfg['model']['num_cells'],
            num_layers=cfg['model']['num_layers'],        
            dropout_rate=cfg['model']['dar_dropout_rate'],
#            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=[len(data['train']), 6],
            num_parallel_samples=1,
            trainer=trainer)
        
    if cfg['model']['type'] == 'TransformerEstimator': 
         estimator = TransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
#            model_dim=cfg['model']['model_dim'], 
            model_dim=cfg['model']['model_dim_heads'][0], 
            inner_ff_dim_scale=cfg['model']['inner_ff_dim_scale'],
            pre_seq=cfg['model']['pre_seq'], 
            post_seq=cfg['model']['post_seq'], 
            act_type=cfg['model']['act_type'], 
#            num_heads=cfg['model']['num_heads'], 
            num_heads=cfg['model']['model_dim_heads'][1], 
            dropout_rate=cfg['model']['trans_dropout_rate'],
#            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=[len(data['train']), 6],
            num_parallel_samples=1,
            trainer=trainer)

    logger.info("Fitting: %s" % cfg['model']['type'])
    model = estimator.train(gluon_train)
    
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        gluon_test = ListDataset(data['test-nocat'], freq=freq)
    else:
        gluon_test = ListDataset(data['test'], freq=freq)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=gluon_test,
        predictor=model,
        num_eval_samples=1,
    )
    
    agg_metrics, item_metrics = Evaluator()(
        ts_it, forecast_it, num_series=len(data['test'])
    )
    
    mase = agg_metrics['MASE']
    logger.info("MASE : %.3f" % mase)
    return mase

def gluon_fcast(cfg):        
    try:
        err = forecast(data, cfg)
    except Exception as e:
        exc_str = '\n%s' % traceback.format_exc()
        logger.error(exc_str)
        return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'exception': exc_str, 'build_url' : environ.get("BUILD_URL")}
        
    logger.info("MASE: %.3f" % err)
    return {'loss': err, 'status': STATUS_OK, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL")}

def call_hyperopt():
    dropout_rate = [0.05, 0.15]
    space = {
        'trainer' : {
            'max_epochs'                 : hp.choice('max_epochs', [125, 250, 500, 1000, 2000, 4000]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [5, 10, 20, 40, 80]),
            'batch_size'                 : hp.choice('batch_size', [25, 50, 100, 200, 400]),
            'patience'                   : hp.choice('patience', [20, 40, 80]),
            
            'learning_rate'              : hp.uniform('learning_rate', 1e-04, 1e-02),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.4, 0.9),
            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', log(1e-06), log(0.5e-04)),
            'weight_decay'               : hp.uniform('weight_decay', 00.5e-08, 10.0e-08),
        },
        'model' : hp.choice('model', [
            {
                'type'                       : 'SimpleFeedForwardEstimator',
                'num_hidden_dimensions'      : hp.choice('num_hidden_dimensions', [[5], [10], [25], [50], [100],
                                                                                   [5, 5], [10, 10], [25, 25], [50, 25], [50, 50], [100, 50], [100, 100],
                                                                                   [100, 50, 50]])
            },
            {
                'type'                       : 'DeepFactorEstimator',
                'num_hidden_global'          : hp.choice('num_hidden_global', [5, 10, 20, 40, 80, 160, 320]),
                'num_layers_global'          : hp.choice('num_layers_global', [1, 2, 3]),
                'num_factors'                : hp.choice('num_factors', [5, 10, 20]),
                'num_hidden_local'           : hp.choice('num_hidden_local', [2, 5, 10]),
                'num_layers_local'           : hp.choice('num_layers_local', [1, 2, 3]),
            },
            {
                'type'                       : 'DeepAREstimator',
                'num_cells'                  : hp.choice('num_cells', [5, 10, 20, 40, 80, 160, 320]),
                'num_layers'                 : hp.choice('num_layers', [1, 3, 5, 7]),
                
                'dar_dropout_rate'           : hp.uniform('dar_dropout_rate', dropout_rate[0], dropout_rate[1]),
            },
            {
                'type'                       : 'TransformerEstimator',
                'model_dim_heads'            : hp.choice('model_dim_heads', [[2, 2], [4, 2], [8, 2], [16, 2], [32, 2], [64, 2],
                                                                             [4, 4], [8, 4], [16, 4], [32, 4], [64, 4],
                                                                             [8, 8], [16, 8], [32, 8], [64, 8],
                                                                             [16, 16], [32, 16], [64, 16]]),
                'inner_ff_dim_scale'         : hp.choice('inner_ff_dim_scale', [2, 3, 4, 5]),
                'pre_seq'                    : hp.choice('pre_seq', ['dn']),
                'post_seq'                   : hp.choice('post_seq', ['drn']),
                'act_type'                   : hp.choice('act_type', ['softrelu']),               
                'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate[0], dropout_rate[1]),
            },
        ])
    }
                
            
    # Search MongoDB for best trial for exp_key:
    # echo 'db.jobs.find({"exp_key" : "XXX", "result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo --host heika m4_daily
    # echo 'db.jobs.remove({"exp_key" : "XXX", "result.status" : "new"})' | mongo --host heika
    if use_cluster:
        exp_key = "%s" % str(date.today())
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/%s-%s/jobs' % (dataset_name, version), exp_key=exp_key)
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, trials=trials, max_evals=500)
    else:
        best = fmin(gluon_fcast, space, algo=tpe.suggest, show_progressbar=False, max_evals=20)
         
    return space_eval(space, best) 
    
if __name__ == "__main__":
    data = load_plos_m3_data()
    params = call_hyperopt()
    logger.info("Best params: %s" % params)