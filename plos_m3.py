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
from traceback import format_exc
from math import log, sqrt
from json import loads
from functools import partial

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import sklearn.preprocessing as skpp

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
    
########################################################################################################

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

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
    logger.warning("DATASET not set, using: %s" % dataset_name)
    
num_eval_samples = 1
freq="M"
prediction_length = 18
    
def load_plos_m3_data(path):
    data = {}
    for dataset in ["train", "test"]:
        data[dataset] = []
        data["%s-nocat" % dataset] = []
        with open("%s/%s/data.json" % (path, dataset)) as fp:
            for line in fp:
               ts_data = loads(line)
               data[dataset].append(ts_data)
               ts_data_copy = ts_data.copy()
               del(ts_data_copy['feat_static_cat'])
               data["%s-nocat" % dataset].append(ts_data_copy)
    return data
    
def seasonality_test(ts, period, tcrit):
    if (len(ts) < 3*period):
        return False
    else:
        xacf = acf(ts)
        clim = tcrit / sqrt(len(ts)) * sqrt(np.cumsum(np.arange(1, int(2*xacf**2))))
        return (abs(xacf[period]) > clim[period])

def define_xforms(cfg):
    scaler = None
    if cfg['preprocessing'] == 'min_max':
        scaler = skpp.MinMaxScale
    if cfg['preprocessing'] == 'max_abs':
        scaler = skpp.mMinMaxScaler
    if cfg['preprocessing'] == 'power_std':
        scaler = skpp.PowerTransformer

    if not cfg['deseasonalise']['model'] is not None:
        decomp = partial(sm.tsa.seasonal_decompose, model = cfg['deseasonalise'])
    else:
        decomp = None
    
    return (scaler, decomp)

def xform_data(data, cfg):
    (scaler, decomp) = define_xforms(cfg)

    scalers = []
    decomps = []
    xformed_data = []    
    for ts in xformed_data['train']:
        if scaler is not None:
            ts_scaler = scaler()
            scaled_ts = ts_scaler.fit(ts.target)
        else:
            ts_scaler = None
            scaled_ts = ts.target
        scalers.append(ts_scaler)
            
        if decomp is not None and seasonality_test(ts.target, 12, 1.645):
            ts_decomper = decomp()
            decomped_ts = ts_decomper.fit(scaled_ts)
        else:
            ts_decomper = None
            decomped_ts = scaled_ts
        decomps.append(ts_decomper)
        
        xformed_data.append(decomped_ts)
        
    return (scalers, decomps, xformed_data)

def unxform_data(xformed_data, scalers, decomps):
    data = xformed_data
    return data
        
def forecast(data, cfg):
    logger.info("Params: %s " % cfg)

#    (scalers, decomps, xformed_data) = xform_data(data, cfg)
    
#    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
#        gluon_train = ListDataset(data['train-nocat'], freq=freq)
#    else:
    gluon_train = ListDataset(data['train'], freq=freq)
    
#    trainer=Trainer(
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
 
    logger.info("MASE  : %.4f" % agg_metrics['MASE'])
    logger.info("sMAPE : %.4f" % float(float(agg_metrics['sMAPE'])))
    return agg_metrics['MASE']

def gluon_fcast(cfg):        
    try:
        err = forecast(data, cfg)
        if np.isnan(err) or np.isinf(err):
            return {'loss': err, 'status': STATUS_FAIL, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL")}
    except Exception as e:
        exc_str = format_exc()
        logger.error('\n%s' % exc_str)
        return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'exception': exc_str, 'build_url' : environ.get("BUILD_URL")}
        
    return {'loss': float(float(err)*100), 'status': STATUS_OK, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL")}

def call_hyperopt():
    dropout_rate = {
        'min' : 0.08,
        'max' : 0.12
    }
    
#    transformer_seqs = ['d', 'r', 'n', 'dn', 'nd', 'rn', 'nr', 'dr', 'rd',
#                        'drn', 'dnr', 'rdn', 'rnd', 'nrd', 'ndr']
    space = {
        'preprocessing' : hp.choice('preprocessing', ['default', 'min_max', 'max_abs', 'power_std']),
        
        'deseasonalise' : hp.choice('deseasonalise', [
                                        {'model' : None},
                                        {'model' : 'mult', 'coeff_as_xreg' : False},
                                        {'model' : 'mult', 'coeff_as_xreg' : True},
                                        {'model' : 'add', 'coeff_as_xreg' : False},
                                        {'model' : 'add', 'coeff_as_xreg' : True}]),
        
#        'trainer' : {
#            'max_epochs'                 : hp.choice('max_epochs', [125, 250, 500, 1000, 2000, 4000]),
#            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [5, 10, 20, 40, 80]),
#            'batch_size'                 : hp.choice('batch_size', [25, 50, 100, 200, 400]),
#            'patience'                   : hp.choice('patience', [20, 40, 80]),
#            
#            'learning_rate'              : hp.uniform('learning_rate', 1e-04, 1e-02),
#            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.4, 0.9),
#            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', log(1e-06), log(0.5e-04)),
#            'weight_decay'               : hp.uniform('weight_decay', 00.5e-08, 10.0e-08),
#        },
        'trainer' : {
            'max_epochs'                 : hp.choice('max_epochs', [250, 500, 1000, 2000]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [60, 320, 640]),
            'batch_size'                 : hp.choice('batch_size', [160, 180, 200, 220, 240]),
            'patience'                   : hp.choice('patience', [60, 80, 100]),
            
            'learning_rate'              : hp.uniform('learning_rate', 1e-03, 3e-03),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.5, 0.6),
            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', log(2e-06), log(5e-06)),
            'weight_decay'               : hp.uniform('weight_decay', 5.0e-09, 15.0e-09),
        },
        'model' : hp.choice('model', [
#            {
#                'type'                       : 'SimpleFeedForwardEstimator',
#                'num_hidden_dimensions'      : hp.choice('num_hidden_dimensions', [[5], [10], [25], [50], [100],
#                                                                                   [5, 5], [10, 10], [25, 25], [50, 25], [50, 50], [100, 50], [100, 100],
#                                                                                   [100, 50, 50]])
#            },
#            {
#                'type'                       : 'DeepFactorEstimator',
#                'num_hidden_global'          : hp.choice('num_hidden_global', [5, 10, 20, 40, 80, 160, 320]),
#                'num_layers_global'          : hp.choice('num_layers_global', [1, 2, 3]),
#                'num_factors'                : hp.choice('num_factors', [5, 10, 20]),
#                'num_hidden_local'           : hp.choice('num_hidden_local', [2, 5, 10]),
#                'num_layers_local'           : hp.choice('num_layers_local', [1, 2, 3]),
#            },
            {
                'type'                       : 'DeepAREstimator',
                'num_cells'                  : hp.choice('num_cells', [600, 800, 1000, 1200]),
                'num_layers'                 : hp.choice('num_layers', [4, 5, 6]),
                
                'dar_dropout_rate'           : hp.uniform('dar_dropout_rate', dropout_rate['min'], dropout_rate['max']),
            },
#            {
#                'type'                       : 'TransformerEstimator',
#                'model_dim_heads'            : hp.choice('model_dim_heads', [[2, 2], [4, 2], [8, 2], [16, 2], [32, 2], [64, 2],
#                                                                             [4, 4], [8, 4], [16, 4], [32, 4], [64, 4],
#                                                                             [8, 8], [16, 8], [32, 8], [64, 8],
#                                                                             [16, 16], [32, 16], [64, 16]]),
#                'inner_ff_dim_scale'         : hp.choice('inner_ff_dim_scale', [2, 3, 4, 5]),
#                'pre_seq'                    : hp.choice('pre_seq', ['dn']),
#                'post_seq'                   : hp.choice('post_seq', ['drn']),
#                'act_type'                   : hp.choice('act_type', ['softrelu']),               
#                'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate['min'], dropout_rate['max']),
#            },
#            {
#                'type'                       : 'TransformerEstimator',
#                'model_dim_heads'            : hp.choice('model_dim_heads', [[32, 8], [64, 16], [128, 32]]),
#                'inner_ff_dim_scale'         : hp.choice('inner_ff_dim_scale', [2, 3, 4, 5]),
#                'pre_seq'                    : hp.choice('pre_seq', transformer_seqs),
#                'post_seq'                   : hp.choice('post_seq', transformer_seqs),
#                'act_type'                   : hp.choice('act_type', ['softrelu']),               
#                'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate['min'], dropout_rate['max']),
#            },
        ])
    }
                
            
    # Search MongoDB for best trial for exp_key:
    # echo 'db.jobs.find({"exp_key" : "XXX", "result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo --host heika m4_daily
    # echo 'db.jobs.remove({"exp_key" : "XXX", "result.status" : "new"})' | mongo --host heika
    if use_cluster:
        exp_key = "%s" % str(date.today())
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/%s-%s/jobs' % (dataset_name, version), exp_key=exp_key)
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, trials=trials, max_evals=10000)
    else:
        best = fmin(gluon_fcast, space, algo=tpe.suggest, show_progressbar=False, max_evals=20)
         
    return space_eval(space, best) 
    
if __name__ == "__main__":
    data = load_plos_m3_data("./m3_monthly")
    params = call_hyperopt()
    logger.info("Best params: %s" % params)