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
import pandas as pd
from datetime import date
from math import sqrt
#from statistics import mean

from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from os import environ
from traceback import format_exc
from json import loads

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
#from gluonts.evaluation import Evaluator

########################################################################################################

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

if "VERSION" in environ:    
    version = environ.get("VERSION")
    logger.info("Using version : %s" % version)
    
    use_cluster = True
else:
    version = "final"
    logger.warning("VERSION not set, using: %s" % version)
    
    use_cluster = False

if "DATASET" in environ:    
    dataset_name = environ.get("DATASET")
    logger.info("Using dataset : %s" % dataset_name)
    
    use_cluster = True
else:
    dataset_name = "final"
    logger.warning("DATASET not set, using: %s" % dataset_name)
    
num_eval_samples = 1
freq_pd = "M"
freq = 12
prediction_length = 18
        
def detrend(insample_data):
    """
    Calculates a & b parameters of LRL
    :param insample_data:
    :return:
    """
    x = np.arange(len(insample_data))
    a, b = np.polyfit(x, insample_data, 1)
    return a, b


def deseasonalize(original_ts, ppy):
    """
    Calculates and returns seasonal indices
    :param original_ts: original data
    :param ppy: periods per year
    :return:
    """
    """
    # === get in-sample data
    original_ts = original_ts[:-out_of_sample]
    """
    if seasonality_test(original_ts, ppy):
        # print("seasonal")
        # ==== get moving averages
        ma_ts = moving_averages(original_ts, ppy)

        # ==== get seasonality indices
        le_ts = original_ts * 100 / ma_ts
        le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
        le_ts = np.reshape(le_ts, (-1, ppy))
        si = np.nanmean(le_ts, 0)
        norm = np.sum(si) / (ppy * 100)
        si = si / norm
    else:
        # print("NOT seasonal")
        si = np.full(ppy, 100)

    return si


def moving_averages(ts_init, window):
    """
    Calculates the moving averages for a given TS
    :param ts_init: the original time series
    :param window: window length
    :return: moving averages ts
    """
    """
    As noted by Professor Isidro Lloret Galiana:
    line 82:
    if len(ts_init) % 2 == 0:
    
    should be changed to
    if window % 2 == 0:
    
    This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
    In order for the results to be fully replicable this change is not incorporated into the code below
    """
    
#    if len(ts_init) % 2 == 0:
#        ts_ma = pd.rolling_mean(ts_init, window, center=True)
#        ts_ma = pd.rolling_mean(ts_ma, 2, center=True)
#        ts_ma = np.roll(ts_ma, -1)
#    else:
#        ts_ma = pd.rolling_mean(ts_init, window, center=True)

    if window % 2 == 0:
        ts_ma = pd.Series(ts_init).rolling(window, center=True).mean()
        ts_ma = ts_ma.rolling(2, center=True).mean()
        ts_ma = np.roll(ts_ma.values, -1)
    else:
        ts_ma = pd.Series(ts_init).rolling(window, center=True).values
        
    return ts_ma

def acf(data, k):
    """
    Autocorrelation function
    :param data: time series
    :param k: lag
    :return:
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)

def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    :param original_ts: time series
    :param ppy: periods per year
    :return: boolean value: whether the TS is seasonal
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit

def smape(a, b):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()

def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))

    return np.mean(abs(y_test - y_hat_test)) / masep

#######################################################################################
    
def load_plos_m3_data(path):
    data = {}
    for dataset in ["train", "test"]:
        data[dataset] = []
        data["%s-nocat" % dataset] = []
        with open("%s/%s/data.json" % (path, dataset)) as fp:
            for line in fp:
               ts_data = loads(line)               
               data[dataset].append(ts_data)
               
    season_coeffs = []          
    for idx in range(len(data["train"])):
        ts_train = data["train"][idx]["target"]
        
        # determine seasonality coeffs
        seasonality_in = deseasonalize(np.array(ts_train), freq)
        season_coeffs.append(seasonality_in)
        
        #  deaseasonalise training data
        for i in range(0, len(ts_train)):
            ts_train[i] = ts_train[i] * 100 / seasonality_in[i % freq]
            
        #  deaseasonalise test data
        ts_test  = data["test"][idx]["target"]
        for i in range(0, len(ts_test)):
            ts_test[i] = ts_test[i] * 100 / seasonality_in[i % freq]
            
        data["train"][idx]["target"] = ts_train
        data["test"][idx]["target"] = ts_test
        
#        data["train-nocat"][idx]["target"] = ts_train
        
    return data, season_coeffs
     
def forecast(data, season_coeffs, cfg):
    logger.info("Params: %s " % cfg)
        
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        gluon_train = ListDataset(data['train-nocat'].copy(), freq=freq_pd)
    else:
        gluon_train = ListDataset(data['train'].copy(), freq=freq_pd)
    
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
            freq=freq_pd,
#            scaling=use_default_scaler,
            prediction_length=prediction_length, 
            num_hidden_dimensions = cfg['model']['num_hidden_dimensions'],
            num_parallel_samples=1,
            trainer=trainer)

    if cfg['model']['type'] == 'DeepFactorEstimator': 
         estimator = DeepFactorEstimator(
            freq=freq_pd,
#            scaling=use_default_scaler,
            prediction_length=prediction_length,
            num_hidden_global=cfg['model']['num_hidden_global'], 
            num_layers_global=cfg['model']['num_layers_global'], 
            num_factors=cfg['model']['num_factors'], 
            num_hidden_local=cfg['model']['num_hidden_local'], 
            num_layers_local=cfg['model']['num_layers_local'],
            trainer=trainer)
         
    if cfg['model']['type'] == 'DeepAREstimator':            
        estimator = DeepAREstimator(
            freq=freq_pd,
#            scaling=use_default_scaler,
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
            freq=freq_pd,
#            scaling=use_default_scaler,
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
    logger.info(estimator)
    
    model = estimator.train(gluon_train)
    
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        gluon_test = ListDataset(data['test-nocat'].copy() , freq=freq_pd)
    else:
        gluon_test = ListDataset(data['test'].copy(), freq=freq_pd)
    
    forecast_it, ts_it = make_evaluation_predictions(dataset=gluon_test, predictor=model, num_eval_samples=1)
    forecasts = list(forecast_it)
    
#    agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(data['test']))
#    logger.info("GluonTS  MASE : %.6f" % agg_metrics['MASE'])
#    logger.info("GluonTS sMAPE : %.3f" % float(100 * agg_metrics['sMAPE']))

    mases = []
    smapes = []
    # add seasonality and compute MASE
    for idx in range(len(forecasts)):
        ts_train = data['train'][idx]['target']
        for i in range(0, len(ts_train)):
            ts_train[i] = ts_train[i] * season_coeffs[idx][i % freq] / 100   

        ts_test = data['test'][idx]['target']
        for i in range(0, len(ts_test)):
            ts_test[i] = ts_test[i] * season_coeffs[idx][i % freq] / 100  
            
        y_hat_test = forecasts[idx].samples.reshape(-1)
        for i in range(len(ts_train), len(ts_train) + prediction_length):
            y_hat_test[i - len(ts_train)] = y_hat_test[i - len(ts_train)] * season_coeffs[idx][i % freq] / 100

        mases.append(mase(np.array(ts_test[:-prediction_length]), np.array(ts_test[-prediction_length:]), y_hat_test, freq))
        smapes.append(smape(np.array(ts_test[-prediction_length:]), y_hat_test))
    mean_mase = np.mean(mases)
    logger.info("MASE  : %.6f" % mean_mase)
    logger.info("sMAPE : %.3f" % float(100 * float(np.mean(smapes))))
    return mean_mase

def gluon_fcast(cfg):        
    try:
        err = forecast(data, season_coeffs, cfg)
        if np.isnan(err) or np.isinf(err):
            return {'loss': err, 'status': STATUS_FAIL, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL")}
    except Exception as e:
        exc_str = format_exc()
        logger.error('\n%s' % exc_str)
        return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'exception': exc_str, 'build_url' : environ.get("BUILD_URL")}
        
    return {'loss': err, 'status': STATUS_OK, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL")}

def call_hyperopt():
    dropout_rate = {
        'min' : 0.08,
        'max' : 0.12
    }
    
#    transformer_seqs = ['d', 'r', 'n', 'dn', 'nd', 'rn', 'nr', 'dr', 'rd',
#                        'drn', 'dnr', 'rdn', 'rnd', 'nrd', 'ndr']
    space = {
#        'preprocessing' : hp.choice('preprocessing', [None, 'min_max', 'max_abs', 'power_std']),
#        
#        'deseasonalise' : hp.choice('deseasonalise', [
#                                        {'model' : None},
#                                        {'model' : 'mult', 'coeff_as_xreg' : False},
#                                        {'model' : 'mult', 'coeff_as_xreg' : True},
#                                        {'model' : 'add', 'coeff_as_xreg' : False},
#                                        {'model' : 'add', 'coeff_as_xreg' : True},
#                                    ]),
        
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
            'max_epochs'                 : hp.choice('max_epochs', [128, 256, 512, 1024, 2048]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [32, 64, 128, 256, 512]),
            'batch_size'                 : hp.choice('batch_size', [32, 64, 128, 256]),
            'patience'                   : hp.choice('patience', [8, 16, 32, 64]),
            
            'learning_rate'              : hp.uniform('learning_rate', 1e-03, 3e-03),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.5, 0.6),
            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', np.log(2e-06), np.log(5e-06)),
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
                'num_cells'                  : hp.choice('num_cells', [4, 8, 16, 32, 64, 128, 256, 512]),
                'num_layers'                 : hp.choice('num_layers', [1, 3, 5, 7, 9]),
                
                'dar_dropout_rate'           : hp.uniform('dar_dropout_rate', dropout_rate['min'], dropout_rate['max']),
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
                'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate['min'], dropout_rate['max']),
            },
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
    data, season_coeffs  = load_plos_m3_data("./m3_monthly")
    params = call_hyperopt()
    logger.info("Best params: %s" % params)