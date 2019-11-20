#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:17:31 2019

@author: mulderg
"""

# awk 'BEGIN {print "date.time,series,value"} /epoch_loss/ {print $1, $2 ",loss,", substr($NF, 14)} /Learning/ {print $1, $2 ",learning.rate,", $NF}'
# ggplot(log_data) + geom_point(aes(x = date.time, y = value)) + facet_grid(series ~ ., scales = "free")

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

from plos_m3 import load_plos_m3_data, forecast
import mxnet as mx
import numpy as np

rand_seed = 24
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

num_eval_samples = 1
freq="M"
prediction_length = 12
    
if __name__ == "__main__":
    data = load_plos_m3_data("./m3_monthly_all")
    cfg = {
			"model" : {
				"dar_dropout_rate" : 0.09083452710754299,
				"num_cells" : 640,
				"num_layers" : 5,
				"type" : "DeepAREstimator"
			},
			"trainer" : {
				"batch_size" : 200,
				"learning_rate" : 0.002012125054989275,
				"learning_rate_decay_factor" : 0.5694992744148958,
				"max_epochs" : 1500,
				"minimum_learning_rate" : 0.000003350876288799133,
				"num_batches_per_epoch" : 320,
				"patience" : 80,
				"weight_decay" : 9.70059777491261e-9
			}
		}
    sMAPE = forecast(data, cfg)
    logger.info("sMAPE: %.4f" % sMAPE)
