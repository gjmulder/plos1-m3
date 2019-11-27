#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:17:31 2019

@author: mulderg
"""

# awk 'BEGIN {print "date.time,series,value"} /epoch_loss/ {print $1, $2 ",loss,", substr($NF, 14)} /Learning/ {print $1, $2 ",learning.rate,", $NF}' nohup.log > final_run_training_loss.csv
# read_csv("final_run_training_loss.csv") %>% filter(date.time > ymd_hms("2019-11-20 18:00:00")) %>% ggplot(aes(x = date.time, y = value)) + geom_point() + facet_grid(series ~ ., scales = "free") + scale_y_log10()

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
prediction_length = 18
    
if __name__ == "__main__":
    data = load_plos_m3_data("./m3_monthly_all")
    cfg = {
#			"model" : {
#				"dar_dropout_rate" : 0.09083452710754299,
#				"num_cells" : 640,
#				"num_layers" : 5,
#				"type" : "DeepAREstimator"
#			},
#			"trainer" : {
#				"batch_size" : 200,
#				"learning_rate" : 0.002012125054989275,
#				"learning_rate_decay_factor" : 0.5694992744148958,
#				"max_epochs" : 5000,
#				"minimum_learning_rate" : 0.000003350876288799133,
#				"num_batches_per_epoch" : 320,
#				"patience" : 80,
#				"weight_decay" : 9.70059777491261e-9
#			}
			"model" : {
				"dar_dropout_rate" : 0.1059116635338751,
				"num_cells" : 1000,
				"num_layers" : 5,
				"type" : "DeepAREstimator"
			},
			"trainer" : {
				"batch_size" : 160,
				"learning_rate" : 0.0016089266220730123,
				"learning_rate_decay_factor" : 0.573950423419042,
				"max_epochs" : 500,
				"minimum_learning_rate" : 0.000004885314833535221,
				"num_batches_per_epoch" : 320,
				"patience" : 80,
				"weight_decay" : 9.101039588533137e-9
			}
		}
    loss = forecast(data, cfg)
    logger.info("Loss: %.4f" % loss)
