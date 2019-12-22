#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:17:31 2019

@author: mulderg
"""

# awk 'BEGIN {print "date.time,series,value"} /epoch_loss/ {print $1, $2 ",loss,", substr($NF, 14)} /Learning/ {print $1, $2 ",learning.rate,", $NF}' nohup.log > final_run_training_loss.csv
# read_csv("final_run_training_loss.csv") %>% filter(date.time > ymd_hms("2019-11-20 18:00:00")) %>% ggplot(aes(x = date.time, y = value)) + geom_point() + facet_grid(series ~ ., scales = "free") + scale_y_log10()

from logging import basicConfig, getLogger
from logging import DEBUG as log_level
#from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

from plos_m3 import rand_seed, gluonts_fcast
import mxnet as mx
import numpy as np
from pprint import pformat

mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)
    
if __name__ == "__main__":
	"result" : {
		"loss" : 7.0423109573112805,
		"status" : "ok",
		"cfg" : {
			"box_cox" : true,
			"model" : {
				"dar_dropout_rate" : 0.09400281426733578,
				"num_cells" : 288,
				"num_layers" : 10,
				"type" : "DeepAREstimator"
			},
			"tcrit" : -1,
			"trainer" : {
				"batch_size" : 288,
				"learning_rate" : 0.0005057298188293406,
				"learning_rate_decay_factor" : 0.26508738312094304,
				"max_epochs" : 144,
				"minimum_learning_rate" : 0.000017170580611915995,
				"num_batches_per_epoch" : 1280,
				"patience" : 24,
				"weight_decay" : 6.9587678330069426e-9
			}
		}
		"err_metrics" : {
			"train" : {
				"mase" : 0.4999524652957916,
				"smape" : 7.0423109573112805
			},
			"test" : {
				"mase" : 0.6042534112930298,
				"smape" : 8.242183214418231
			}
		},
		"build_url" : "http://10.10.70.18:8080/job/gpu_asushimu/INSTANCE=0,label=asushimu/91/"
	},

            
    results = gluonts_fcast(cfg)
    logger.info("Final results:\n%s" % pformat(results, indent=4, width=160))
