#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:17:31 2019

@author: mulderg
"""

# awk 'BEGIN {print "date.time,series,value"} /epoch_loss/ {print $1, $2 ",loss,", substr($NF, 14)} /Learning/ {print $1, $2 ",learning.rate,", $NF}' nohup.log > final_run_training_loss.csv
# read_csv("final_run_training_loss.csv") %>% filter(date.time > ymd_hms("2019-11-20 18:00:00")) %>% ggplot(aes(x = date.time, y = value)) + geom_point() + facet_grid(series ~ ., scales = "free") + scale_y_log10()



from plos_m3 import load_plos_m3_data, forecast
import mxnet as mx
import numpy as np

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

rand_seed = 24
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

num_eval_samples = 1
freq_pd = "M"
freq = 12
prediction_length = 18
    
if __name__ == "__main__":
    data, season_coeffs = load_plos_m3_data("./m3_monthly_all")
#"loss" : 1.0307178497314453,
#"status" : "ok",
#    cfg = {
#		"model" : {
#			"dar_dropout_rate" : 0.09603267184884913,
#			"num_cells" : 512,
#			"num_layers" : 5,
#			"type" : "DeepAREstimator"
#		},
#		"trainer" : {
#			"batch_size" : 256,
#			"learning_rate" : 0.00122504362261288,
#			"learning_rate_decay_factor" : 0.5840807488994838,
#			"max_epochs" : 30000,
#			"minimum_learning_rate" : 0.000004764018847166416,
#			"num_batches_per_epoch" : 64,
#			"patience" : 32,
#			"weight_decay" : 7.1153516607763915e-9
#		}
#	}
#    "loss" : 1.0176126497101416,
#    "status" : "ok",
    cfg = {
            "model" : {
                    "dar_dropout_rate" : 0.11271613303817168,
                    "num_cells" : 512,
                    "num_layers" : 3,
                    "type" : "DeepAREstimator"
            },
            "trainer" : {
                    "batch_size" : 200,
                    "learning_rate" : 0.0007973069989280367,
                    "learning_rate_decay_factor" : 0.7203949865256299,
                    "max_epochs" : 9000,
                    "minimum_learning_rate" : 0.0000012099127344131707,
                    "num_batches_per_epoch" : 80,
                    "patience" : 80,
                    "weight_decay" : 8.150350130265227e-8
    
    
            }
    }
    loss = forecast(data, season_coeffs, cfg)
    logger.info("Loss: %.4f" % loss)
