library(lubridate)
library(tidyverse)
library(ggpubr)
# library(tidyquant)

mongo_data <- read_csv("mongo_results_plos1_m3.csv")

model_counts <-
  mongo_data %>%
  group_by(model.type) %>%
  summarise(models = n()) %>%
  mutate(model.type.count = paste0(substring(model.type, 1, nchar(model.type) -
                                               9),
                                   " (",
                                   models,
                                   " models)"))

mongo_plot_data <-
  mongo_data %>%
  mutate(search.time = (end.time - min(end.time))/60) %>%
  mutate(training.duration.minutes = as.numeric(end.time - start.time)) %>%
  arrange(end.time)

  # filter(end.time > min(start.time) + hours(6)) %>%
  # mutate(delta.smape.error = test.sMAPE - train.sMAPE) # %>%
  # mutate(delta.mase.error = test.MASE - train.MASE) # %>%
  # mutate(rank = 1:nrow(mongo_data)) %>%

# mutate(experiment = as.factor(experiment))
# gather(
#   metric,
#   error,
#   test.sMAPE,
#   train.sMAPE,
#   # train.MASE,
#   # test.MASE,
#   # models,
# )

gg_train_smape_time <-
  mongo_plot_data %>%
  inner_join(model_counts) %>%
  gather(metric, error, train.sMAPE, model.type.count) # %>%
#   ggplot(aes(x = end.time, y = train.sMAPE)) +
#   facet_grid(model.type.counts ~ ., scales = "free") +
#   geom_smooth(size = 1,
#               method = 'lm') +
#   stat_regline_equation(colour = "red") +
#   scale_y_log10() +
#   geom_point(size = 1) +
#   xlab("Result Time") +
#   ylab("Training sMAPE") +
#   ggtitle("Training sMAPE vs. Search Time")
# print(gg_train_smape_time)

# gg_test_smape_time <-
#   ggplot(mongo_plot_data, aes(x = end.time, y = test.sMAPE)) +
#   # facet_grid(model.type ~ ., scales = "free") +
#   # geom_smooth(size = 1,
#   #             method = 'lm') +
#   # stat_regline_equation(colour = "red") +
#   scale_y_log10() +
#   geom_point(size = 1) +
#   xlab("Result Time") +
#   ylab("Testing sMAPE") +
#   ggtitle("Test sMAPE vs. Search Time")
# print(gg_test_smape_time)

mongo_plot_data %>%
  select(train.sMAPE, test.sMAPE, search.time) %>%
  gather(sMAPE, error,-search.time) %>%
  ggplot(aes(x = search.time, y = error, colour = sMAPE)) +
  geom_point(size = 0.75, alpha = 0.75) +
  scale_y_log10() +
  scale_color_manual(labels = c("Test", "Train"),
                     values = c("red", "blue")) +
  labs(
    title = "Test and Training sMAPE vs. HyperOpt Search Duration",
    x = "HyperOpt Search Duration (minutes)",
    y = "sMAPE (log scale)",
    colour = "Data Set\n"
  ) ->
  gg_smape_time
print(gg_smape_time)

# mongo_plot_data %>%
#   select(train.sMAPE, test.sMAPE, end.time) %>%
#   ggplot(aes(x = end.time, y = test.sMAPE)) +
#   geom_point(size = 1) +
#   geom_barchart(aes(open = train.sMAPE, high = train.sMAPE, low = test.sMAPE, close = test.sMAPE)) +
#   scale_y_log10() +
#   xlab("Result Time") +
#   ylab("sMAPE") +
#   ggtitle("Test and Training Error sMAPE vs. Result Time") ->
#   gg_smape_barchart_time
# print(gg_smape_barchart_time)

mongo_plot_data %>%
  select(train.sMAPE, test.sMAPE, training.duration.minutes) %>%
  gather(sMAPE, error,-training.duration.minutes) %>%
  ggplot(aes(x = training.duration.minutes, y = error, colour = sMAPE)) +
  geom_point(size = 0.75, alpha = 0.75) +
  scale_y_log10() +
  scale_color_manual(labels = c("Test", "Train"),
                     values = c("red", "blue")) +
  labs(
    title = "Test and Training sMAPE vs. Duration of GluonTS Model Training",
    x = "Duration of GluonTS Model Training (minutes)",
    y = "sMAPE (log scale)",
    colour = "Data Set\n"
  ) ->
  gg_duration
print(gg_duration)

####################################################

# final_deep_ar <- read_csv("final_run_training_loss.csv") %>%
#   filter(series == "loss")
#
# gg_train_rmse <-
#   ggplot(final_deep_ar, aes(x = epoch, y = value)) +
#   geom_point(size = 0.1) +
#   # facet_grid(series ~ ., scales = "free") +
#   scale_y_log10() +
#   xlab("Training Epoch") +
#   ylab("Training RMSE") +
#   ggtitle("Training RMSE as a function of Training Epoch for GluonTS DeepAR 3 layers, 512 cells/layer, no deseasonalisation")
# print(gg_train_rmse)
