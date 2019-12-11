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
  mutate(training.duration.minutes = as.numeric(end.time - start.time))

  # filter(end.time > min(start.time) + hours(6)) %>%
  # mutate(delta.smape.error = test.sMAPE - train.sMAPE) # %>%
  # mutate(delta.mase.error = test.MASE - train.MASE) # %>%
  # arrange(end.time) %>%
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

gg_train_mase_per_model <-
  mongo_plot_data %>%
  # filter(search.time > 120) %>%
  inner_join(model_counts) %>%
  select(train.MASE, search.time, model.type.count) %>%
  # gather(metric, error, -search.time) %>%
  ggplot(aes(x = search.time, y = train.MASE)) +
  facet_wrap(~ model.type.count) +
  geom_smooth(size = 0.5,
              method = 'lm') +
  stat_regline_equation(colour = "red") +
  scale_y_log10() +
  geom_point(size = 0.5) +
  labs(
    title = "Training MASE Trend per Model vs. HyperOpt Search Duration",
    subtitle = paste0("Data set: ", Sys.getenv("DATASET"), ", Run: ", Sys.getenv("VERSION")),
    x = "HyperOpt Search Duration (minutes)",
    y = "Training MASE (log scale)")
print(gg_train_mase_per_model)
ggsave("train_mase_per_model.png", width=8, height=6)

mongo_plot_data %>%
  select(train.MASE, test.MASE, search.time) %>%
  gather(MASE, error,-search.time) %>%
  ggplot(aes(x = search.time, y = error, colour = MASE)) +
  geom_point(size = 0.5, alpha = 0.75) +
  scale_y_log10() +
  scale_color_manual(labels = c("Test", "Train"),
                     values = c("red", "blue")) +
  labs(
    title = "Test and Training MASE vs. HyperOpt Search Duration",
    subtitle = paste0("Data set: ", Sys.getenv("DATASET"), ", Run: ", Sys.getenv("VERSION")),
    x = "HyperOpt Search Duration (minutes)",
    y = "MASE (log scale)",
    colour = "Data Set\n"
  ) ->
  gg_mase_time
print(gg_mase_time)
ggsave("test_train_mase_hyperopt_duration.png", width=8, height=6)

mongo_plot_data %>%
  select(train.MASE, test.MASE, training.duration.minutes) %>%
  gather(MASE, error,-training.duration.minutes) %>%
  ggplot(aes(x = training.duration.minutes, y = error, colour = MASE)) +
  geom_point(size = 0.5, alpha = 0.75) +
  scale_y_log10() +
  scale_color_manual(labels = c("Test", "Train"),
                     values = c("red", "blue")) +
  labs(
    title = "Test and Training MASE vs. Duration of GluonTS Model Training",
    subtitle = paste0("Data set: ", Sys.getenv("DATASET"), ", Run: ", Sys.getenv("VERSION")),
    x = "Duration of GluonTS Model Training (minutes)",
    y = "MASE (log scale)",
    colour = "Data Set\n"
  ) ->
  gg_mase_duration
print(gg_mase_duration)
ggsave("test_train_mase_run_duration.png", width=8, height=6)

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
