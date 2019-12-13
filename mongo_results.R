library(lubridate)
library(tidyverse)
library(ggpubr)
# library(tidyquant)

mongo_data <- read_csv("mongo_results_plos1_m3.csv")
min_err <- 0.5
max_err <- 2.0

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
  # filter(search.time > 120) %>%
  inner_join(model_counts) %>%
  mutate(search.time = (as.numeric(end.time) - min(as.numeric(end.time))) / 3600) %>%
  mutate(training.time = (as.numeric(end.time) - as.numeric(start.time)) / 3600) %>%
  mutate(run.num = row_number())
  # filter(end.time > min(start.time) + hours(6))
# mutate(delta.smape.error = test.sMAPE - train.sMAPE) # %>%
# mutate(delta.mase.error = test.MASE - train.MASE) # %>%
# arrange(end.time) %>%
#  %>%

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
  filter(train.MASE < max_err) %>%
  # select(train.MASE, search.time, model.type.count) %>%
  # gather(metric, error, -search.time) %>%
  ggplot(aes(x = search.time, y = train.MASE)) +
  facet_wrap( ~ model.type.count) +
  geom_smooth(size = 0.5,
              method = 'lm') +
  stat_regline_equation(colour = "red") +
  ylim(NA, max_err) +
  geom_point(size = 0.5) +
  labs(
    title = "Training MASE Trend per Model vs. HyperOpt Search time",
    subtitle = paste0(
      "Data set: ",
      Sys.getenv("DATASET"),
      ", Run: ",
      Sys.getenv("VERSION")
    ),
    x = "HyperOpt Search time (hours)",
    y = "Training MASE"
  )
print(gg_train_mase_per_model)
ggsave("train_mase_per_model.png",
       width = 8,
       height = 6)

hours <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)
gg_hyperopt_path <-
  mongo_plot_data %>%
  ggplot(aes(
    x = train.MASE,
    y = test.MASE)) +
  facet_wrap( ~ model.type, scales = "free") +
  geom_point(size=0.5) +
  geom_abline(intercept = 0.0, slope = 1.0, linetype = "dotted") +
  geom_path(size=0.1, arrow = arrow(type = "closed", length = unit(0.125, "inches"))) +
  # geom_text(aes(label=run.num), position = position_dodge(0.1), size=3) +
  # scale_y_log10() +
  # scale_x_log10() +
  coord_cartesian(xlim = c(min_err, max_err), ylim = c(min_err, max_err)) +
  scale_size(breaks = hours) +
  labs(
    title = "Train vs. Test MASE (HyperOpt Search Path)",
    subtitle = paste0(
      "Data set: ",
      Sys.getenv("DATASET"),
      ", Run: ",
      Sys.getenv("VERSION")
    ),
    x = "Train Set MASE",
    y = "Test Set MASE"
  )
print(gg_hyperopt_path)
ggsave("test_train_mase_hyperopt_path.png",
       width = 8,
       height = 6)

gg_hyperopt_search_time <-
  mongo_plot_data %>%
  ggplot() +
  # facet_wrap( ~ model.type) +
  geom_point(aes(x = train.MASE,
                 size = search.time,
                 y = test.MASE),
             shape = "circle plus") +
  geom_abline(intercept = 0.0, slope = 1.0, linetype = "dotted") +
  scale_y_log10() +
  scale_x_log10() +
  coord_cartesian(xlim = c(min_err, max_err), ylim = c(min_err, max_err)) +
  scale_size(breaks = hours) +
  labs(
    title = "Train vs. Test MASE (HyperOpt Search Time)",
    subtitle = paste0(
      "Data set: ",
      Sys.getenv("DATASET"),
      ", Run: ",
      Sys.getenv("VERSION")
    ),
    # x = "Train Set MASE (log scale)",
    # y = "Test Set MASE (log scale)"
    x = "Train Set MASE",
    y = "Test Set MASE",
    size = "HyperOpt Search\nTime (hours)"
  )
print(gg_hyperopt_search_time)
ggsave("test_train_mase_hyperopt_search_time.png",
       width = 8,
       height = 6)


gg_model_time <-
  mongo_plot_data %>%
  filter(train.MASE < 2.0) %>%
  ggplot() +
  facet_wrap( ~ model.type) +
  geom_point(aes(x = train.MASE,
                 y = test.MASE,
                 size = training.time),
             shape = "circle plus") +
  geom_abline(intercept = 0.0, slope = 1.0, linetype = "dotted") +
  # scale_y_log10() +
  # scale_x_log10() +
  coord_cartesian(xlim = c(min_err, max_err), ylim = c(min_err, max_err)) +
  scale_size(breaks = c(1, 2, 4, 8, 16, 32)) +
  labs(
    title = "Train vs. Test MASE (GluonTS Model Build Time)",
    subtitle = paste0(
      "Data set: ",
      Sys.getenv("DATASET"),
      ", Run: ",
      Sys.getenv("VERSION")
    ),
    # x = "Train Set MASE (log scale)",
    # y = "Test Set MASE (log scale)"
    x = "Train Set MASE",
    y = "Test Set MASE",
    size = "GluonTS Model\nBuild Time (hours)"
  )
print(gg_model_time)
ggsave("test_train_mase_model_time.png",
       width = 8,
       height = 6)

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
