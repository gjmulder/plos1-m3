library(lubridate)
library(tidyverse)
library(ggpubr)
# library(tidyquant)

mongo_data <- read_csv("mongo_results_plos1_m3.csv")
min_err <- min(mongo_data$train.MASE) * 5 / 6
max_err <- min(mongo_data$train.MASE) * 18 / 6

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

mongo_plot_data %>%
  select(
    model.type,
    train.MASE,
    train.sMAPE,
    test.MASE,
    test.sMAPE,
    search.time,
    training.time
  ) %>%
  arrange(train.sMAPE) %>%
  group_by(model.type) %>%
  top_n(1) ->
  top_models
write_csv(top_models, path = "top_models.csv")

hours <- c(1, 2, 4, 8, 16, 32, 64, 128, 256)

mongo_plot_data_decimate <-
  mongo_plot_data %>%
  filter(run.num %% 20 == 0)

#####################################################################

gg_train_mase_per_model <-
  mongo_plot_data %>%
  filter(train.MASE < max_err) %>%
  ggplot(aes(x = search.time, y = train.MASE)) +
  geom_smooth(size = 0.5,
              method = 'lm') +
  stat_regline_equation(colour = "red") +
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
  ) +
  facet_wrap(~ model.type.count)
print(gg_train_mase_per_model)
ggsave("train_mase_per_model.png",
       width = 8,
       height = 6)

gg_hyperopt_path <-
  ggplot() +
  geom_abline(intercept = 0.0,
              slope = 1.0,
              linetype = "dotted") +
  geom_point(
    data = mongo_plot_data %>% filter(train.MASE < max_err &
                                        test.MASE < max_err),
    mapping = aes(x = train.MASE,
                  y = test.MASE),
    size = 0.25,
    alpha = 0.5
  ) +
  geom_path(
    data = mongo_plot_data %>% filter(train.MASE < max_err &
                                        test.MASE < max_err),
    mapping = aes(x = train.MASE,
                  y = test.MASE),
    size = 0.1,
    alpha = 0.5
  ) +
  geom_text(
    data = mongo_plot_data_decimate %>% filter(train.MASE < max_err &
                                                 test.MASE < max_err),
    mapping = aes(x = train.MASE,
                  y = test.MASE,
                  label = run.num),
    nudge_x = min_err / 15,
    size = 2,
    colour = "blue"
  ) +
  scale_size(breaks = hours) +
  labs(
    title = paste0(
      "Train vs. Test MASE (HyperOpt Search Path, ",
      nrow(mongo_data),
      " total models)"
    ),
    subtitle = paste0(
      "Data set: ",
      Sys.getenv("DATASET"),
      ", Run: ",
      Sys.getenv("VERSION")
    ),
    x = "Train Set MASE",
    y = "Test Set MASE"
  ) +
  facet_wrap(~ model.type.count)
print(gg_hyperopt_path)
ggsave("test_train_mase_hyperopt_path.png",
       width = 8,
       height = 6)

gg_hyperopt_search_time <-
  mongo_plot_data %>%
  ggplot() +
  geom_abline(intercept = 0.0,
              slope = 1.0,
              linetype = "dotted") +
  geom_point(
    aes(x = train.MASE,
        size = search.time,
        y = test.MASE),
    shape = "circle plus",
    alpha = 0.5
  ) +
  coord_cartesian(xlim = c(min_err, max_err),
                  ylim = c(min_err, max_err)) +
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
# facet_wrap( ~ model.type)
print(gg_hyperopt_search_time)
ggsave("test_train_mase_hyperopt_search_time.png",
       width = 8,
       height = 6)


gg_model_time <-
  mongo_plot_data %>%
  filter(train.MASE < 2.0) %>%
  ggplot() +
  geom_abline(intercept = 0.0,
              slope = 1.0,
              linetype = "dotted") +
  geom_point(
    aes(x = train.MASE,
        y = test.MASE,
        size = training.time),
    shape = "circle plus",
    alpha = 0.5
  ) +
  # scale_y_log10() +
  # scale_x_log10() +
  coord_cartesian(xlim = c(min_err, max_err),
                  ylim = c(min_err, max_err)) +
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
  ) +
  facet_wrap(~ model.type)
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
