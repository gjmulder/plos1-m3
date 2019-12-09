library(lubridate)
library(tidyverse)

mongo_data <- read_csv("mongo_results_plos1_m3.csv")

mongo_data %>%
  group_by(model.type) %>%
  summarise(models = n()) %>%
  mutate(model.type.count = paste0(substring(model.type, 1, nchar(model.type)-9),
                                   " (",
                                   models,
                                   " models)")) ->
  model_counts

mongo_data %>%
<<<<<<< HEAD
  mutate(training.duration.secs = as.numeric(end.time - start.time)) %>%
  arrange(desc(MASE)) %>%
=======
  mutate(training.duration.hours = as.numeric(end.time - start.time)) %>%
  arrange(desc(sMAPE)) %>%
>>>>>>> preprocess
  inner_join(model_counts) %>%
  mutate(rank = 1:nrow(mongo_data)) %>%
  filter(MASE < 20) %>%
  mutate(experiment = as.factor(experiment)) ->
  mongo_plot_data

<<<<<<< HEAD
dl_MASE <- min(mongo_plot_data$MASE)
ARIMA_MASE <- 0.89
bnn_iter_MASE <- 0.93

gg <- ggplot(mongo_plot_data) +
  geom_hline(yintercept = dl_MASE) +
  geom_text(aes(x = 0, y = dl_MASE - 0.015, hjust = "left"), size=3,
            label = paste0("Deep Auto Regressive = ", round(dl_MASE, 2))) +
  geom_hline(yintercept = ARIMA_MASE) +
  geom_text(aes(x = 0, y = ARIMA_MASE - 0.015, hjust = "left"), size=3,
            label = paste0("ARIMA = ", ARIMA_MASE)) +
  geom_hline(yintercept = bnn_iter_MASE) +
  geom_text(aes(x = 0, y = bnn_iter_MASE - 0.015, hjust = "left"), size=3,
            label = paste0("BNN Iterative = ", bnn_iter_MASE)) +
  scale_y_log10()
=======
dl_smape <- min(mongo_plot_data$sMAPE)
theta_smape <- 10.89
bnn_iterative_smape <- 12.09

gg <- ggplot(mongo_plot_data) +
  geom_hline(yintercept = dl_smape) +
  geom_text(aes(x = 0, y = dl_smape - 0.04, hjust = "left"), size=3,
            label = paste0("Deep Auto Regressive = ", round(dl_smape, 2))) +
  geom_hline(yintercept = theta_smape) +
  geom_text(aes(x = 0, y = theta_smape - 0.04, hjust = "left"), size=3,
            label = paste0("Theta = ", theta_smape)) +
  geom_hline(yintercept = bnn_iterative_smape) +
  geom_text(aes(x = 0, y = bnn_iterative_smape - 0.04, hjust = "left"), size=3,
            label = paste0("BNN Iterative = ", bnn_iterative_smape))
  # scale_y_log10()
>>>>>>> preprocess

gg_rank <-
  gg +
  geom_point(aes(x = rank, y = MASE, colour = experiment),
             size = 2) +
  facet_grid(model.type.count ~ .) +
  xlab("Rank") +
  ylab("MASE (log scale)") +
  ggtitle("Ranked model accuracy")

gg_duration <-
  gg +
<<<<<<< HEAD
  geom_point(aes(x = training.duration.secs, y = MASE, colour = model.type.count),
             size = 2) +
  # geom_smooth(aes(x = training.duration.secs, y = MASE), se = FALSE) +
  xlab("Duration (secs)") +
  ylab("MASE (log scale)") +
=======
  geom_point(aes(x = training.duration.hours, y = sMAPE, colour = model.type.count),
             size = 2) +
  # geom_line(aes(x = training.duration.hours, y = sMAPE)) +
  xlab("Duration (hours)") +
  ylab("sMAPE (log scale)") +
>>>>>>> preprocess
  ggtitle("Trial duration versus model accuracy")

print(gg_rank)
print(gg_duration)

