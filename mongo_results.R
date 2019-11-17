# library(lubridate)
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
  mutate(training.duration.secs = as.numeric(end.time - start.time)) %>%
  arrange(desc(sMAPE)) %>%
  inner_join(model_counts) %>%
  mutate(rank = 1:nrow(mongo_data)) %>%
  filter(sMAPE < 20) %>%
  mutate(experiment = as.factor(experiment)) ->
  mongo_plot_data

dl_smape <- min(mongo_plot_data$sMAPE)
ets_smape <- 7.12
bnn_smape <- 7.96

gg <- ggplot(mongo_plot_data) +
  geom_hline(yintercept = dl_smape) +
  geom_text(aes(x = 0, y = dl_smape - 0.3, hjust = "left"), size=3,
            label = paste0("Transformer = ", round(dl_smape, 2))) +
  geom_hline(yintercept = ets_smape) +
  geom_text(aes(x = 0, y = ets_smape - 0.3, hjust = "left"), size=3,
            label = paste0("ETS = ", ets_smape)) +
  geom_hline(yintercept = bnn_smape) +
  geom_text(aes(x = 0, y = bnn_smape - 0.3, hjust = "left"), size=3,
            label = paste0("BNN = ", bnn_smape)) +
  scale_y_log10()

gg_rank <-
  gg +
  geom_point(aes(x = rank, y = sMAPE, colour = experiment),
             size = 0.1) +
  facet_grid(model.type.count ~ .) +
  xlab("Rank") +
  ylab("sMAPE (log scale)") +
  ggtitle("Ranked model accuracy")

gg_duration <-
  gg +
  geom_point(aes(x = training.duration.secs, y = sMAPE, colour = model.type.count),
             size = 0.5) +
  geom_smooth(aes(x = training.duration.secs, y = sMAPE), se = FALSE) +
  xlab("Duration (secs)") +
  ylab("sMAPE (log scale)") +
  ggtitle("Trial duration versus model accuracy")

print(gg_rank)
print(gg_duration)
