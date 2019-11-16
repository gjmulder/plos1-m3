# library(lubridate)
library(tidyverse)

mongo_data <- read_csv("mongo_results_plos1_m3.csv")

mongo_data %>%
  mutate(duration = end.time - start.time) %>%
  arrange(desc(sMAPE)) %>%
  mutate(rank = 1:nrow(mongo_data)) %>%
  filter(sMAPE < 20) %>%
  mutate(experiment = as.factor(experiment)) ->
  mongo_plot_data

gg <- ggplot(mongo_plot_data) +
  geom_hline(yintercept = min(mongo_plot_data$sMAPE), colour = "green") +
  geom_point(aes(x = rank, y = sMAPE, colour = experiment), size = 0.7) +
  scale_y_log10() +
  # ylim(5,20) +
  geom_hline(yintercept = 7.19) +
  geom_text(aes(x = 0, y = 7), label = "ETS") +
  geom_hline(yintercept = 8.17) +
  geom_text(aes(x = 0, y = 8), label = "BNN")


print(gg)
