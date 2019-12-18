library(Mcomp)
library(lubridate)
library(jsonlite)
library(Metrics)
library(tidyverse)

results <- read_csv("3hours_15features.csv")
# results
df <-
  results %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = ymd(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y)

mapes <-
  results %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = ymd(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y) %>%
  group_by(IDX) %>%
  top_n(-1, start_date) %>%
  mutate(ts_smape = smape(Y, predicted_Y))

print(paste0("Mean sMAPE for test set: ", round(mean(mapes$ts_smape)*100, 2)))
