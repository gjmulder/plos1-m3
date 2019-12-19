library(Mcomp)
library(anytime)
library(jsonlite)
library(Metrics)
library(tidyverse)
library(purrr)

results15 <- read_csv("3hours_15features.csv")

df15 <-
  results15 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y)
mapes15 <-
  results15 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y) %>%
  group_by(IDX) %>%
  top_n(-1, start_date) %>%
  mutate(ts_smape = smape(Y, predicted_Y))

print(paste0("Mean sMAPE for 15 feature test set: ", round(mean(mapes15$ts_smape)*100, 2)))

#############################################################################

results_txt <- readLines("3hours_27features.json")
results_list <- transpose(lapply(results_txt, fromJSON))
results27 <- as.data.frame(lapply(results_list, unlist))

df27 <-
  results27 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y)

mapes27 <-
  results27 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y) %>%
  group_by(IDX) %>%
  top_n(-1, start_date) %>%
  mutate(ts_smape = smape(Y, predicted_Y))

print(paste0("Mean sMAPE for 27 feature test set: ", round(mean(mapes27$ts_smape)*100, 2)))
