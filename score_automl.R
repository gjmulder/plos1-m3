library(Mcomp)
library(anytime)
library(jsonlite)
library(Metrics)
library(tidyverse)
library(purrr)

results15 <- read_csv("15feature_automl_prediction.csv")

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

print(paste0("Mean sMAPE for 3 hour 15 feature test set: ", round(mean(mapes15$ts_smape)*100, 2)))

#############################################################################

results27 <- read_csv("27feature_automl_prediction.csv")

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

print(paste0("Mean sMAPE for 3 hour 27 feature test set: ", mean(mapes27$ts_smape)*100, 2))

#############################################################################

results5_27 <- read_csv("5hour_27feature_automl_prediction.csv")

df5_27<-
  results5_27 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y)

mapes5_27 <-
  results5_27 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y) %>%
  group_by(IDX) %>%
  top_n(-1, start_date) %>%
  mutate(ts_smape = smape(Y, predicted_Y))

print(paste0("Mean sMAPE for 3.7 hour 27 feature test set: ", mean(mapes5_27$ts_smape)*100, 2))

#############################################################################

results1_40 <- read_csv("1hour_40feature_automl_prediction.csv")

df1_40 <-
  results1_40 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y)

mapes1_40 <-
  results1_40 %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y) %>%
  group_by(IDX) %>%
  top_n(-1, start_date) %>%
  mutate(ts_smape = smape(Y, predicted_Y))

print(paste0("Mean sMAPE for 1 hour 40 feature test set: ", mean(mapes1_40$ts_smape)*100, 2))

#############################################################################

results6_9 <- read_csv("6hour_9feature_automl_prediction.csv")

df6_9 <-
  results6_9  %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y)

mapes6_9  <-
  results6_9  %>%
  filter(DATA_SPLIT == "TEST") %>%
  mutate(start_date = anytime(START_DATE)) %>%
  select(start_date, IDX, Y, predicted_Y) %>%
  group_by(IDX) %>%
  top_n(-1, start_date) %>%
  mutate(ts_smape = smape(Y, predicted_Y))

print(paste0("Mean sMAPE for 6 hour 9 feature test set: ", mean(mapes6_9$ts_smape)*100, 2))

#############################################################################

# results_txt <- readLines("3hours_27features.json")
# results_list <- transpose(lapply(results_txt, fromJSON))
# results27 <- as.data.frame(lapply(results_list, unlist))
#
# df27 <-
#   results27 %>%
#   filter(DATA_SPLIT == "TEST") %>%
#   mutate(start_date = anytime(START_DATE)) %>%
#   select(start_date, IDX, Y, predicted_Y)
#
# mapes27 <-
#   results27 %>%
#   filter(DATA_SPLIT == "TEST") %>%
#   mutate(start_date = anytime(START_DATE)) %>%
#   select(start_date, IDX, Y, predicted_Y) %>%
#   group_by(IDX) %>%
#   top_n(-1, start_date) %>%
#   mutate(ts_smape = smape(Y, predicted_Y))
#
# print(paste0("Mean sMAPE for 27 feature test set: ", round(mean(mapes27$ts_smape)*100, 2)))
