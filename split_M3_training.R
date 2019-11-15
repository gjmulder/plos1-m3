library(Mcomp)
library(parallel)
library(lubridate)
# library(anytime)
library(jsonlite)
# library(forecast)
# library(zoo)
library(tidyverse)

set.seed(42)
options(warn = 0)
options(width = 1024)

###########################################################################
# Config ####

if (interactive()) {
  prop_ts <- NA
  num3_cores <- 4
} else
{
  prop_ts <- NA
  num3_cores <- 16
}
use_parallel <- TRUE #is.na(prop_ts)

###########################################################################
# Preprocess M data ####

if (is.na(prop_ts)) {
  m3_data <- M3
} else {
  m3_data <- sample(M3, prop_ts * length(M3))
}

get_start <- function(ts) {
  iso_date <- date_decimal(as.numeric(time(ts)[1]))
  return (substring(iso_date, 1, 19))
}

ts_to_json <- function(idx, ts_list, type_list) {
  json <- (paste0(toJSON(
    list(
      start = get_start(ts_list[[idx]]),
      target = ts_list[[idx]],
      feat_static_cat = c(idx, as.numeric(type_list[[idx]]))
      # feat_dynamic_real = matrix(rexp(10 * length(ts_list[[idx]])), ncol =
      #                              10)
    ),
    auto_unbox = TRUE
  ), "\n"))
  return(json)
}

process_period <- function(period, m3_data, final_mode) {
  print(period)
  m3_period_data <- keep(subset(m3_data, period), function(ts) length(ts$x) > 80)

  # len_m3_period <-
  #   unlist(lapply(m3_period_data, function(ts)
  #     return(length(ts$x))))
  # print(ggplot(tibble(ts_length = len_m3_period)) + geom3_histogram(aes(x = ts_length), bins=100) + ggtitle(period) + scale_x_log10())

  # if (use_parallel) {
  #   m3_data_x_deseason <- mclapply(1:length(m3_data_x), function(idx)
  #     return(deseasonalise(m3_data_x[[idx]], m3_horiz[[idx]])), mc.cores = num3_cores)
  # } else {
  #   m3_data_x_deseason <- lapply(1:length(m3_data_x), function(idx)
  #     return(deseasonalise(m3_data_x[[idx]], m3_horiz[[idx]])))
  # }

  m3_st <-
    lapply(m3_period_data, function(ts)
      return(ts$st))

  m3_type_str <-
    lapply(m3_period_data, function(ts)
      return(ts$type))
  m3_type_levels <-
    levels(as.factor(unlist(m3_type_str)))
  m3_type <-
    lapply(m3_type_str, function(type)
      return(factor(type, levels = m3_type_levels)))

  m3_horiz <-
    lapply(m3_period_data, function(ts)
      return(ts$h))

  ###########################################################################
  # Create TS depending on final_mode ####

  if (final_mode) {
    dirname <-
      paste0("/tmp/m3_",
             tolower(period),
             '/')
    m3_train <-
      lapply(m3_period_data, function(ts)
        return(ts$x))

    # train + test
    m3_test <-
      lapply(m3_period_data, function(ts)
        return(c(ts$x, ts$xx)))
  } else {
    dirname <-
      paste0("/tmp/m3_",
             tolower(period),
             '/')
    m3_train <-
      lapply(m3_period_data, function(ts)
        return(subset(ts$x, end = (
          length(ts$x) - ts$h
        ))))

    # train + test
    m3_test <-
      lapply(m3_period_data, function(ts)
        return(ts$x))
  }

  ###########################################################################
  # Write JSON train and test data ####

  json <-
    lapply(1:length(m3_train),
           ts_to_json,
           m3_train,
           m3_type)
  sink(paste0(dirname, "train/data.json"))
  lapply(json, cat)
  sink()

  json <-
    lapply(1:length(m3_test),
           ts_to_json,
           m3_test,
           m3_type)
  sink(paste0(dirname, "test/data.json"))
  lapply(json, cat)
  sink()

  return(length(m3_train))
}

final_mode <- TRUE
# periods <- as.vector(levels(m3_data[[1]]$period))
periods <- c("monthly")
res <- unlist(lapply(periods, process_period, m3_data, final_mode))
names(res) <- periods
print(res)
print(sum(res))
