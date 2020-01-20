what<-c()

# setwd("C:/Users/vangelis spil/Dropbox/M Comp - AI/R experiment/Test preprocessing")
library(forecast)
library(Mcomp)
library(nnet)
library(neuralnet)
library(RSNNS)
library(randtests)
# library(brnn)

SeasonalityTest <- function( insample, ppy, tcrit){

  if (length(insample)<3*ppy){
    test_seasonal = FALSE
  }else{
    xacf = acf(insample, plot = FALSE)$acf[-1, 1, 1]
    clim = tcrit/sqrt(length(insample)) * sqrt(cumsum(c(1, 2 * xacf^2)))
    test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )

    if (is.na(test_seasonal)==TRUE){ test_seasonal = FALSE }
  }

  return(test_seasonal)
}

SeasonalDec<-function(insample,horizon,frequency,type){
  tcrit = 1.645
  seasonal<-matrix(1,ncol=1,nrow=horizon)   #seasonality indexes for forecasting (1 if non seasonal)
  Seasonal_Indexes=matrix(1,ncol=1,nrow=length(insample))  #seasonality indexes for the insample model (1 if non seasonal)
  Des_insample=insample #Deseasonalized t-s (equals insample if non seasonal)
  if (frequency>1){
    test_seasonal <- SeasonalityTest(insample, frequency, tcrit)
    if (test_seasonal==TRUE){
      Seasonal_Indexes=DecomposeC(insample,frequency)$Seasonality
      Des_insample=insample/Seasonal_Indexes
      for (s in 1:horizon){
        seasonal[s] = Seasonal_Indexes[length(insample)-2*frequency+s]
      }
    }
  }
  return(list(Seasonal_Indexes,seasonal,Des_insample))
}

DecomposeC = function(insample,frequency){

  frame<-data.frame(matrix(data = NA, nrow = length(insample), ncol = 1, byrow = FALSE,dimnames = NULL))
  colnames(frame)<-c("ID"); frame$Data<-insample

  ID<-c(); IDref<-c(1:frequency) # which month is this observation?
  for (i in 1:(length(insample)%/%frequency)){ ID<-c(ID,IDref) }
  ID<-c(ID,head(IDref,(length(insample)%%frequency))) ;frame$ID<-ID


  if (frequency==1){
    frame$Seasonality<-1
    frame$kmo<-NA # moving average based on frequency
  }else{
    if (frequency%%2==0){
      frame$kmo<-ma(insample,order=frequency,centre=TRUE)
    }else{
      frame$kmo<-ma(insample,order=frequency,centre=FALSE)
    }

  }

  #Calculate SR and SI
  SRTable<-1
  if (frequency>1){
    frame$LE<-frame$Data/frame$kmo
    LE<-matrix(data = NA, nrow = frequency, ncol = 2, byrow = FALSE,dimnames = NULL)
    LE<-data.frame(LE); colnames(LE)<-c("ID","LE"); LE$ID<-c(1:frequency)

    if (length(frame[ is.na(frame$LE)==FALSE, ]$LE)>=3*frequency){
      for (i in 1:frequency){
        LE$LE[i]<-mean(frame$LE[ (frame$ID==i) & (is.na(frame$LE)==FALSE) & (frame$LE<max(frame$LE[(frame$ID==i)&(is.na(frame$LE)==FALSE)])) & (frame$LE>min(frame$LE[(frame$ID==i)&(is.na(frame$LE)==FALSE)])) ])
      }
    }else{
      for (i in 1:frequency){
        LE$LE[i]<-median(frame$LE[ (frame$ID==i) & (is.na(frame$LE)==FALSE) ])
      }
    }

    SRTable<-frame$LE

    sndarize=mean(LE$LE) ; LE$LE<-LE$LE/sndarize ;  frame$LE<-NULL; frame$kmo<-NA
    DE<-c(); DEref<-LE$LE
    for (i in 1:(length(insample)%/%frequency)){ DE<-c(DE,DEref) }
    DE<-c(DE,head(DEref,(length(insample)%%frequency))) ; frame$Seasonality<-DE
  }

  if (is.na(frame$Seasonality)[1]==TRUE){ frame$Seasonality<-1 }
  frame$Deseasonalized<-frame$Data/frame$Seasonality

  #Calculate Randomness
  frame$kmo<-ma(frame$Deseasonalized,order=3,centre=FALSE)
  frame$kmo3<-ma(frame$kmo,order=3,centre=FALSE)
  frame$kmo3[2]<-frame$kmo[2]
  frame$kmo3[1]<-((frame$Deseasonalized[1]+frame$Deseasonalized[2])/2)+(frame$kmo3[2]-frame$kmo3[3])/2
  frame$kmo3[length(insample)-1]<-frame$kmo[length(insample)-1]
  frame$kmo3[length(insample)]<-((frame$Deseasonalized[length(insample)]+frame$Deseasonalized[length(insample)-1])/2)+(frame$kmo3[length(insample)-1]-frame$kmo3[length(insample)-2])/2
  frame$Randomness<-frame$Deseasonalized/frame$kmo3
  frame$kmo3=frame$kmo=frame$ID=LE<-NULL

  #Calculate Trend and Cyrcle
  TC<-frame$Deseasonalized/frame$Randomness ; frame$Deseasonalized<-NULL
  xs<-c(1:length(insample))
  frame$Trend<-as.numeric(predict(lm(TC~xs)))
  frame$Cyrcle<-TC/frame$Trend

  frame$SR<-SRTable ; frame$Period<-ID

  return(frame)
}
#
# ###
#
# CreateSamplesH<-function(datasample,xi,xo){
#
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
# sMAPE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
# MASE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#
#     BestInputs<-c()
#     for (NNid in 1:18){
#
#       SSEinputnodes<-c()
#       for (xi in 1:5){
#
#         #create samples
#         samplegenerate<-CreateSamplesH(datasample=tsmulti,xi=xi,xo=NNid)[,c(1:xi,xi+NNid)]
#         #create 10 folds
#         foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#         start<-1 ; end<-foldlength
#         for (fid in 1:9){
#           Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#           start<-start+foldlength ; end<-end+foldlength
#         }
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#         KfoldsIn=KfoldsOut<-NULL
#
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#         SSE<-0
#         for (TestFolds in 1:10){
#           model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                      size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                      learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                      shufflePatterns = FALSE, linOut = TRUE)
#           for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#           SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#         }
#         SSEinputnodes<-c(SSEinputnodes,SSE)
#
#       }
#       BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#
#
#       BestInputs<-c(BestInputs,BestInputNodes)
#
#
#     }
#
#
#     #These are the models for each fh
#     modelsBPNN<-NULL ; GoF<-0
#     for (NNid in 1:18){
#
#       samplegenerate<-CreateSamplesH(datasample=tsmulti,xi=BestInputs[NNid],xo=NNid)
#       modelsBPNN[length(modelsBPNN)+1]<-list(mlp(as.matrix(samplegenerate[,1:BestInputs[NNid]]), as.matrix(samplegenerate[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]),
#                                                  size = (2*BestInputs[NNid]+1), maxit = 500,initFunc = "Randomize_Weights",
#                                                  learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                                                  shufflePatterns = FALSE, linOut = TRUE))
#
#       ttttrend<-CreateSamplesH(datasample=trendin,xi=BestInputs[NNid],xo=NNid)[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]
#       tttseas<-CreateSamplesH(datasample=Decinsample,xi=BestInputs[NNid],xo=NNid)[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]
#       finsample<-CreateSamplesH(datasample=insample,xi=BestInputs[NNid],xo=NNid)[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]
#       ffitted<-InvBoxCox((modelsBPNN[[length(modelsBPNN)]]$fitted.values*(MAX-MIN)+MIN+ttttrend)*tttseas,lambda = lamda)
#
#       GoF<-GoF+(mean((finsample-ffitted)^2)*100/(mean(finsample)^2))
#
#     }
#     GoF<-GoF/18
#
#     #Generate forecasts
#     MLPfs<-c()
#     for (i in 1:18){
#
#       tempin<-t(as.matrix(tail(tsmulti,BestInputs[i])))
#       MLPfs<-c(MLPfs,as.numeric(predict(modelsBPNN[[i]],tempin)))
#
#     }
#     MLPf<-InvBoxCox((MLPfs*(MAX-MIN)+MIN+trendout)*Decoutsample,lambda = lamda)
#
#     forecasts<-data.frame(MLPf)
#
#     #Make negative forecasts equal to zero
#     #Make negative forecasts equal to zero
#     for (k in 1:18){
#       if(forecasts[k,1]<0) { forecasts[k,1]<-0 }
#     }
#
#     #Benchmark for MASE
#     forecastsNaiveSD<-rep(NA,frequency)
#     for (j in (frequency+1):observations){
#       forecastsNaiveSD<-c(forecastsNaiveSD,insample[j-frequency])
#     }
#     masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     Results$MASE[tsi]<-mean(abs(forecasts[,1]-outsample))/masep
#     Results$GoF[tsi]<-GoF
#
#     sMAPE18[tsi,]<-(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     MASE18[tsi,]<-abs(forecasts[,1]-outsample)/masep
#
#     #plot(c(insample,outsample),type="l")
#     #lines(c(insample,forecasts[,1]),col="red",type="l")
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(sMAPE18, file=paste("Results MLP Best multi NNs sMAPE.csv"),row.names=FALSE)
# write.csv(MASE18, file=paste("Results MLP Best multi NNs MASE.csv"),row.names=FALSE)
# write.csv(Results, file=paste("Results MLP Best multi NNs.csv"),row.names=FALSE)
# ###
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
# sMAPE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
# MASE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#
#     finsample<-InvBoxCox((as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#     ffitted<-InvBoxCox((modelsBPNN$fitted.values*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #this contains both insample and outsample
#     tsmulti<-(c((BoxCox(insample,lambda=lamda)/Decinsample)-trendin)-MIN)/(MAX-MIN)
#
#     MLPfs=MLPf<-c()
#     tsmulti<-c(tsmulti,MLPfs)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       #tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#
#       #forecast using t-s methods
#       MLPfs<-c(MLPfs,as.numeric(predict(modelsBPNN,tempin)))
#       tsmulti<-c(tsmulti,MLPfs[length(MLPfs)])
#
#     }
#
#     MLPf<-InvBoxCox((MLPfs*(MAX-MIN)+MIN+trendout)*Decoutsample,lambda = lamda)
#
#     forecasts<-data.frame(MLPf)
#
#     #Make negative forecasts equal to zero
#     for (k in 1:18){
#       if(forecasts[k,1]<0) { forecasts[k,1]<-0 }
#     }
#
#     #Benchmark for MASE
#     forecastsNaiveSD<-rep(NA,frequency)
#     for (j in (frequency+1):observations){
#       forecastsNaiveSD<-c(forecastsNaiveSD,insample[j-frequency])
#     }
#     masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     Results$MASE[tsi]<-mean(abs(forecasts[,1]-outsample))/masep
#     Results$GoF[tsi]<-GoF
#
#     sMAPE18[tsi,]<-(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     MASE18[tsi,]<-abs(forecasts[,1]-outsample)/masep
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(sMAPE18, file=paste("Results MLP Best interatial sMAPE.csv"),row.names=FALSE)
# write.csv(MASE18, file=paste("Results MLP Best interatial MASE.csv"),row.names=FALSE)
# write.csv(Results, file=paste("Results MLP Best interatial.csv"),row.names=FALSE)
#
#
#
# CreateSamplesM<-function(datasample,xi,xo){
#
#   #Normalize insample from 0 to 1
#
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
# sMAPE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
# MASE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamplesM(datasample=tsmulti,xi=xi,xo=18)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+18)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+18)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamplesM(datasample=tsmulti,xi=BestInputNodes,xo=18)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+18)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     ttttrend<-CreateSamplesM(datasample=trendin,xi=BestInputNodes,xo=18)
#     tttseas<-CreateSamplesM(datasample=Decinsample,xi=BestInputNodes,xo=18)
#     finsample<-CreateSamplesM(datasample=insample,xi=BestInputNodes,xo=18)[(BestInputNodes+1):(BestInputNodes+18)]
#     ffitted<-InvBoxCox((modelsBPNN$fitted.values*(MAX-MIN)+MIN+ttttrend[(BestInputNodes+1):(BestInputNodes+18)])*tttseas[(BestInputNodes+1):(BestInputNodes+18)],lambda = lamda)
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #this contains both insample and outsample
#     tsmulti<-c((BoxCox(insample,lambda=lamda)/Decinsample)-trendin)
#
#     tempin<-t(as.matrix(tail(head(tsmulti,observations),BestInputNodes)))
#     tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#     MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#
#     MLPf<-InvBoxCox((MLPfs*(MAX-MIN)+MIN+trendout)*Decoutsample,lambda = lamda)
#
#     forecasts<-data.frame(MLPf)
#
#     #Make negative forecasts equal to zero
#     for (k in 1:18){
#       if(forecasts[k,1]<0) { forecasts[k,1]<-0 }
#     }
#
#     #Benchmark for MASE
#     forecastsNaiveSD<-rep(NA,frequency)
#     for (j in (frequency+1):observations){
#       forecastsNaiveSD<-c(forecastsNaiveSD,insample[j-frequency])
#     }
#     masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     Results$MASE[tsi]<-mean(abs(forecasts[,1]-outsample))/masep
#     Results$GoF[tsi]<-GoF
#
#     sMAPE18[tsi,]<-(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     MASE18[tsi,]<-abs(forecasts[,1]-outsample)/masep
#
#     #plot(c(insample,outsample),type="l")
#     #lines(c(insample,forecasts[,1]),col="red",type="l")
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(sMAPE18, file=paste("Results MLP Best multi-step sMAPE.csv"),row.names=FALSE)
# write.csv(MASE18, file=paste("Results MLP Best multi-step MASE.csv"),row.names=FALSE)
# write.csv(Results, file=paste("Results MLP Best multi-step.csv"),row.names=FALSE)

# print(paste0("ETS Started at ", date()))
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     tsmulti<-c(insample,outsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       if (i==1){
#         modeltype<-substring(gsub(",","", ets(tempin)$method),5)
#         modeltype<-substr(modeltype,1,nchar(modeltype)-1)
#         Damped<-FALSE
#         if (nchar(modeltype)>3){
#           Damped<-TRUE
#           modeltype<-gsub("d","", modeltype)
#         }
#       }
#
#       #forecast using t-s methods
#       model<-forecast(ets(tempin,model=modeltype,damped=Damped),h=1)
#       modelf<-model$mean
#
#       if (i==1){
#         GoF<-mean((model$fitted-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
#
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results ETS.csv"),row.names=FALSE)
# print(paste0("ETS Finished at ", date()))
#
# print(paste0("ETS BC Started at ", date()))
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
#
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
# startt<-Sys.time()
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#     tsmulti<-c(BoxCox(insample,lambda=lambda),BoxCox(outsample,lambda=lambda))
#
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       if (i==1){
#         modeltype<-substring(gsub(",","", ets(tempin)$method),5)
#         modeltype<-substr(modeltype,1,nchar(modeltype)-1)
#         Damped<-FALSE
#         if (nchar(modeltype)>3){
#           Damped<-TRUE
#           modeltype<-gsub("d","", modeltype)
#         }
#       }
#
#       #forecast using t-s methods
#       model<-forecast(ets(tempin,model=modeltype,damped=Damped),h=1)
#       modelf<-InvBoxCox(model$mean,lambda=lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox(model$fitted,lambda=lambda)-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results ETS BC.csv"),row.names=FALSE)
#
# print(paste0("ETS BC Finished at ", date()))

# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
#
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
# startt<-Sys.time()
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     tsmulti<-c(insample,outsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-forecast(auto.arima(tempin),h=1)
#       modelf<-model$mean
#
#       if (i==1){
#         GoF<-mean((model$fitted-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results ARIMA.csv"),row.names=FALSE)

print(paste0("ARIMA BC started at", date()))
CreateSamples<-function(datasample,xi){

  #Normalize insample from 0 to 1
  xo<-1
  ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
  sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
  for (cid in (xi+xo):length(datasample)){
    sample[cid,]<-datasample[(cid-xi-xo+1):cid]
  }
  sample<-as.matrix(data.frame(na.omit(sample)))

  return(sample)
}

frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)

#Tables for sMAPE and MASE TOTAL
Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
colnames(Results)<-c("sMAPE","MASE","Time","Gof")

startt<-Sys.time()

for (tsi in 1:1428){

  insample<-data[[tsi]]$x
  outsample<-data[[tsi]]$xx
  observations<-length(insample)

  if (observations>80){

    ttotrain<-Sys.time()

    #Tables for sMAPE and MASE rolling
    RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
    colnames(RollingResults)<-c("sMAPE","MASE")

    lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
    tsmulti=BoxCox(insample,lambda=lambda)
    tsmulti<-c(BoxCox(insample,lambda=lambda),BoxCox(outsample,lambda=lambda))

    for (i in 1:18){

      #Sample for rolling
      tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
      tempout<-outsample[i]

      #forecast using t-s methods
      model<-forecast(auto.arima(tempin),h=1)
      modelf<-InvBoxCox(model$mean,lambda=lambda)

      if (i==1){
        GoF<-mean((InvBoxCox(model$fitted,lambda=lambda)-as.numeric(insample))^2)*100/(mean(insample)^2)
      }

      forecasts<-data.frame(modelf)

      #Make negative forecasts equal to zero
      for (k in 1:ncol(forecasts)){
        if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
      }

      #Benchmark for MASE
      maseinsample<-head(c(insample,outsample),observations+i-1)
      forecastsNaiveSD<-rep(NA,frequency)
      for (j in (frequency+1):length(maseinsample)){
        forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
      }
      masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)

      #Estimate errors sMAPE and MASE
      RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
      RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep


    }

    #Save errors
    Results$Time[tsi]<-c(Sys.time()-ttotrain)
    Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
    Results$MASE[tsi]<-mean(RollingResults$MASE)
    Results$Gof[tsi]<-GoF

  }

}
what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
write.csv(Results, file=paste("Results ARIMA BC.csv"),row.names=FALSE)
print(paste0("ARIMA BC ended at", date()))

#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
#
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-ses(tempin,h=1)
#       modelf<-model$mean*Decoutsample[i]
#
#       if (i==1){
#         GoF<-mean((model$fitted*Decinsample-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results SES.csv"),row.names=FALSE)
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
#
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(BoxCox(insample,lambda=lambda)/Decinsample,BoxCox(outsample,lambda=lambda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-ses(tempin,h=1)
#       modelf<-InvBoxCox(model$mean*Decoutsample[i],lambda=lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox(model$fitted*Decinsample,lambda=lambda)-as.numeric(insample))^2,na.rm = TRUE)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results SES BC.csv"),row.names=FALSE)
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# startt<-Sys.time()
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
#
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-holt(tempin,h=1,damped=FALSE)
#       modelf<-model$mean*Decoutsample[i]
#
#       if (i==1){
#         GoF<-mean((model$fitted*Decinsample-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results holt.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(BoxCox(insample,lambda=lambda)/Decinsample,BoxCox(outsample,lambda=lambda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-holt(tempin,h=1,damped=FALSE)
#       modelf<-InvBoxCox(model$mean*Decoutsample[i],lambda=lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox(model$fitted*Decinsample,lambda=lambda)-as.numeric(insample))^2,na.rm = TRUE)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results holt BC.csv"),row.names=FALSE)

# print(paste0("Naive Started at ", date()))
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-naive(tempin,h=1)
#       modelf<-model$mean*Decoutsample[i]
#
#       if (i==1){
#         GoF<-mean((model$fitted*Decinsample-as.numeric(insample))^2,na.rm = TRUE)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results naive.csv"),row.names=FALSE)
#
# print(paste0("Naive Finished at ", date()))

# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(BoxCox(insample,lambda=lambda)/Decinsample,BoxCox(outsample,lambda=lambda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-naive(tempin,h=1)
#       modelf<-InvBoxCox(model$mean*Decoutsample[i],lambda=lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox(model$fitted*Decinsample,lambda=lambda)-as.numeric(insample))^2,na.rm = TRUE)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results naive BC.csv"),row.names=FALSE)
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-holt(tempin,h=1,damped=TRUE)
#       modelf<-model$mean*Decoutsample[i]
#
#       if (i==1){
#         GoF<-mean((model$fitted*Decinsample-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results damped.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(BoxCox(insample,lambda=lambda)/Decinsample,BoxCox(outsample,lambda=lambda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-holt(tempin,h=1,damped=TRUE)
#       modelf<-InvBoxCox(model$mean*Decoutsample[i],lambda=lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox(model$fitted*Decinsample,lambda=lambda)-as.numeric(insample))^2,na.rm = TRUE)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results damped BC.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# ThetaG <- function( input, fh){
#
#   outtest<-naive(input,h=fh)$mean
#   wses=wlrl<-1/2 ; theta=2
#
#   observations=length(input)
#   xs = c(1:observations)
#   xf = xff = c((observations+1):(observations+fh))
#   dat=data.frame(input=input,xs=xs)
#   newdf <- data.frame(xs = xff)
#
#   estimate <- lm(input ~ poly(xs, par=1, raw=TRUE))
#   thetaline0In  =predict(estimate)+input-input
#   thetaline0Out =predict(estimate,newdf)+outtest-outtest
#
#   thetaline2In  =ses(theta*input+(1-theta)*thetaline0In,h=fh)$fitted
#   thetaline2Out =ses(theta*input+(1-theta)*thetaline0In,h=fh)$mean
#
#   forecastsIn  = as.numeric(thetaline2In*wses)+as.numeric(thetaline0In*wlrl)+input-input
#   forecastsOut = as.numeric(thetaline2Out*wses)+as.numeric(thetaline0Out*wlrl)+outtest-outtest
#
#   output=list(fitted=forecastsIn,mean=forecastsOut,fitted0=thetaline0In,mean0=thetaline0Out,fitted2=thetaline2In,mean2=thetaline2Out,modelest=estimate)
#
#   return(output)
# }
#
# startt<-Sys.time()
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
#
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model1<-ses(tempin, h=1)
#       model2<-holt(tempin, h=1,damped = TRUE)
#       model3<-holt(tempin, h=1, damped=FALSE)
#
#       modelf<-(as.numeric(model1$mean)+as.numeric(model2$mean)+as.numeric(model3$mean))*Decoutsample[i]/3
#
#       if (i==1){
#         GoF<-mean((((as.numeric(model1$fitted)+as.numeric(model2$fitted)+as.numeric(model3$fitted))*Decinsample/3)-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results com.csv"),row.names=FALSE)
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# ThetaG <- function( input, fh){
#
#   outtest<-naive(input,h=fh)$mean
#   wses=wlrl<-1/2 ; theta=2
#
#   observations=length(input)
#   xs = c(1:observations)
#   xf = xff = c((observations+1):(observations+fh))
#   dat=data.frame(input=input,xs=xs)
#   newdf <- data.frame(xs = xff)
#
#   estimate <- lm(input ~ poly(xs, par=1, raw=TRUE))
#   thetaline0In  =predict(estimate)+input-input
#   thetaline0Out =predict(estimate,newdf)+outtest-outtest
#
#   thetaline2In  =ses(theta*input+(1-theta)*thetaline0In,h=fh)$fitted
#   thetaline2Out =ses(theta*input+(1-theta)*thetaline0In,h=fh)$mean
#
#   forecastsIn  = as.numeric(thetaline2In*wses)+as.numeric(thetaline0In*wlrl)+input-input
#   forecastsOut = as.numeric(thetaline2Out*wses)+as.numeric(thetaline0Out*wlrl)+outtest-outtest
#
#   output=list(fitted=forecastsIn,mean=forecastsOut,fitted0=thetaline0In,mean0=thetaline0Out,fitted2=thetaline2In,mean2=thetaline2Out,modelest=estimate)
#
#   return(output)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(BoxCox(insample,lambda=lambda)/Decinsample,BoxCox(outsample,lambda=lambda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model1<-ses(tempin, h=1)
#       model2<-holt(tempin, h=1,damped = TRUE)
#       model3<-holt(tempin, h=1, damped=FALSE)
#
#       modelf<-InvBoxCox((as.numeric(model1$mean)+as.numeric(model2$mean)+as.numeric(model3$mean))*Decoutsample[i]/3,lambda = lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox((as.numeric(model1$fitted)+as.numeric(model2$fitted)+as.numeric(model3$fitted))*Decinsample/3,lambda = lambda)-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results com BC.csv"),row.names=FALSE)
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# ThetaG <- function( input, fh){
#
#   outtest<-naive(input,h=fh)$mean
#   wses=wlrl<-1/2 ; theta=2
#
#   observations=length(input)
#   xs = c(1:observations)
#   xf = xff = c((observations+1):(observations+fh))
#   dat=data.frame(input=input,xs=xs)
#   newdf <- data.frame(xs = xff)
#
#   estimate <- lm(input ~ poly(xs, par=1, raw=TRUE))
#   thetaline0In  =predict(estimate)+input-input
#   thetaline0Out =predict(estimate,newdf)+outtest-outtest
#
#   thetaline2In  =ses(theta*input+(1-theta)*thetaline0In,h=fh)$fitted
#   thetaline2Out =ses(theta*input+(1-theta)*thetaline0In,h=fh)$mean
#
#   forecastsIn  = as.numeric(thetaline2In*wses)+as.numeric(thetaline0In*wlrl)+input-input
#   forecastsOut = as.numeric(thetaline2Out*wses)+as.numeric(thetaline0Out*wlrl)+outtest-outtest
#
#   output=list(fitted=forecastsIn,mean=forecastsOut,fitted0=thetaline0In,mean0=thetaline0Out,fitted2=thetaline2In,mean2=thetaline2Out,modelest=estimate)
#
#   return(output)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-ThetaG( input=tempin, fh=1)
#       modelf<-model$mean*Decoutsample[i]
#
#       if (i==1){
#         GoF<-mean((model$fitted*Decinsample-as.numeric(insample))^2)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results theta.csv"),row.names=FALSE)

# print(paste0("theta BC started at", date()))
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# ThetaG <- function( input, fh){
#
#   outtest<-naive(input,h=fh)$mean
#   wses=wlrl<-1/2 ; theta=2
#
#   observations=length(input)
#   xs = c(1:observations)
#   xf = xff = c((observations+1):(observations+fh))
#   dat=data.frame(input=input,xs=xs)
#   newdf <- data.frame(xs = xff)
#
#   estimate <- lm(input ~ poly(xs, par=1, raw=TRUE))
#   thetaline0In  =predict(estimate)+input-input
#   thetaline0Out =predict(estimate,newdf)+outtest-outtest
#
#   thetaline2In  =ses(theta*input+(1-theta)*thetaline0In,h=fh)$fitted
#   thetaline2Out =ses(theta*input+(1-theta)*thetaline0In,h=fh)$mean
#
#   forecastsIn  = as.numeric(thetaline2In*wses)+as.numeric(thetaline0In*wlrl)+input-input
#   forecastsOut = as.numeric(thetaline2Out*wses)+as.numeric(thetaline0Out*wlrl)+outtest-outtest
#
#   output=list(fitted=forecastsIn,mean=forecastsOut,fitted0=thetaline0In,mean0=thetaline0Out,fitted2=thetaline2In,mean2=thetaline2Out,modelest=estimate)
#
#   return(output)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#
#     lambda=BoxCox.lambda(insample,method="loglik",lower=0, upper=1)
#     tsmulti=BoxCox(insample,lambda=lambda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-c(BoxCox(insample,lambda=lambda)/Decinsample,BoxCox(outsample,lambda=lambda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-ts(tsmulti[1:(observations-1+i)],frequency = 12)
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       model<-ThetaG( input=tempin, fh=1)
#       modelf<-InvBoxCox(model$mean*Decoutsample[i],lambda=lambda)
#
#       if (i==1){
#         GoF<-mean((InvBoxCox(model$fitted*Decinsample,lambda=lambda)-as.numeric(insample))^2,na.rm = TRUE)*100/(mean(insample)^2)
#       }
#
#       forecasts<-data.frame(modelf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results theta BC.csv"),row.names=FALSE)
# print(paste0("theta BC ended at ", date()))

#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     if (p.value<0.01){
#       finsample<-InvBoxCox((as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#       ffitted<-InvBoxCox((modelsBPNN$fitted.values*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#     }else{
#       finsample<-InvBoxCox((as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#       ffitted<-InvBoxCox((modelsBPNN$fitted.values*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#     }
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     if (p.value<0.01){
#       tsmulti<-c((BoxCox(insample,lambda=lamda)/Decinsample)-trendin,(BoxCox(outsample,lambda=lamda)/Decoutsample)-trendout)
#     }else{
#       tsmulti<-c(BoxCox(insample,lambda=lamda)/Decinsample,BoxCox(outsample,lambda=lamda)/Decoutsample)
#     }
#
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       if (p.value<0.01){
#         MLPf<-InvBoxCox((MLPfs[1]*(MAX-MIN)+MIN+trendout[i])*Decoutsample[i],lambda = lamda)
#       }else{
#         MLPf<-InvBoxCox((MLPfs[1]*(MAX-MIN)+MIN)*Decoutsample[i],lambda = lamda)
#       }
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       if(forecasts[1,]<0) { forecasts[1,]<-0 }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,]-tempout)/(forecasts[1,]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP BCandDesandLRL.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#   GoF<-NA
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     finsample<-InvBoxCox(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN,lambda = lamda)
#     ffitted<-InvBoxCox(modelsBPNN$fitted.values*(MAX-MIN)+MIN,lambda = lamda)
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     tsmulti<-BoxCox(c(insample,outsample),lambda = lamda)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       MLPf<-InvBoxCox(MLPfs[1]*(MAX-MIN)+MIN,lambda = lamda)
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP Box-Cox.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-insample/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     if (p.value<0.01){
#       finsample<-(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations]
#       ffitted<-(modelsBPNN$fitted.values*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations]
#     }else{
#       finsample<-(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations]
#       ffitted<-(modelsBPNN$fitted.values*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations]
#
#     }
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     if (p.value<0.01){
#       tsmulti<-c((insample/Decinsample)-trendin,(outsample/Decoutsample)-trendout)
#     }else{
#       tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#     }
#
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       if (p.value<0.01){
#         MLPf<-(MLPfs[1]*(MAX-MIN)+MIN+trendout[i])*Decoutsample[i]
#       }else{
#         MLPf<-(MLPfs[1]*(MAX-MIN)+MIN)*Decoutsample[i]
#       }
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP DesandLRL.csv"),row.names=FALSE)
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-0
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#
#     finsample<-exp(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)
#     ffitted<-exp(modelsBPNN$fitted.values*(MAX-MIN)+MIN)
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     tsmulti<-BoxCox(c(insample,outsample),lambda = lamda)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       MLPf<-InvBoxCox(MLPfs[1]*(MAX-MIN)+MIN,lambda = lamda)
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP Log.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     p.value<-cox.stuart.test(insample)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(insample~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-insample-trendin
#     }else{
#       tsmulti<-insample
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     finsample<-(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)+trendin[(BestInputNodes+1):observations]
#     ffitted<-(modelsBPNN$fitted.values*(MAX-MIN)+MIN)+trendin[(BestInputNodes+1):observations]
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     tsmulti<-c(insample-trendin,outsample-trendout)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       MLPf<-MLPfs[1]*(MAX-MIN)+MIN+trendout[i]
#
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP LRL.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","Gof")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     tsmulti<-insample
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#
#     finsample<-as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN
#     ffitted<-modelsBPNN$fitted.values*(MAX-MIN)+MIN
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     tsmulti<-c(insample,outsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       MLPf<-(MLPfs[1]*(MAX-MIN)+MIN)
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$Gof[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP.csv"),row.names=FALSE)
#
#
# print(paste0("MLP Started at ", date()))
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#
#     finsample<-InvBoxCox((as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#     ffitted<-InvBoxCox((modelsBPNN$fitted.values*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     tsmulti<-c(BoxCox(insample,lambda = lamda)/Decinsample,BoxCox(outsample,lambda = lamda)/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       MLPf<-InvBoxCox((MLPfs[1]*(MAX-MIN)+MIN)*Decoutsample[i],lambda = lamda)
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP BCandDes.csv"),row.names=FALSE)
#
# print(paste0("MLP Finished at ", date()))
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     if (p.value<0.01){
#       finsample<-InvBoxCox(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations],lambda = lamda)
#       ffitted<-InvBoxCox(modelsBPNN$fitted.values*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations],lambda = lamda)
#     }else{
#       finsample<-InvBoxCox(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN,lambda = lamda)
#       ffitted<-InvBoxCox(modelsBPNN$fitted.values*(MAX-MIN)+MIN,lambda = lamda)
#     }
#
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     if (p.value<0.01){
#       tsmulti<-c(BoxCox(insample,lambda = lamda)-trendin,BoxCox(outsample,lambda = lamda)-trendout)
#     }else{
#       tsmulti<-BoxCox(c(insample,outsample),lambda = lamda)
#     }
#
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       if (p.value<0.01){
#         MLPf<-InvBoxCox(MLPfs[1]*(MAX-MIN)+MIN+trendout[i],lambda=lamda)
#       }else{
#         MLPf<-InvBoxCox(MLPfs[1]*(MAX-MIN)+MIN,lambda=lamda)
#       }
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP BCandLRL.csv"),row.names=FALSE)
#
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     DecModel<-SeasonalDec(insample=insample,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-insample/Decinsample
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#
#     finsample<-(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations]
#     ffitted<-(modelsBPNN$fitted.values*(MAX-MIN)+MIN)*Decinsample[(BestInputNodes+1):observations]
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     tsmulti<-c(insample/Decinsample,outsample/Decoutsample)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       MLPf<-(MLPfs[1]*(MAX-MIN)+MIN)*Decoutsample[i]
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP Des.csv"),row.names=FALSE)
#
#
#
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#   GoF<-NA
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     p.value<-cox.stuart.test(insample)$p.value
#
#     if (p.value<0.01){
#       tsmulti<-diff(insample)
#     }else{
#       tsmulti<-insample
#     }
#
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#         model<-mlp(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                    size = (2*xi+1), maxit = 500,initFunc = "Randomize_Weights",
#                    learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                    shufflePatterns = FALSE, linOut = TRUE)
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#     modelsBPNN<-mlp(as.matrix(samplegenerate[,1:BestInputNodes]), as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)]),
#                     size = (2*BestInputNodes+1), maxit = 500,initFunc = "Randomize_Weights",
#                     learnFunc = "SCG", hiddenActFunc = "Act_Logistic",
#                     shufflePatterns = FALSE, linOut = TRUE)
#
#     if (p.value<0.01){
#       finsample<-(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)+insample[(BestInputNodes+1):(observations-1)]
#       ffitted<-(modelsBPNN$fitted.values*(MAX-MIN)+MIN)+insample[(BestInputNodes+1):(observations-1)]
#     }else{
#       finsample<-(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN)
#       ffitted<-(modelsBPNN$fitted.values*(MAX-MIN)+MIN)
#
#     }
#
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#     #Tables for sMAPE and MASE rolling
#     RollingResults<-data.frame(matrix(NA,ncol=2,nrow=18))
#     colnames(RollingResults)<-c("sMAPE","MASE")
#     #this contains both insample and outsample
#     if (p.value<0.01){
#       tsmulti<-diff(c(insample,outsample))
#     }else{
#       tsmulti<-c(insample,outsample)
#     }
#
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#       tempout<-outsample[i]
#
#       #forecast using t-s methods
#       MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#       if (p.value<0.01){
#         MLPf<-MLPfs[1]*(MAX-MIN)+MIN+c(insample,outsample)[observations+i-1]
#       }else{
#         MLPf<-MLPfs[1]*(MAX-MIN)+MIN
#       }
#
#
#       forecasts<-data.frame(MLPf)
#
#       #Make negative forecasts equal to zero
#       for (k in 1:ncol(forecasts)){
#         if(forecasts[1,k]<0) { forecasts[1,k]<-0 }
#       }
#
#       #Benchmark for MASE
#       maseinsample<-head(c(insample,outsample),observations+i-1)
#       forecastsNaiveSD<-rep(NA,frequency)
#       for (j in (frequency+1):length(maseinsample)){
#         forecastsNaiveSD<-c(forecastsNaiveSD,maseinsample[j-frequency])
#       }
#       masep<-mean(abs(maseinsample-forecastsNaiveSD),na.rm = TRUE)
#
#       #Estimate errors sMAPE and MASE
#       RollingResults$sMAPE[i]<-mean(200*abs(forecasts[1,k]-tempout)/(forecasts[1,k]+tempout))
#       RollingResults$MASE[i]<-mean(abs(forecasts[1,k]-tempout))/masep
#
#
#     }
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(RollingResults$sMAPE)
#     Results$MASE[tsi]<-mean(RollingResults$MASE)
#     Results$GoF[tsi]<-GoF
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(Results, file=paste("Results MLP DIF test.csv"),row.names=FALSE)
#
#
#
#
# print(paste0("BNN interatial started at ", date()))
# CreateSamples<-function(datasample,xi){
#
#   #Normalize insample from 0 to 1
#   xo<-1
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
# sMAPE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
# MASE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamples(datasample=tsmulti,xi=xi)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#
#         found<-100
#         while (length(found)==1){
#           found<-tryCatch(model<-brnn(as.matrix(KfoldsIn[[TestFolds]][,1:xi]), as.numeric(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)]),
#                                       neurons=(2*xi+1),normalize=FALSE,epochs=500,verbose=F), error=function(e) 100)
#         }
#
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamples(datasample=tsmulti,xi=BestInputNodes)
#
#
#     found<-100
#     while (length(found)==1){
#       found<-tryCatch(modelsBPNN<-brnn(as.matrix(KfoldsIn[[TestFolds]][,1:BestInputNodes]), as.numeric(KfoldsIn[[TestFolds]][,(BestInputNodes+1):(BestInputNodes+1)]),
#                                   neurons=(2*BestInputNodes+1),normalize=FALSE,epochs=500,verbose=F), error=function(e) 100)
#     }
#
#     finsample<-InvBoxCox((as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+1)])*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#     ffitted<-InvBoxCox((predict(modelsBPNN)*(MAX-MIN)+MIN+trendin[(BestInputNodes+1):observations])*Decinsample[(BestInputNodes+1):observations],lambda = lamda)
#
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #this contains both insample and outsample
#     tsmulti<-(c((BoxCox(insample,lambda=lamda)/Decinsample)-trendin)-MIN)/(MAX-MIN)
#
#     MLPfs=MLPf<-c()
#     tsmulti<-c(tsmulti,MLPfs)
#
#     for (i in 1:18){
#
#       #Sample for rolling
#       tempin<-t(as.matrix(tail(head(tsmulti,observations+i-1),BestInputNodes)))
#       #tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#
#       #forecast using t-s methods
#       MLPfs<-c(MLPfs,as.numeric(predict(modelsBPNN,tempin)))
#       tsmulti<-c(tsmulti,MLPfs[length(MLPfs)])
#
#     }
#
#     MLPf<-InvBoxCox((MLPfs*(MAX-MIN)+MIN+trendout)*Decoutsample,lambda = lamda)
#
#     forecasts<-data.frame(MLPf)
#
#     #Make negative forecasts equal to zero
#     for (k in 1:18){
#       if(forecasts[k,1]<0) { forecasts[k,1]<-0 }
#     }
#
#     #Benchmark for MASE
#     forecastsNaiveSD<-rep(NA,frequency)
#     for (j in (frequency+1):observations){
#       forecastsNaiveSD<-c(forecastsNaiveSD,insample[j-frequency])
#     }
#     masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     Results$MASE[tsi]<-mean(abs(forecasts[,1]-outsample))/masep
#     Results$GoF[tsi]<-GoF
#
#     sMAPE18[tsi,]<-(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     MASE18[tsi,]<-abs(forecasts[,1]-outsample)/masep
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(sMAPE18, file=paste("Results BNN Best interatial sMAPE.csv"),row.names=FALSE)
# write.csv(MASE18, file=paste("Results BNN Best interatial MASE.csv"),row.names=FALSE)
# write.csv(Results, file=paste("Results BNN Best interatial.csv"),row.names=FALSE)
# print(paste0("BNN interatial ended at ", date()))

# CreateSamplesH<-function(datasample,xi,xo){
#
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
# sMAPE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
# MASE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#
#     BestInputs<-c()
#     for (NNid in 1:18){
#
#       SSEinputnodes<-c()
#       for (xi in 1:5){
#
#         #create samples
#         samplegenerate<-CreateSamplesH(datasample=tsmulti,xi=xi,xo=NNid)[,c(1:xi,xi+NNid)]
#         #create 10 folds
#         foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#         start<-1 ; end<-foldlength
#         for (fid in 1:9){
#           Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#           start<-start+foldlength ; end<-end+foldlength
#         }
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#         KfoldsIn=KfoldsOut<-NULL
#
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#         KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#         KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#         SSE<-0
#         for (TestFolds in 1:10){
#
#           found<-100
#           while (length(found)<=1){
#             found<-tryCatch(model<-brnn((as.matrix(KfoldsIn[[TestFolds]][,1:xi])),
#                                         as.numeric(as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+1)])),
#                                         neurons=(2*xi+1),normalize=FALSE,epochs=500,verbose=F), error=function(e) 100)
#           }
#
#           for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#           SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+1)])^2)
#         }
#         SSEinputnodes<-c(SSEinputnodes,SSE)
#
#       }
#       BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#
#
#       BestInputs<-c(BestInputs,BestInputNodes)
#
#
#     }
#
#
#     #These are the models for each fh
#     modelsBPNN<-NULL ; GoF<-0
#     for (NNid in 1:18){
#
#       samplegenerate<-CreateSamplesH(datasample=tsmulti,xi=BestInputs[NNid],xo=NNid)
#
#
#
#       found<-100
#       while (length(found)<=1){
#         found<-tryCatch(kkkk<-brnn(as.matrix(samplegenerate[,1:BestInputs[NNid]]),
#                                    as.numeric(as.matrix(samplegenerate[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)])),
#                                    neurons=(2*BestInputs[NNid]+1),normalize=FALSE,epochs=500,verbose=F), error=function(e) 100)
#       }
#       modelsBPNN[length(modelsBPNN)+1]<-list(kkkk)
#
#       ttttrend<-CreateSamplesH(datasample=trendin,xi=BestInputs[NNid],xo=NNid)[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]
#       tttseas<-CreateSamplesH(datasample=Decinsample,xi=BestInputs[NNid],xo=NNid)[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]
#       finsample<-CreateSamplesH(datasample=insample,xi=BestInputs[NNid],xo=NNid)[,(BestInputs[NNid]+NNid):(BestInputs[NNid]+NNid)]
#       ffitted<-InvBoxCox((modelsBPNN[[length(modelsBPNN)]]$fitted.values*(MAX-MIN)+MIN+ttttrend)*tttseas,lambda = lamda)
#
#       GoF<-GoF+(mean((finsample-ffitted)^2)*100/(mean(finsample)^2))
#
#     }
#     GoF<-GoF/18
#
#     #Generate forecasts
#     MLPfs<-c()
#     for (i in 1:18){
#
#       tempin<-t(as.matrix(tail(tsmulti,BestInputs[i])))
#       MLPfs<-c(MLPfs,as.numeric(predict(modelsBPNN[[i]],tempin)))
#
#     }
#     MLPf<-InvBoxCox((MLPfs*(MAX-MIN)+MIN+trendout)*Decoutsample,lambda = lamda)
#
#     forecasts<-data.frame(MLPf)
#
#     #Make negative forecasts equal to zero
#     #Make negative forecasts equal to zero
#     for (k in 1:18){
#       if(forecasts[k,1]<0) { forecasts[k,1]<-0 }
#     }
#
#     #Benchmark for MASE
#     forecastsNaiveSD<-rep(NA,frequency)
#     for (j in (frequency+1):observations){
#       forecastsNaiveSD<-c(forecastsNaiveSD,insample[j-frequency])
#     }
#     masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     Results$MASE[tsi]<-mean(abs(forecasts[,1]-outsample))/masep
#     Results$GoF[tsi]<-0
#
#     sMAPE18[tsi,]<-(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     MASE18[tsi,]<-abs(forecasts[,1]-outsample)/masep
#
#     #plot(c(insample,outsample),type="l")
#     #lines(c(insample,forecasts[,1]),col="red",type="l")
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(sMAPE18, file=paste("Results BNN Best multi NNs sMAPE.csv"),row.names=FALSE)
# write.csv(MASE18, file=paste("Results BNN Best multi NNs MASE.csv"),row.names=FALSE)
# write.csv(Results, file=paste("Results BNN Best multi NNs.csv"),row.names=FALSE)
# ###
#
#
#
# CreateSamplesM<-function(datasample,xi,xo){
#
#   #Normalize insample from 0 to 1
#
#   ####  ####  ####  ####  ####  Create data set ####  ####  ####  #### ####  ####  ####  ####
#   sample<-matrix(NA,nrow=length(datasample),ncol=(xi+xo)) #all possible n-samples
#   for (cid in (xi+xo):length(datasample)){
#     sample[cid,]<-datasample[(cid-xi-xo+1):cid]
#   }
#   sample<-as.matrix(data.frame(na.omit(sample)))
#
#   return(sample)
# }
#
# frequency = 12; descr = "M3-Monthly"; data = subset(M3, frequency)
# startt<-Sys.time()
# #Tables for sMAPE and MASE TOTAL
# Results<-data.frame(matrix(NA,ncol=4,nrow=1428))
# colnames(Results)<-c("sMAPE","MASE","Time","GoF")
#
# sMAPE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
# MASE18<-data.frame(matrix(NA,ncol=18,nrow=1428))
#
#
# for (tsi in 1:1428){
#
#   insample<-data[[tsi]]$x
#   outsample<-data[[tsi]]$xx
#   observations<-length(insample)
#
#
#   if (observations>80){
#
#     ttotrain<-Sys.time()
#
#     lamda<-BoxCox.lambda(insample, lower=0, upper=1)
#     tsmulti<-BoxCox(insample,lambda=lamda)
#
#     DecModel<-SeasonalDec(insample=tsmulti,horizon=18,frequency=12)
#     Decinsample<-DecModel[[1]]
#     Decoutsample<-DecModel[[2]]
#
#     tsmulti<-tsmulti/Decinsample
#
#     p.value<-cox.stuart.test(tsmulti)$p.value
#
#     if (p.value<0.01){
#       trendmodel<-lm(tsmulti~c(1:observations))
#       trendin<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c(1:observations))
#       trendout<-as.numeric(coef(trendmodel)[1]+coef(trendmodel)[2]*c((observations+1):(observations+18)))
#       tsmulti<-tsmulti-trendin
#     }else{
#       tsmulti<-tsmulti
#       trendin<-rep(0,observations)
#       trendout<-rep(0,18)
#     }
#
#
#     MAX<-max(tsmulti) ; MIN<-min(tsmulti)
#     tsmulti<-(tsmulti-MIN)/(MAX-MIN)
#
#     #test multiple input nodes and find the optimal using K-fold cross-validation
#     SSEinputnodes<-c()
#
#     for (xi in 1:5){
#
#       #create samples
#       samplegenerate<-CreateSamplesM(datasample=tsmulti,xi=xi,xo=18)
#       #create 10 folds
#       foldlength<-floor(nrow(samplegenerate)/10) ; Kfolds<-NULL
#       start<-1 ; end<-foldlength
#       for (fid in 1:9){
#         Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:end,])
#         start<-start+foldlength ; end<-end+foldlength
#       }
#       Kfolds[length(Kfolds)+1]<-list(samplegenerate[start:nrow(samplegenerate),])
#
#       KfoldsIn=KfoldsOut<-NULL
#
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[10]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[9]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[8]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[7]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[6]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[5]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[3]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[4]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[2]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[3]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[1]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[2]])
#       KfoldsIn[length(KfoldsIn)+1]<-list(rbind(Kfolds[[2]],Kfolds[[3]],Kfolds[[4]],Kfolds[[5]],Kfolds[[6]],Kfolds[[7]],Kfolds[[8]],Kfolds[[9]],Kfolds[[10]]))
#       KfoldsOut[length(KfoldsOut)+1]<-list(Kfolds[[1]])
#
#       SSE<-0
#       for (TestFolds in 1:10){
#
#         found<-100
#         while (length(found)<=1){
#           found<-tryCatch(model<-brnn(as.matrix(KfoldsIn[[TestFolds]][,1:xi]),
#                                       as.matrix(KfoldsIn[[TestFolds]][,(xi+1):(xi+18)]),
#                                       neurons=(2*xi+1),normalize=FALSE,epochs=500,verbose=F), error=function(e) 100)
#         }
#
#         for.model<-predict(model,as.matrix(KfoldsOut[[TestFolds]][,1:xi]))
#         SSE<-SSE+sum((for.model-KfoldsOut[[TestFolds]][,(xi+1):(xi+18)])^2)
#       }
#       SSEinputnodes<-c(SSEinputnodes,SSE)
#
#     }
#
#     BestInputNodes<-which.min(SSEinputnodes) #best length of input nodes
#     samplegenerate<-CreateSamplesM(datasample=tsmulti,xi=BestInputNodes,xo=18)
#
#
#     found<-100
#     while (length(found)<=1){
#       found<-tryCatch(modelsBPNN<-brnn((as.matrix(samplegenerate[,1:BestInputNodes])),
#                                   as.numeric(as.matrix(samplegenerate[,(BestInputNodes+1):(BestInputNodes+18)])),
#                                   neurons=(2*xi+1),normalize=FALSE,epochs=500,verbose=F), error=function(e) 100)
#     }
#
#
#     ttttrend<-CreateSamplesM(datasample=trendin,xi=BestInputNodes,xo=18)
#     tttseas<-CreateSamplesM(datasample=Decinsample,xi=BestInputNodes,xo=18)
#     finsample<-CreateSamplesM(datasample=insample,xi=BestInputNodes,xo=18)[(BestInputNodes+1):(BestInputNodes+18)]
#     ffitted<-InvBoxCox((modelsBPNN$fitted.values*(MAX-MIN)+MIN+ttttrend[(BestInputNodes+1):(BestInputNodes+18)])*tttseas[(BestInputNodes+1):(BestInputNodes+18)],lambda = lamda)
#     GoF<-mean((finsample-ffitted)^2)*100/(mean(finsample)^2)
#
#
#     #this contains both insample and outsample
#     tsmulti<-c((BoxCox(insample,lambda=lamda)/Decinsample)-trendin)
#
#     tempin<-t(as.matrix(tail(head(tsmulti,observations),BestInputNodes)))
#     tempin<-as.matrix(data.frame((tempin-MIN)/(MAX-MIN)) )
#     MLPfs<-as.numeric(predict(modelsBPNN,tempin))
#
#     MLPf<-InvBoxCox((MLPfs*(MAX-MIN)+MIN+trendout)*Decoutsample,lambda = lamda)
#
#     forecasts<-data.frame(MLPf)
#
#     #Make negative forecasts equal to zero
#     for (k in 1:18){
#       if(forecasts[k,1]<0) { forecasts[k,1]<-0 }
#     }
#
#     #Benchmark for MASE
#     forecastsNaiveSD<-rep(NA,frequency)
#     for (j in (frequency+1):observations){
#       forecastsNaiveSD<-c(forecastsNaiveSD,insample[j-frequency])
#     }
#     masep<-mean(abs(insample-forecastsNaiveSD),na.rm = TRUE)
#
#     #Save errors
#     Results$Time[tsi]<-c(Sys.time()-ttotrain)
#     Results$sMAPE[tsi]<-mean(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     Results$MASE[tsi]<-mean(abs(forecasts[,1]-outsample))/masep
#     Results$GoF[tsi]<-GoF
#
#     sMAPE18[tsi,]<-(200*abs(forecasts[,1]-outsample)/(forecasts[,1]+outsample))
#     MASE18[tsi,]<-abs(forecasts[,1]-outsample)/masep
#
#     #plot(c(insample,outsample),type="l")
#     #lines(c(insample,forecasts[,1]),col="red",type="l")
#
#   }
#
# }
# what<-c(what,as.numeric(Sys.time()-startt,units="secs")/1045)
# write.csv(sMAPE18, file=paste("Results BNN Best multi-step sMAPE.csv"),row.names=FALSE)
# write.csv(MASE18, file=paste("Results BNN Best multi-step MASE.csv"),row.names=FALSE)
# write.csv(Results, file=paste("Results BNN Best multi-step.csv"),row.names=FALSE)

