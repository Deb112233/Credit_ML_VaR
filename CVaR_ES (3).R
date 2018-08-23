#=== load R libraries ==#
library(curl)
library(tidyr)
library(dplyr)
library(lattice)
library(pROC)
library(ROCR)
library(randomForest)
library(caret)
library(gbm)
library(reshape2)
library(ggplot2)
#=== Assume a credit risk portfolio (same size with test_set) ==#
#== Value-at-Risk (VaR) is to model portfolio's credit risk exposure (assume one year) ==#
#== below is a Monte Carlo simulation based VaR model ===#
set.seed(123)
credit <- loan_data_clean[sample(1:nrow(loan_data_clean), 
                                 round(0.2*nrow(loan_data_clean))), ]
n <- nrow(credit)
runs <- 100
#=== Key parameters in VaR model ===#
#== Exposure at Default(EAD)==#
EAD <- credit$amount
#== Loss Given Default(LGD) ==#
LGD <- sample(seq(0.4, 0.45, by = 0.001),replace=T, nrow(credit))
#== Probability of Default(PD) ==#
PD <- ifelse(credit$default=="yes", 1,0)
#== asset/loan correlation parameter(rho) ==#
rho <- sum(ifelse(credit$default=="yes", 1,0))/nrow(credit)
#=== Set lossrealisation function ===#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  # keep track of the loss in this portfolio
  totalloss <- 0
  # draw a normal random variable for the systematic factor
  sf <- rnorm(1)
  # loop through all obligors to see if they default
  for(obligor in 1:n){
    # draw specific factor
    of <- rnorm(1)
    # asset value for this obligor
    x <- sqrt(rho)*sf + sqrt(1-rho^2)*of
    # critical threshold for this obligor
    c <- qnorm(0.95, mean(PD), sd(PD))
    # check for default
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

#== VaR model based on Monte Carlo simulation ==#
losses <- c() 
for(run in 1:runs){
  # add a new realisation of the loss variable
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}
#== VaR confidence level: normally use 95% for internal VaR model ==#
alpha <- 0.95
#== Sort losses and select the maximum losses at alpha ==#
losses <- sort(losses)
j <- floor(alpha*runs)
var_actual <- losses[j]

#=== Expected Shortfall (ES) of credit portfolio ==#
# Select the losses that are larger than VaR
largelosses <- losses[losses >= var_actual]
ES_actual <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_actual <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_actual", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR based on predictions by different models ==#
#=== VaR by Logistic model ===#
pred_model <- pred_logit
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(rho)*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_logit <- losses[j]
largelosses <- losses[losses >= var_logit]
ES_logit <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_logit <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_logit", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR by Probit regression model ==#
pred_model <- pred_probit
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all VaR models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(rho)*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_probit <- losses[j]
largelosses <- losses[losses >= var_probit]
ES_probit <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_probit <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_probit", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR by Random Forest model ==#
pred_model <- ifelse(pred_rf=="yes", 1, 0)
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all VaR models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(rho)*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_rf <- losses[j]
largelosses <- losses[losses >= var_rf]
ES_rf <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_rf <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_rf", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR by Tuned Random Forest model ==#
pred_model <- ifelse(pred_rf_tune=="yes", 1, 0)
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all VaR models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(rho)*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_rf_tune <- losses[j]
largelosses <- losses[losses >= var_rf_tune]
ES_rf_tune <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_rf_tune <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_rf_tuned", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR by GBM model ==#
pred_model <- pred_gbm
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all VaR models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(rho)*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_gbm <- losses[j]
largelosses <- losses[losses >= var_gbm]
ES_gbm <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_gbm <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_gbm", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR by GBM_OOB model ==#
pred_model <- pred_oob_gbm
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all VaR models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(abs(rho))*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_oob_gbm <- losses[j]
largelosses <- losses[losses >= var_oob_gbm]
ES_oob_gbm <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_oob_gbm <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_gbm(oob)", xlab ="Credit default loss ", ylab =" Density ")

#=== VaR by GBM_CV model ==#
pred_model <- pred_cv_gbm
PD <- pred_model
rho <- sum(pred_model)/length(pred_model)
#== Parameters: n, runs, EAD & LGD are the same across all VaR models ==#
lossrealisation<-function (n,PD,EAD,LGD,rho){
  totalloss <- 0
  sf <- rnorm(1)
  for(obligor in 1:n){
    of <- rnorm(1)
    x <- sqrt(abs(rho))*sf + sqrt(1-rho^2)*of
    c <- qnorm(0.95, mean(PD), sd(PD))
    if(x < c){
      totalloss <- totalloss + EAD[obligor] * LGD[obligor];
    }
  }
  return(totalloss)
}

losses <- c() 
for(run in 1:runs){
  losses <- c(losses,lossrealisation(n,PD,EAD,LGD,rho))
}

losses <- sort(losses)
j <- floor(alpha*runs)
var_cv_gbm <- losses[j]
largelosses <- losses[losses >= var_cv_gbm]
ES_cv_gbm <- mean (largelosses)
#== Histogram of loss distribution ==#
hist_cv_gbm <- hist(losses, freq =FALSE , main =" Histogram of credit default loss_gbm(cv)", xlab ="Credit default loss ", ylab =" Density ")

#== VaR forecast errors: deviationS from actual portfolio's VaR ==#
forc_err_logit <- sqrt(abs(var_logit-var_actual))
forc_err_probit <- sqrt(abs(var_probit-var_actual))
forc_err_rf <- sqrt(abs(var_rf-var_actual))
forc_err_rf_tune <- sqrt(abs(var_rf_tune-var_actual))
forc_err_gbm <- sqrt(abs(var_gbm-var_actual))
forc_err_gbm_oob <- sqrt(abs(var_oob_gbm-var_actual))
forc_err_gbm_cv <- sqrt(abs(var_cv_gbm-var_actual))
#== rank models' VaR by forecast errors ==#
forc_list <- data.frame(Models=c("Logit Regression", "Probit Regression",
                                "Random Forest", "Tune Random Forest", "GBM",
                                "GBM_OOB", "GBM_CV"), 
                       forc_errors=c(forc_err_logit, forc_err_probit,
                                     forc_err_rf, forc_err_rf_tune, 
                                     forc_err_gbm, forc_err_gbm_oob, forc_err_gbm_cv))
var_es <- data.frame(VaR=c(var_logit, var_probit, var_rf, var_rf_tune, 
                           var_gbm, var_oob_gbm, var_cv_gbm),
                     ES=c(ES_logit, ES_probit, ES_rf, ES_rf_tune,
                          ES_gbm, ES_oob_gbm, ES_cv_gbm))
var_es_err <- cbind(forc_list, var_es)
rank_var <- var_es_err[order(var_es_err$forc_errors), ]
write.table(rank_var, file = "C:/Users/siqiwen/Desktop/Credit/rank_var.csv",  row.names=FALSE, sep = ",",  qmethod = "double")

#== plot models' forecast errors by ranking ==#
ggplot(rank_var, aes(x=reorder(Models, forc_errors), y=forc_errors))+
  geom_bar(stat = "identity", fill="darkred", width = .5)+
  labs(title="VaR forecast erros by Models")+
  theme(axis.text.x = element_text(angle=65, vjust = 0.6))
#== plot VaR & ES of all models ==#
var_act <- data.frame(Models="Actual Portfolio VaR", forc_errors=0, 
                      VaR=var_actual, ES=ES_actual)
rank_var <- rbind(var_act, rank_var)
rank_var_long <- melt(rank_var[, -2])
theme_set((theme_bw()))
ggplot(rank_var_long, aes(x=reorder(Models, value), y=value, fill=variable))+
  geom_bar(stat = "identity", position="dodge", width = .5)+
  labs(title="All Models' VaR & ES")+
  theme(axis.text.x = element_text(angle=65, vjust = 0.6))

