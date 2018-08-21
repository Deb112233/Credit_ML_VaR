
#=== create training & testing datasets ===#
set.seed(123)
train_index <- sample(1:nrow(loan_data_clean), round(0.8*nrow(loan_data_clean)))
train_set <- loan_data_clean[train_index, ] # create training dataset
test_set <- loan_data_clean[-train_index, ]
rm(train_index)

#=== Logistic Regression Model ===#
log_full <- glm(default ~., family = binomial(link = logit), data=train_set)
summary(log_full) 
#== re-model logistic according to Pr(>|Z|) of variables in log_full ===#
log_logit<- glm(default ~ credit_history + emp_cat + age + housing + months_loan_duration, family = binomial(link = logit), data = train_set)
#=== Probit regression model is another option for credit default analysis ==#
log_probit <- glm(default ~ credit_history + emp_cat + age + housing + months_loan_duration, family = binomial(link = probit), data=train_set)

#== Predictions by Logistic & probit regression models ==#
pred_full <- predict(log_full, newdata = test_set, type = "response")
pred_logit <- predict(log_logit, newdata = test_set, type = "response")
pred_probit <- predict(log_probit, newdata = test_set, type = "response")
#== check the ranges of predictions ==#
range(pred_full) 
range(pred_logit)
range(pred_probit)

#== Explore the optimal cutoff level based on model accuracy ==#
cutoffs <- seq(0.1,0.9,0.1)
accuracy <- NULL
for (i in seq(along = cutoffs)){
  prediction <- ifelse(log_full$fitted.values >= cutoffs[i], 1, 0)
  accuracy <- c(accuracy,length(which(ifelse(train_set$default=="yes", 1, 0) == prediction))/length(prediction)*100)
}
plot(cutoffs, accuracy, pch =19,type='b',col= "steelblue",
     main ="Logistic Regression", xlab="Cutoff Level", ylab = "Accuracy %")
#=== Re-model Logistic & Probit models with cutoff ===#
#=== optimal cutoff level is 0.4 according to cutoff-accuracy graph ==#
pred_full_cutoff <- pred_full
pred_logit_cutoff <- pred_logit
pred_probit_cutoff <- pred_probit
pred_list <- cbind(pred_full_cutoff, pred_logit_cutoff, pred_probit_cutoff)
for (n in colnames(pred_list)) {
  cutoff_model <- quantile(pred_list[, n], 0.6)
  pred_cutoff <- ifelse(pred_list[, n]>cutoff_model, 1, 0)
  assign(n, pred_cutoff)
}

#=== Model Validations by bad_rate & ROC ===#
#=== compare predictions with actual default records ===#
#=== measure by loan acceptance rate & bad rate (False Negative) ==#
accepted_full <- cbind(actual_default=ifelse(test_set$default=="yes", 1, 0), 
                       pred_full_cutoff)[pred_full_cutoff==0,1]
bad_rate_full <- sum(accepted_full)/length(accepted_full)
accepted_logit <- cbind(actual_default=ifelse(test_set$default=="yes", 1, 0), 
                        pred_logit_cutoff)[pred_logit_cutoff==0,1]
bad_rate_logit <- sum(accepted_logit)/length(accepted_logit)
accepted_probit <- cbind(actual_default=ifelse(test_set$default=="yes", 1, 0), 
                         pred_probit_cutoff)[pred_probit_cutoff==0,1]
bad_rate_probit <- sum(accepted_probit)/length(accepted_probit)
print(bad_rate_full) 
print(bad_rate_logit)
print(bad_rate_probit)

#=== ROC curve ===#
library(pROC)
library(ROCR)
actual <- test_set$default
auc_full <- auc(actual, pred_full)
auc_logit <- auc(actual, pred_logit)
auc_probit <- auc(actual, pred_probit)
print(paste0("Test set AUC (Logit_full): ", auc_full))
print(paste0("Test set AUC (Logit): ", auc_logit))                         
print(paste0("Test set AUC (Probit): ", auc_probit))

#=== Train a random forest model ===#
library(randomForest)
set.seed(123) 
rf_model <- randomForest(formula = default~., data = train_set)
#===  check rf model result and variables' importance ===#
print(rf_model)
rf_model$importance
#=== OOB error matrix ===#
err <- rf_model$err.rate
oob_err <- err[nrow(err), "OOB"]
plot(rf_model)
legend(x = "right", cex = .65,
       legend = colnames(err),
       fill = 1:ncol(err))
pred_rf <- predict(rf_model, newdata = test_set, type = "class")

#=== Tuning Random Forest model ===#
#== Establish a set of hyperparameters to minimise OOB errors ===#
mtry <- seq(4, ncol(train_set)*0.8, 2)
nodesize <- seq(3,8,2)
sampsize <- nrow(train_set) * c(0.7, 0.8)
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, 
                          sampsize = sampsize)
oob_err_vec <- c()
for (i in 1:nrow(hyper_grid)) {
  # train a rf model
  model <- randomForest(formula = default ~ ., 
                        data = train_set,
                        mtry = hyper_grid$mtry[i],
                        nodesize = hyper_grid$nodesize[i],
                        sampsize = hyper_grid$sampsize[i])
  # store OOB errors 
  oob_err_vec[i] <- model$err.rate[nrow(model$err.rate), "OOB"]
}
#== the optimal set of hyperparameters based on OOB error ==#
best_i <- which.min(oob_err_vec)
opt_hyper <- hyper_grid[best_i,]
print(opt_hyper)
# Best RF model based on the optimal set of hyperparameters
rf_tune <- randomForest(formula=default~., data=train_set, 
                        mtry=opt_hyper[,1], 
                        nodesize=opt_hyper[,2],
                        sampsize=opt_hyper[,3])
err_tune <- rf_tune$err.rate
oob_err_tune <- err_tune[nrow(err_tune), "OOB"]
pred_rf_tune <- predict(rf_tune, newdata = test_set, type = "class")

#=== RF models performance ===#
#== compare predictions with actual default records ===#
library(caret)
conf_matrix_rf <- confusionMatrix(data = pred_rf,
                                  reference = test_set$default)
conf_matrix_rf_tune <- confusionMatrix(data = pred_rf_tune,
                                  reference = test_set$default)
print(conf_matrix_rf)
print(conf_matrix_rf_tune)
#=== RF model's test accuracy & OOB accuracy ===#
#=== RF model without tuning ==#
paste0("Test Accuracy: ", conf_matrix_rf$overall[1])
paste0("OOB Accuracy: ", 1 - oob_err)
#=== RF model after tuning ==#
paste0("Test Accuracy: ", conf_matrix_rf_tune$overall[1])
paste0("OOB Accuracy: ", 1 - oob_err_tune)

#=== RF model's AUC ===#
pred_rf_prob <- predict(rf_model, newdata = test_set, type = "prob")
pred_rf_tune_prob <- predict(rf_tune, newdata = test_set, type = "prob")
library(pROC)
auc_rf <- auc(ifelse(test_set$default=="yes", 1,0), pred_rf_prob[,"yes"])
auc_rf_tune <- auc(ifelse(test_set$default=="yes", 1,0), pred_rf_tune_prob[,"yes"])
print(paste0("Test set AUC (Random Forest): ", auc_rf))
print(paste0("Test set AUC (Tuned Random Forest ): ", auc_rf_tune))

#=== GBM (Gradient Boosting Machine) ===#
train_set$default <- ifelse(train_set$default == "yes", 1, 0)
library(gbm)
set.seed(123)
gbm_model <- gbm(formula = default~., distribution = "bernoulli",
                    data = train_set,
                    n.trees = 10000)
summary(gbm_model)
test_set$default <- ifelse(test_set$default == "yes", 1, 0)
pred_gbm <- predict(gbm_model, 
                    newdata = test_set,
                    n.trees = 10000,
                    type = "response")
range(pred_gbm)

#=== Tuning GBM model  with hyperparameter at 'Early Stopping'==#
#== optimal hyperparameter (n.trees) based on OOB error
ntree_opt_oob <- gbm.perf(gbm_model, 
                          method = "OOB", 
                          oobag.curve = TRUE)

#== Optimal hyperparamter (cv.folders) ==# 
#== Train new GBM model with cv.folders ==#
set.seed(123)
gbm_model_cv <- gbm(formula = default ~ ., 
                       distribution = "bernoulli", 
                       data = train_set,
                       n.trees = 10000,
                       cv.folds = 2)
#== optimal hyperparameter (n.trees) based on gbm_model_cv ==#
ntree_opt_cv <- gbm.perf(object = gbm_model_cv, 
                         method = "cv")
print(paste0("Optimal n.trees (OOB Estimate): ", ntree_opt_oob))                         
print(paste0("Optimal n.trees (CV Estimate): ", ntree_opt_cv))

#== Predictions based on ntree_opt-oob & ntree_opt_cv ==#
pred_oob_gbm <- predict(gbm_model, 
                        newdata = test_set,
                        n.trees = ntree_opt_oob)
pred_cv_gbm <- predict(object = gbm_model, 
                       newdata = test_set,
                       n.trees = ntree_opt_cv)   
#== AUC comparison between GBM and OOB & CV adjusted models==#
auc_gbm <- auc(test_set$default, pred_gbm)
auc_oob_gbm <- auc(test_set$default, pred_oob_gbm)
auc_cv_gbm <- auc(test_set$default, pred_cv_gbm)
print(paste0("Test set AUC (GBM): ", auc_gbm))
print(paste0("Test set AUC (OOB): ", auc_oob_gbm))                         
print(paste0("Test set AUC (CV): ", auc_cv_gbm))

#== Rank all models by AUC measures ==#
auc_list <- data.frame(Models=c("Logit_full", "Logit Regression", "Proit Regression",
                                   "Random Forest", "Tuned Random Forest", "GBM",
                                   "GBM_OOB", "GBM_CV"), 
                          AUC=c(auc_full, auc_logit, auc_probit, auc_rf, auc_rf_tune,
                                auc_gbm, auc_oob_gbm, auc_cv_gbm))
rank_models <- auc_list[sort(auc_list$AUC), "Models"]
levels(rank_models)

library(ROCR)
preds_list <- list(pred_full, pred_logit, pred_probit, 
                   pred_rf, pred_rf_best, pred_gbm, pred_oob_gbm, pred_cv_gbm)
m <- length(preds_list)
actuals_list <- rep(list(test_set$default), m)
pred_all <- prediction(preds_list, actuals_list)
rocs <- performance(pred_all, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Full Logistic", "Logistic", "Probit", "Random Forest", "Tuned RF", 
                  "GBM", "GBM_OOB", "GBM_CV"),cex = .65,
       xjust = 1, yjust = 1, fill = 1:m)