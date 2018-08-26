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
#=== Load credit data from github url ===#
loan_data <- read.csv(curl("https://raw.githubusercontent.com/finsl/Credit_ML_VaR/master/credit_data.csv"))
#=== check data names & NAs ===#
attach(loan_data)
names(loan_data);summary(loan_data)
length(which(is.na(loan_data)))

#=== Descriptive Summary of factor variables ===#
summary_table <- loan_data %>%
  gather(variable, value, job, housing, purpose, credit_history) %>%
  group_by(default, variable, value) %>%
  summarise (n = n()) %>%
  mutate(freq = n / sum(n))
sum_fv <- cbind(subset(summary_table, default=="no"), subset(summary_table, default=="yes"))
write.table(sum_fv, file = "C:/Users/xxxx/Desktop/Credit/sum_table.csv",  row.names=FALSE, sep = ",",  qmethod = "double")
rm(list = ls(pattern = "sum"))

#==== Explore relationships between variables ===#
#== int_rate & emp_length have NAs(didn't run hist), years_at_residence has smaller bin width (didn't run hist)
histogram(~age|default, type = "count", xlab = "Age", ylab = "Frequency",
          breaks = seq(min(age),max(age),by=((max(age) - min(age))/(length(age)-1))))
histogram(~annual_inc|default, type = "count", xlab = "Annual Income", ylab = "Frequency",
          breaks = seq(min(annual_inc),max(annual_inc),by=((max(annual_inc) - min(annual_inc))/(length(annual_inc)-1))))
densityplot(~months_loan_duration|default, xlab = "Loan Duration")
densityplot(~int_rate|default, xlab = "Interest Rate")
densityplot(~amount|default, xlab = "Loan Amount")
xyplot(annual_inc ~ months_loan_duration|default, xlab = "Loan Duration in Months", ylab = "Anuual Income",
  panel = function(x,y) {
  panel.xyplot(x,y, pch = 16)
}) 
xyplot(amount ~ age|default, xlab = "Age", ylab = "Loan Amount",
  panel = function(x,y) {
  panel.xyplot(x,y, pch = 16)
})
xyplot(int_rate ~ age|default, xlab = "Age", ylab = "Interest Rate",
  panel = function(x,y) {
  panel.xyplot(x,y, pch = 16)
})
xyplot(int_rate ~ annual_inc|default, xlab = "Annual Income", ylab = "Interest Rate",
       panel = function(x,y) {
         panel.xyplot(x,y, pch = 16)
       })

trellis.par.set(col.whitebg())
bwplot(int_rate~purpose+dependents|default,  scales = list(x = list(rot = 45)), xlab = "Loan Purpose & No. of Dependents", ylab = "Interest Rate")
bwplot(int_rate~housing+credit_history|default, scales = list(x = list(rot = 45)), xlab = "Ownership & Credit History", ylab = "Interest Rate")

#=== Data Preprocessing: remove outliers ===#
loan_outlier <- loan_data[, c("int_rate", "emp_length", "months_loan_duration", "amount", "annual_inc", "age")]
output_no_outlier <- loan_data
for (n in colnames(loan_outlier)) {
  n_break <- sqrt(nrow(loan_outlier))
  hist <- hist(loan_outlier[, n], breaks = n_break, main=n)
  index <- which(loan_data[, n]>hist$breaks[length(hist$breaks)-1])
  output_no_outlier <- output_no_outlier[-index, ]
}
rm(n_break)
rm(index)
rm(n)
rm(hist)
rm(loan_outlier)

#=== Replace the missing interest rate data by median ===#
na_index_ir <- which(is.na(output_no_outlier$int_rate))
median_ir<- median(output_no_outlier$int_rate, na.rm=TRUE)
loan_data_replace <- output_no_outlier
loan_data_replace$int_rate[na_index_ir] <- median_ir
summary(loan_data_replace$int_rate) # Check if the NAs are gone

#=== Binning missing emp_length data ===#
#=== emp_length has small proportion NAs & also hard to be replaced ===#
summary(loan_data_replace$emp_length)
loan_data_replace$emp_cat <- rep(NA, length(loan_data_replace$emp_length))
loan_data_replace$emp_cat[which(loan_data_replace$emp_length <= 15)] <- "0-15"
loan_data_replace$emp_cat[which(loan_data_replace$emp_length > 15 & loan_data_replace$emp_length <= 30)] <- "15-30"
loan_data_replace$emp_cat[which(loan_data_replace$emp_length > 30 & loan_data_replace$emp_length <= 45)] <- "30-45"
loan_data_replace$emp_cat[which(is.na(loan_data_replace$emp_length))] <- "Missing"
loan_data_replace$emp_cat <- as.factor(loan_data_replace$emp_cat)
loan_data_replace$emp_length <- NULL
loan_data_clean <- loan_data_replace
rm(loan_data_replace)
rm(median_ir)
rm(na_index_ir)
