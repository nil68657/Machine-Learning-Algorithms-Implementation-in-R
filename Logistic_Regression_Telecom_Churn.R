# R-Studio logistic regression with Telecom_Churn.csv data

# First set working directory using setwd() to where the datafile is
setwd("C:/Users/nilan/Desktop/Big Data/Logistic")

# Then read OrganicsData.csv
churnData <- read.csv("Telecom_Churn.csv", header=TRUE)

# remove rows with missing values
churnDataClean <- na.omit(churnData)


# set reference value for acct_plan_type column as "Silver"
churnDataClean$acct_plan_type <- relevel(churnDataClean$acct_plan_type, ref = "Silver")

# set reference value for Complaint_Code column as "Pricing"
churnDataClean$Complaint_Code <- relevel(churnDataClean$Complaint_Code, ref = "Pricing")


# fit base model
logitbase <- glm(target_churn ~ 1, data = churnDataClean, family = binomial)

summary(logitbase)


# fit full model
logitfull <- glm(target_churn ~ Avg_Calls + Percent_Increase_MOM +
                   Account_Age + Equipment_Age + Current_Days_OpenWorkOrders +
                   Avg_Days_Delinquent + acct_plan_type + Complaint_Code, 
                 data = churnDataClean, family = binomial)

summary(logitfull)


# run stepwise selection with "forward" method
finalmodel <- step(logitbase, scope=list(upper=logitfull, lower=~1), direction="forward", trace=TRUE)

summary(finalmodel)


# model fit stats for logistic regression
# first find number of row values
n <- nrow(churnDataClean)


# log likelyhood
logLik(finalmodel)

# Likelihood ratio test
install.packages('lmtest', repos="https://cran.r-project.org")
library(lmtest)

lrtest(logitbase, finalmodel)


# McFadden R2
McfadR2 <- 1-((finalmodel$deviance/-2)/(finalmodel$null.deviance/-2))
cat("mcFadden R2=",McfadR2,"\n")

# AIC
AIC<- finalmodel$deviance+2*2
cat("AIC=",AIC,"\n")


# package for logistic regression model fit stats
install.packages('pscl', repos="https://cran.r-project.org")
library(pscl)
pR2(finalmodel)

# for classification table
predprob <- fitted(finalmodel)

probTable <- data.frame(predprob)

table(predprob>.5, churnDataClean$target_churn)

# merge the original dataset with the predicted probabilities
finalTable <- cbind(churnDataClean, probTable)

