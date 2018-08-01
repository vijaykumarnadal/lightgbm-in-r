#lightgbm in r example
#load the required librarys
library(dplyr)
library(lightgbm)
#loading the datasets
train1<-read.csv("E:/avdatasets/train1.csv",header = T)
test1<-read.csv("E:/avdatasets/test1.csv",header = T)
hdata<-read.csv("E:/avdatasets/hero_data.csv",header = T)
#some data exploration
test1$num_wins<-93.69102
test1$kda_ratio<-1

train1c<-inner_join(train1,hdata,by="hero_id")
test1c<-inner_join(test1,hdata,by="hero_id")
View(train1c)
class(train1c$id)
train1c$id<-as.numeric(as.factor(train1c$id))-1
train1c$primary_attr<-as.numeric(as.factor(train1c$primary_attr))-1
train1c$attack_type<-as.numeric(as.factor(train1c$attack_type))-1
train1c$roles<-as.numeric(as.factor(train1c$roles))-1
View(test1)
View(test1c)
test1c$id<-as.numeric(as.factor(test1c$id))-1
test1c$primary_attr<-as.numeric(as.factor(test1c$primary_attr))-1
test1c$attack_type<-as.numeric(as.factor(test1c$attack_type))-1
test1c$roles<-as.numeric(as.factor(test1c$roles))-1
trx<-train1c[,"kda_ratio"]
try<-train1c[,colnames(train1c)!="kda_ratio"]
tstx<-test1c[,"kda_ratio"]
tsty<-test1c[,colnames(test1c)!="kda_ratio"]
trainlg<-lgb.Dataset(data=as.matrix(try),label=as.numeric(trx))
testlg<-lgb.Dataset(data=as.matrix(tsty),label=as.numeric(tstx))
vlda=list(test=testlg)
param<-list(max_bin=9,               
            learning_rate = 0.0021,
            boosting_type = 'gbdt',
            objective = "regression",
            metric = 'mae'(or rmse //depending on your requirement),
            sub_feature = 0.5,
            bagging_fraction = 0.85,
            bagging_freq = 20,
            num_leaves = 60,
            min_data = 500,
            min_hessian = 0.05)

lgmodel<-lgb.train(params = param,data=trainlg,valids=vlda,nrounds = 500,early_stopping_rounds = 40)
#variable importance
vi<-lgb.importance(lgmodel,percentage = T)
p<-predict(lgmodel,data=as.matrix(tsty))