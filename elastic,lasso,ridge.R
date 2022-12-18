library(caret)
library(glmnet)
library(mlbench)
library(psych)
data("BostonHousing")
data= BostonHousing
View(data)

##### data partitioning
set.seed(222)
ind= sample(2, nrow(data),replace = T, prob = c(0.7,0.3))
train= data[ind==1,]
test=data[ind==2,]

#custom control parameters
custom=trainControl(method = "repeatedcv",number = 10,repeats=5,verboseIter = T)

### linear model
lm=train(medv~.,train,method="lm",trControl=custom)

## results
lm$results
summary(lm)
lm
 

#### ridge
set.seed(222)
ridge=train(medv~.,train,method='glmnet',tuneGrid=expand.grid(alpha=0,lambda=seq(0.0001,1,length=5)),trControl=custom)
plot(ridge)
ridge
plot(ridge$finalModel,xvar = "lambda",label=T)
plot(varImp(ridge,scale=F))

x = model.matrix(medv~.,BostonHousing)[,-1]
y = BostonHousing$medv
train = sample(1:nrow(x),nrow(x)/2)
test = (-train)
y.test = y[test]
cv.out = cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlam = cv.out$lambda.min
out = glmnet(x,y,alpha=0)
ridge.coef = predict(out,type="coefficients",s=bestlam)
ridge.coef

##### lasso
set.seed(222)
lasso=train(medv~.,train,method='glmnet',tuneGrid=expand.grid(alpha=1,lambda=seq(0.0001,0.2,length=5)),trControl=custom)
plot(lasso)
lasso
plot(lasso$finalModel,xvar = 'lambda',label=T)
plot(lasso$finalModel,xvar = 'dev',label=T)
plot(varImp(lasso,scale=F))

cv.out = cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam = cv.out$lambda.min

grid=10^seq(10,-2,length=100)
out=glmnet(x,y,alpha=1,lambda=grid)
LASSO.coef = predict(out,type="coefficients",s=bestlam)
LASSO.coef

### elastic net
set.seed(222)
en=train(medv~.,train,method='glmnet',tuneGrid=expand.grid(alpha=seq(0,1,length=10),lambda=seq(0.0001,0.2,length=5)),trControl=custom)
plot(en)
en
plot(en$finalModel,xvar = 'lambda',label=T)
plot(varImp(lasso,scale=F))

cv.out = cv.glmnet(x[train,],y[train],alpha=seq(0.01,0.9,length=10))
plot(cv.out)
bestlam = cv.out$lambda.min

out=glmnet(x,y,alpha=seq(0.01,0.9,length=10),lambda=grid)
Elastic.coef = predict(out,type="coefficients",s=bestlam)
cbind(ridge.coef,LASSO.coef, Elastic.coef)

Elastic.coef

##### compare models
model_list=list(LinearModel=lm,Ridge=ridge,Lasso=lasso,ElasticNet=en)
res=resamples(model_list)
summary(res)

en$bestTune
