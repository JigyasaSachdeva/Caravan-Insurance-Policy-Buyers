### Jigyasa Sachdeva
### UIN- 664791188
### IDS 575- Final Project

#Reading data
caravan <- read.csv("~/Desktop/Third Sem/Statistics for Machine Learning/FinalProject/caravan-insurance-challenge.csv")
#Viewing data
View(caravan)

#Structure of the data is necessary to be evaluated
str(caravan)
summary(caravan)
caravan <- sapply(caravan, as.factor)
caravan <- as.data.frame(caravan)
summary(caravan)       

#Loading dplyr to pipeline the data
library(dplyr)

#Variables starting with A
caravan1 <- caravan %>% dplyr:: select(starts_with("A")) 
#All the variables starting with 'A' are numerical in nature
a <- as.data.frame(sapply(caravan1, as.numeric))

#Starting with 'P'
caravan2 <- caravan %>% dplyr:: select(starts_with("P")) 
p <- colnames(caravan2)
p <- as.data.frame(sapply(caravan2, as.numeric))
summary(p)

#Starting with 'm'
caravan3 <- caravan %>% dplyr:: select(starts_with("M")) 
m <- as.data.frame(sapply(caravan3, as.numeric))
summary(m)

#MAANTHUI
table(m$MAANTHUI)
#numeric
str(m$MAANTHUI)

#MGEMOMV
table(m$MGEMOMV)
#numeric
str(m$MAANTHUI)

#MOSTYPE
str(m$MOSTYPE)
m$MOSTYPE <- as.factor(m$MOSTYPE)
levels(m$MOSTYPE) <- c(levels(m$MOSTYPE), seq(1:41))
table(m$MOSTYPE)

#MOSHOOFD
str(m$MOSHOOFD)
m$MOSHOOFD <- as.factor(m$MOSHOOFD)
levels(m$MOSHOOFD) <- c(levels(m$MOSHOOFD), seq(1:10))
table(m$MOSHOOFD)

#Final data frame
attach(caravan)
data <- cbind(a,m,p,CARAVAN,ORIGIN)
summary(data)

#Split data
attach(data)
train <- data[ORIGIN == 'train',]
test <- data[data$ORIGIN == 'test',]
train <- subset(train, select = -c(ORIGIN))
test <- subset(test, select = -c(ORIGIN))

write.csv(train, 'train.csv')
write.csv(test, 'test.csv')
#These data sets are used throughout the project

#***************************************************************************************
#Baseline Model: Majority classification
table(train$CARAVAN)
pred <- rep(0, 4000)
pred <- as.factor(pred)
library(caret)
#Prediction on test data
confusionMatrix(pred, test$CARAVAN, positive = '1')

#***************************************************************************************
#Better baseline model
#Logistic Regression
set.seed(123)
mod <- glm(CARAVAN~., data = train, family = 'binomial')
summary(mod)
#Prediction on train data
p <- predict(mod, data = train, type = "response")
pred <- as.factor(ifelse(p>0.5, "1", "0"))
confusionMatrix(pred, train$CARAVAN, positive = '1')

#Prediction on test data
p_test <- predict(mod, newdata = test, type = "response")
pred <- as.factor(ifelse(p_test>0.5, "1", "0"))
confusionMatrix(pred, test$CARAVAN, positive = '1')


#***************************************************************************************
#Improving Logistic Regression Model

#Better cut off for logistic regression
#F score with beta = 2
s <- seq(from = 0, to= 1, by = 0.01)
fun = function(s)
{
  Class <- ifelse(p >= s, '1', '0')
  Class <- as.factor(Class)
  c_recall <- confusionMatrix(Class, train$CARAVAN, positive = '1')$byClass['Recall']
  c_precision <- confusionMatrix(Class, train$CARAVAN, positive = '1')$byClass['Precision']
  f_score <- (5*c_recall*c_precision)/ ((4*c_precision) + c_recall)
  return(f_score)
}
which.max(lapply(s, fun))  #9
s[15]
#0.14
#Using this cut off
Class <- ifelse(p >= 0.14, '1', '0')
Class <- as.factor(Class)
confusionMatrix(Class, train$CARAVAN, positive = '1')

#On test data: 
p_test <- predict(mod, newdata = test, type = "response")
Class <- ifelse(p_test >= 0.14, '1', '0')
Class <- as.factor(Class)
confusionMatrix(Class, test$CARAVAN, positive = '1')

#***************************************************************************************
#Improving the further logistic regression model on Python using Grid Search CV

#***************************************************************************************

#For Lasso and Ridge Regression: One hot encoding the data  

#Converting output variable to a numerical
Y <- as.numeric(train$CARAVAN)
#Dummy coding categorical variables
fac <- sapply(train, is.factor)
factor_v <- train[,fac]
cat1 <- as.data.frame(model.matrix(~0+factor_v[,"MOSTYPE"]))
cat2 <- as.data.frame(model.matrix(~0+factor_v[,"MOSHOOFD"]))

library(dplyr)
X <- train %>% select(-c(CARAVAN, MOSTYPE, MOSHOOFD))
X1 <- cbind(X, cat1, cat2)
summary(X1)
X <- data.matrix(X1)

#Using this dummy coded data further everywhere
write.csv(X, 'dummycoded_train.csv')

#Basic lasso regression
install.packages("glmnet")
library(glmnet)
lasso <- glmnet(X, Y, family = "binomial", alpha = 1, lambda = NULL)
set.seed(123) 
cv.lasso <- cv.glmnet(X, Y, alpha = 1, family = "binomial")
model <- glmnet(X, Y, alpha = 1, family = "binomial",
                lambda = cv.lasso$lambda.min)
coef(model)
probabilities <- model %>% predict(X)
predicted.classes <- as.factor(ifelse(probabilities >= 0.5, 1, 0))
train$CARAVAN <- as.factor(train$CARAVAN)
library(caret)
confusionMatrix(predicted.classes, train$CARAVAN, positive = '1')
#The results are worse than baseline, 
#hence continuing with hyper parameter tuning in Python


#Cross validation in lasso
Y <- as.numeric(train$CARAVAN)
#Dummy coding categorical variables
fac <- sapply(train, is.factor)
factor_v <- train[,fac]
cat1 <- as.data.frame(model.matrix(~0+factor_v[,"MOSTYPE"]))
cat2 <- as.data.frame(model.matrix(~0+factor_v[,"MOSHOOFD"]))

library(dplyr)
X <- train %>% select(-c(CARAVAN, MOSTYPE, MOSHOOFD))
X1 <- cbind(X, cat1, cat2)
summary(X1)
X <- data.matrix(X1)

library(glmnet)
library(caret)
set.seed(123) 

#Scope of values for lambda:
grid <- seq(0, 1, by = 0.01)
Y <- as.factor(ifelse(Y==1, 0, 1))
r <- 100000
for (l in grid)
{
  lasso <- glmnet(X, Y, family = "binomial", alpha = 1, lambda = l)
  probabilities <- lasso %>% predict(X)
  predicted.classes <- as.factor(ifelse(probabilities >= 0.5, 1, 0))
  c <- confusionMatrix(predicted.classes, Y, positive = '1')
  r <- c(r, c$byClass['Recall'])
}
r <- r[2,]
las1 <- cbind(grid,r)
las1[which.max(r), ]
#These result aren't great either, hence contuing with ridge, lasso later. 


#**************************************************************************************
#Treating imbalance

#SMOTE on data
set.seed(123)
library(DMwR)
SMOTE_data <- SMOTE(CARAVAN~., data = train, perc.over = 1600, prec.under = 100)
table(SMOTE_data$CARAVAN)
write.csv(SMOTE_data, 'SMOTE_traindata.csv')
#Using this data throughout the project


#***************************************************************************************
#Feature selection
#Stepwise logistic regression using same cut off on SMOTE data
full1 <- glm(CARAVAN~., data= SMOTE_data, family= "binomial")
null1 <- glm(CARAVAN~1, data= SMOTE_data, family= "binomial")
stepf1 <- step(null1, scope= list(lower= null1, upper= full1), direction = "both")
summary(stepf1)

Pred <- predict(stepf1, data= train, type="response")
Class <- ifelse(Pred >= 0.14, '1', "0")
Class <- as.factor(Class)
confusionMatrix(Class, test$CARAVAN, positive = '1')
#All the variables are selected by stepwise logistic regression
#No variables are reduced


#***************************************************************************************
#Feature selection done via statistical testing 
summary(caravan)
library(ggplot2)

x <- which(sapply(data, is.numeric) == TRUE)
names(x)
#[1] "MAANTHUI" "MGEMOMV"  "AWAPART"  "AWABEDR"  "AWALAND"  "APERSAUT" "ABESAUT" 
#[8] "AMOTSCO"  "AVRAAUT"  "AAANHANG" "ATRACTOR" "AWERKT"   "ABROM"    "ALEVEN"  
#[15] "APERSONG" "AGEZONG"  "AWAOREG"  "ABRAND"   "AZEILPL"  "APLEZIER" "AFIETS"  
#[22] "AINBOED"  "ABYSTAND"

a <- aov(MAANTHUI ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = MAANTHUI, x = CARAVAN, fill = CARAVAN))

a <- aov(MGEMOMV ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = MGEMOMV, x = CARAVAN, fill = CARAVAN))

a <- aov(AWAPART ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AWAPART, x = CARAVAN, fill = CARAVAN))

a <- aov(AWABEDR ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AWABEDR, x = CARAVAN, fill = CARAVAN))

a <- aov(AWALAND ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AWALAND, x = CARAVAN, fill = CARAVAN))

a <- aov(APERSAUT ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = APERSAUT, x = CARAVAN, fill = CARAVAN))

a <- aov(ABESAUT ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = ABESAUT, x = CARAVAN, fill = CARAVAN))

a <- aov(AMOTSCO ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AMOTSCO, x = CARAVAN, fill = CARAVAN))

a <- aov(AVRAAUT ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AVRAAUT, x = CARAVAN, fill = CARAVAN))

a <- aov(AAANHANG ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AAANHANG, x = CARAVAN, fill = CARAVAN))

a <- aov(ATRACTOR ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = ATRACTOR, x = CARAVAN, fill = CARAVAN))

a <- aov(AWERKT ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AWERKT, x = CARAVAN, fill = CARAVAN))

a <- aov(ABROM ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = ABROM, x = CARAVAN, fill = CARAVAN))

a <- aov(ALEVEN ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = ALEVEN, x = CARAVAN, fill = CARAVAN))

a <- aov(APERSONG ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = APERSONG, x = CARAVAN, fill = CARAVAN))

a <- aov(AGEZONG ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AGEZONG, x = CARAVAN, fill = CARAVAN))

a <- aov(AWAOREG ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AWAOREG, x = CARAVAN, fill = CARAVAN))

a <- aov(ABRAND ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = ABRAND, x = CARAVAN, fill = CARAVAN))

a <- aov(AZEILPL ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AZEILPL, x = CARAVAN, fill = CARAVAN))

a <- aov(APLEZIER ~ CARAVAN, data)
summary(a)
TukeyHSD(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = APLEZIER, x = CARAVAN, fill = CARAVAN))

a <- aov(AFIETS ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AFIETS, x = CARAVAN, fill = CARAVAN))

a <- aov(AINBOED ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = AINBOED, x = CARAVAN, fill = CARAVAN))

a <- aov(ABYSTAND ~ CARAVAN, data)
summary(a)
ggplot(data = data)+
  geom_boxplot(mapping = aes(y = ABYSTAND, x = CARAVAN, fill = CARAVAN))


y <- which(sapply(data, is.numeric) == FALSE)
names(y)
#[1] "MOSTYPE"  "MGEMLEEF" "MOSHOOFD" "MGODRK"   "MGODPR"   "MGODOV"  
#[8] "MGODGE"   "MRELGE"   "MRELSA"   "MRELOV"   "MFALLEEN" "MFGEKIND" "MFWEKIND"
#[15] "MOPLHOOG" "MOPLMIDD" "MOPLLAAG" "MBERHOOG" "MBERZELF" "MBERBOER" "MBERMIDD"
#[22] "MBERARBG" "MBERARBO" "MSKA"     "MSKB1"    "MSKB2"    "MSKC"     "MSKD"    
#[29] "MHHUUR"   "MHKOOP"   "MAUT1"    "MAUT2"    "MAUT0"    "MZFONDS"  "MZPART"  
#[36] "MINKM30"  "MINK3045" "MINK4575" "MINK7512" "MINK123M" "MINKGEM"  "MKOOPKLA"
#[43] "PWAPART"  "PWABEDR"  "PWALAND"  "PPERSAUT" "PBESAUT"  "PMOTSCO"  "PVRAAUT" 
#[50] "PAANHANG" "PTRACTOR" "PWERKT"   "PBROM"    "PLEVEN"   "PPERSONG" "PGEZONG" 
#[57] "PWAOREG"  "PBRAND"   "PZEILPL"  "PPLEZIER" "PFIETS"   "PINBOED"  "PBYSTAND"
#[64] "CARAVAN"

attach(data)
chisq.test(x = MZFONDS, y = CARAVAN)
table(MAUT0, CARAVAN)

#***************************************************************************************
#Understanding the target variable
View(train)

str(train)
train$MOSTYPE <- as.factor(train$MOSTYPE)
levels(train$MOSTYPE) <- c(levels(train$MOSTYPE), seq(1:41))
train$MOSHOOFD <- as.factor(train$MOSHOOFD)
levels(train$MOSHOOFD) <- c(levels(train$MOSHOOFD), seq(1:10))
train$CARAVAN <- as.factor(train$CARAVAN)

library(ggplot2)
ggplot(data = train)+
  geom_bar(mapping = aes(x=CARAVAN, fill = CARAVAN))+
  ylab("Frequency")+
  xlab("Existing buyers (0 = No, 1 = Yes)")+
  scale_fill_discrete("Existing buyers")

#Correlation amongst numeric variables
library(dplyr)
num <- train %>% select(-c('MOSTYPE', 'MOSHOOFD', 'CARAVAN'))
c <- cor(num)
View(c)


#**************************************************************************************
#Random Forest

#On normal data
library(randomForest)
fit <- randomForest(CARAVAN~., data = train, ntree = 500,
                    importance = TRUE, proximity = T, mtry = 7)
str(fit$confusion)
recall <- fit$confusion[4]/ (fit$confusion[4]+ fit$confusion[2])
precision <- fit$confusion[4]/ (fit$confusion[4]+ fit$confusion[3])
f_score <- (5*recall*precision) / (4*precision + recall)


#Hyper parameter tuning on normal data

#1
#Tuning mtry
k <- 10
nmethod <- 1
folds <- cut(seq(1,nrow(train)),breaks=k,labels=FALSE)
#From a sequence of 1 to the length of data: cut the data in folds
#size depending on the number of folds
f_score <- matrix(-1,k,nmethod, dimnames=list(paste0("Fold", 1:k), c("rf")))

for(i in 1:k)
{ 
  testIndexes <- which(folds==i, arr.ind=TRUE) 
  #Test indexes are where fold = i 
  testData <- train[testIndexes, ] 
  #i is test indexes
  trainData <- train[-testIndexes, ] 
  #train is the rest
  
  ind <- sample(2, nrow(trainData), replace = T, prob = c(0.7, 0.3))
  Train <- trainData[ind == 1, ]
  Validation <- trainData[ind == 2, ]
  #Train data divided in train and validation
  
  f_score <- c()
  #Empty error
  for(mt in seq(1,ncol(Train)))
    #mt is iterated in number of columns 
  {
    library(randomForest)
    rf <- randomForest(CARAVAN~., data = Train, ntree = 10, 
                       mtry = ifelse(mt == ncol(Train),mt - 1,mt))
    #mtry: number of variables in each tree differs from 1 to (total_cols-1)
    predicted <- predict(rf, newdata = Validation, type = "class")
    length(Validation$CARAVAN)
    c <- confusionMatrix(predicted, Validation$CARAVAN, positive = '1')
    recall_t <- c$byClass['Recall']
    precision_t <- c$byClass['Precision']
    f_score <- c(f_score, ((5*recall_t*precision_t) / (4*precision_t + recall_t)))
  }
  #Calculated predicted error above for each value of mtry 
  #Finding which mtry has the most minimum error
  
  bestmtry <- which.max(f_score) 
  library(randomForest)
  rf <- randomForest(CARAVAN~., data = trainData, ntree = 10, mtry = bestmtry)
  rf.pred <- predict(rf, newdata = testData, type = "class")
  models.err[i] <- mean(testData$Target != rf.pred)
  library(caret)
  c <- confusionMatrix(rf.pred, testData$CARAVAN, positive = '1')
  recall_t <- c$byClass['Recall']
  precision_t <- c$byClass['Precision']
  f_score[i] <- (5*recall_t*precision_t) / (4*precision_t + recall_t)
}

mean(f_score, na.rm = T)
#0.07573

#Chosen mtry on train data
bestmtry #81
rf <- randomForest(CARAVAN~., data = Train, ntree = 10, mtry = 81)
recall <- rf$confusion[4]/ (rf$confusion[4]+ rf$confusion[2])
precision <- rf$confusion[4]/ (rf$confusion[4]+ rf$confusion[3])
f_score <- (5*recall*precision) / (4*precision + recall)
#0.1393355 


#2
#Tuning ntree
k <- 10
nmethod <- 1
folds <- cut(seq(1,nrow(train)),breaks=k,labels=FALSE)
#From a sequence of 1 to the length of data: cut the data in folds
#size depending on the number of folds
f_score <- matrix(-1,k,nmethod, dimnames=list(paste0("Fold", 1:k), c("rf")))

for(i in 1:k)
{ 
  testIndexes <- which(folds==1, arr.ind=TRUE) 
  #Test indexes are where fold = i 
  testData <- train[testIndexes, ] 
  #i is test indexes
  trainData <- train[-testIndexes, ] 
  #train is the rest
  
  ind <- sample(2, nrow(trainData), replace = T, prob = c(0.7, 0.3))
  Train <- trainData[ind == 1, ]
  Validation <- trainData[ind == 2, ]
  #Train data divided in train and validation
  
  f_score <- c()
  #Empty error
  se <- seq(10, 500, by = 10)
  for(nt in se)
    #mt is iterated in number of columns 
  {
    library(randomForest)
    rf <- randomForest(CARAVAN~., data = Train, ntree = nt, mtry = 81)
    predicted <- predict(rf, newdata = Validation, type = "class")
    c <- confusionMatrix(predicted, Validation$CARAVAN, positive = '1')
    recall_t <- c$byClass['Recall']
    precision_t <- c$byClass['Precision']
    f_score <- c(f_score, ((5*recall_t*precision_t) / (4*precision_t + recall_t)))
  }
  #Calculated predicted error above for each value of mtry 
  #Finding which mtry has the most minimum error
  
  bestntree <- which.max(f_score) 
  library(randomForest)
  rf <- randomForest(CARAVAN~., data = trainData, ntree = bestntree, mtry = 81)
  rf.pred <- predict(rf, newdata = testData, type = "class")
  library(caret)
  c <- confusionMatrix(rf.pred, testData$CARAVAN, positive = '1')
  recall_t <- c$byClass['Recall']
  precision_t <- c$byClass['Precision']
  f_score[i] <- (5*recall_t*precision_t) / (4*precision_t + recall_t)
  
}

plot(se, f_score, xlab = "Number of trees", ylab = "F score")
f_score[23] #maximum
se[23]
# ntree = 230

mean(f_score, na.rm = T)
#0.103273

#Chosen ntree on train data
bestntree #23
bestntree = bestntree*10
rf <- randomForest(CARAVAN~., data = Train, ntree = 230, mtry = 81)
recall <- rf$confusion[4]/ (rf$confusion[4]+ rf$confusion[2])
precision <- rf$confusion[4]/ (rf$confusion[4]+ rf$confusion[3])
f_score <- (2.5*recall*precision) / (1.5*precision + recall)
#0.1393355 

#Since the f score is same before and after tuning ntree
#Considering 10 fold validation only for tuning mtry


#***************************************************************************************
#On SMOTE data
library(randomForest)
fit <- randomForest(CARAVAN~., data = SMOTE_data, ntree = 500,
                    importance = TRUE, proximity = T, mtry = 7)
recall <- fit$confusion[4]/ (fit$confusion[4]+ fit$confusion[2])
precision <- fit$confusion[4]/ (fit$confusion[4]+ fit$confusion[3])
f_score <- (5*recall*precision) / (4*precision + recall)
#0.96309494

#predicting on train data
predicted <- predict(fit, newdata = train)
class <- as.factor(predicted)
c <- confusionMatrix(class, train$CARAVAN, positive = '1')
recall_t <- c$byClass['Recall']
precision_t <- c$byClass['Precision']
f_score <- (5*recall_t*precision_t) / (4*precision_t + recall_t)
#0.06443299
#hence the model has been overfitted
#Using normal data > SMOTE for further hyper parameter tuning on Random Forest
#Further hyper parameter tuning is done in Python


#**************************************************************************************
#Support Vector Machine

library(Matrix)
library(e1071) #For SVM
library(caret) #For Confusion Matrix
options(scipen=99)
set.seed(123)

#Train
dummycoded_train <- read.csv("~/Desktop/Data/dummycoded_train.csv", 
                             row.names=1, 
                             stringsAsFactors=FALSE)
summary(dummycoded_train)
attach(dummycoded_train)
X <- dummycoded_train
Y <- as.factor(ifelse(train$CARAVAN == 1, 1, -1))

#Test
#Test data one hot encoding
fac <- sapply(test, is.factor)
factor_v <- test[,fac]
cat1 <- as.data.frame(model.matrix(~0+factor_v[,"MOSTYPE"]))
cat2 <- as.data.frame(model.matrix(~0+factor_v[,"MOSHOOFD"]))
X_test <- test %>% select(-c(CARAVAN, MOSTYPE, MOSHOOFD))
X_test1 <- cbind(X_test, cat1, cat2)
summary(X_test1)
Y_test <- as.factor(ifelse(test$CARAVAN==1, 1, -1))


#Basic SVM
Model_1 <- svm(X, Y, kernel = "linear", probability = TRUE)
pred_1 <- predict(Model_1, X, probability = T)
confusionMatrix(pred_1, Y, positive = '1')

#Assigning class weights
wts <- 100 / table(Y)
svm1 <- svm(X, factor(Y), probability=TRUE, 
            kernel = "linear", class.weights = wts)
pred_1 <- predict(svm1, X, probability = T)
confusionMatrix(pred_1, Y, positive = '1')
#Improved results


#Tuning cost
cost_val <- c(0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
k <- 10
nmethod <- 1
#train: X, Y
#test: X_test1, Y_test
train <- cbind(X, Y)
folds <- cut(seq(1,nrow(train)),breaks=k,labels=FALSE)
#From a sequence of 1 to the length of data: cut the data in folds
#size depending on the number of folds
f_score <- matrix(-1,k,nmethod, dimnames=list(paste0("Fold", 1:k), c("svm")))

cost <- -1
for(i in 1:k)
{ 
  testIndexes <- which(folds== i, arr.ind=TRUE) 
  
  #Test indexes are where fold = i 
  testData <- train[testIndexes, ] 
  X_test <- testData %>% select(-Y)
  Y_test <- testData$Y
  
  #i is test indexes
  trainData <- train[-testIndexes, ] 
  #train is the rest
  
  X <- trainData %>% select(-Y)
  Y <- trainData$Y
  wts <- 100 / table(Y)
  
  f_score <- c()
  #Empty error
  for(c in cost_val)
  {
    library(e1071)
    model <- svm(X, factor(Y), probability=TRUE, cost= c,
                 kernel = "linear", class.weights = wts)
    pred <- predict(model, newdata= X_test, probability = F)
    c <- confusionMatrix(as.factor(pred), as.factor(Y_test), positive = '1')
    recall_t <- c$byClass['Recall']
    precision_t <- c$byClass['Precision']
    f_score <- c(f_score, ((5*recall_t*precision_t) / (4*precision_t + recall_t)))
  }
  
  index <- which.max(f_score) 
  bestcost <- cost_val[index]
  cost <- c(cost, bestcost)
}
cost
models_cost <- cost[-1]
bestcost <- 0.125

#Chosen cost on train data
bestcost #0.125
wts <- 100 / table(train$Y)
svm_1 <- svm(X, factor(Y), probability=TRUE, cost= 0.125,
             kernel = "linear", class.weights = wts)

pred_1 <- predict(svm_1, data = X, probability = F)
c <- confusionMatrix(as.factor(pred_1), as.factor(Y), positive = '1')
recall_t <- c$byClass['Recall']
precision_t <- c$byClass['Precision']
f_score <- (5*recall_t*precision_t) / (4*precision_t + recall_t)

#Final evaluation done on Python 
#**************************************************************************************





