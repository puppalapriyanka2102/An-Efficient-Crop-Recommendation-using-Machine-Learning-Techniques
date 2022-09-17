---
  title: "Project"
author: "Priyanka Puppala"
date: "Last Updated: `r format(Sys.time(), '%d, %B, %Y at %H:%M')`"
output: rmdformats::readthedown
---
  # Project
  # Introduction
  # Problem Statement
  # Data Collection
  
  
  

#Importing requied libraries
library(readr)
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(dplyr)
# library(ACSWR)
library(caret)
library(e1071)
library(factoextra)
library(mlbench)
library(NeuralNetTools)
library(nnet)
library(pROC)
library(RSADBE)
library(survival)
library(rpart)

crop <- read.csv2("Cropdata.csv", header = TRUE, sep = ",")
View(crop)


crop$PH <- as.numeric(crop$PH )
crop$EC <- as.numeric(crop$EC )
crop$N <- as.numeric(crop$N )
crop$P <- as.numeric(crop$P )
crop$k <- as.numeric(crop$k )
crop$Total <- as.numeric(crop$Total )
#Handling missing values
crop$PH[is.na(crop$PH)] <- mean(crop$PH, na.rm = TRUE)
crop$N[is.na(crop$N)] <- mean(crop$N, na.rm = TRUE)
crop$P[is.na(crop$P)] <- mean(crop$P, na.rm = TRUE)
crop$k[is.na(crop$k)] <- mean(crop$k, na.rm = TRUE)
crop$Total[is.na(crop$Total)] <- mean(crop$Total, na.rm = TRUE)
crop$Time.line <- as.factor(crop$Time.line)
sum(is.na(crop))



str(crop)
head(crop)
summary(crop)




crop$Total <- round(crop$Total,0)

#**************************************************Step_1*******************************************
#The first step is create two new columns as follows:
# Categories in grade coloumn- Converting grades into low or high risk
crop_new <- mutate(crop, 
                   Crop_Type = case_when(Total %in% 1:200 ~ "Ground Nut",  
                                         Total %in%  200:214	~ "Sugar Cane",
                                         Total %in% 215:235 ~ "Grape",
                                         Total %in% 236:244  ~ "Onion",
                                         Total %in% 245:250 ~ "Banana",
                                         Total  %in%  251:100000 ~ "Turmeric"))



#Creating a csv file 
write.table(crop_new, file = "crop_new.csv",
            sep = ",",
            row.names = FALSE)
View(crop_new)


# Data Preparation


sample_set <- sample(2, nrow(crop_new), 
                     replace = TRUE, 
                     prob = c(0.7, 0.3))
train <- crop_new[sample_set==1,]
head(train)


#Creating a csv file 
write.table(train, file = "crop_train.csv",
            sep = ",",
            row.names = FALSE)



test <- crop_new[sample_set==2,]
head(test)

#Creating a csv file 
write.table(crop_new, file = "test.csv",
            sep = ",",
            row.names = FALSE)






## Data Cleaning


library(DataExplorer)
sum(is.na(train))
sum(is.na(test))
plot_missing(train)


# Exploratory Data Analysis (EDA)
* describe - can computes the statistics of all numerical variables 

library(Hmisc)
describe(train)
describe(test)




#Two continuous variables
# Taking  PH & EC 




library(ggplot2)

q <- ggplot(data = train, aes(x =Time.line , y = log(PH)  ))+
  geom_line(colour = "darkgreen") + 
  geom_point(aes(colour = factor(Crop_Type)), size =3) +
  geom_point(colour = "grey90", size = 1.5)+
  labs(title = 'Crop according to PH for  Time.line 2015-2020',
       y='PH of the soil',x='Time.line')
q


library(plotly)

fig <- train %>%
  plot_ly(
    x = ~log(PH), 
    y = ~log(P), 
    size = ~k, 
    color = ~Crop_Type, 
    frame = ~Time.line, 
    text = ~P, 
    hoverinfo = "text",
    type = 'scatter',
    mode = 'markers'
    
  )

fig <- fig %>% layout(
  xaxis = list(
    type = "log"
  )
)

fig




plot_ly(train, x = ~log(PH), y = ~Crop_Type , 
        type = 'scatter', 
        mode = 'markers',
        marker = list(color = "darkgreen" ),  opacity = 0.5) %>%  
  layout(title = 'Crop according to PH for  Time.line 2015-2020', 
         yaxis = list(title = 'Time.line'), 
         xaxis = list(title = 'PH of the soil ') )





#  Algorithms


train$Crop_Type <- as.factor(train$Crop_Type)
library(mlbench)
library(caret)

# Example of Boosting Algorithms
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"



## Modelling
### SvmRadial

set.seed(seed)
fit.svmRadial <- train(Crop_Type~., data=train, method="svmRadial", metric=metric, trControl=control)
fit.svmRadial

### Stochastic Gradient Boosting

# Stochastic Gradient Boosting
set.seed(seed)
fit.gbm <- train(Crop_Type~., data=train, method="gbm", metric=metric, trControl=control, verbose=FALSE)
fit.gbm

### kNN

# kNN
set.seed(seed)
fit.knn <- train(Crop_Type~., data=train, method="knn", metric=metric, preProc=c("center", "scale"), trControl=control)
fit.knn

# Model Selection 
### summarize results


# summarize results
boosting_results <- resamples(list(svmRadial=fit.svmRadial, gbm=fit.gbm, knn =fit.knn))
summary(boosting_results)
dotplot(boosting_results)



# Bagging Algorithms
# Bagged CART
#Random Forest


control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"


## Bagged CART

# Bagged CART
set.seed(seed)
fit.treebag <- train(Crop_Type~., data=train, method="treebag", metric=metric, trControl=control)

## Random Forest

# Random Forest
set.seed(seed)
fit.rf <- train(Crop_Type~., data=train, method="rf", metric=metric, trControl=control)


### summarize results

# summarize results
bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
summary(bagging_results)
dotplot(bagging_results)

#Stacking Algorithms

# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('lda', 'rpart','glm')
set.seed(seed)

models <- caretList(Crop_Type~., data=train, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)



# correlation between results
modelCor(results)
splom(results)


#combine the predictions of the classifiers using a simple linear model


# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)




# stack using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)













































































































































































































