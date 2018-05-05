#Setting the working directory
setwd('D:/Projects/Titanic/')

#Loading the required packages
library(dplyr)
library(ggplot2)
library(tidyr)
library(parallel)
library(randomForest)
library(caret)
library(car)

#Reading in the datasets
train <- read.csv('Datasets/train.csv')
test <- read.csv('Datasets/test.csv')

str(train)
any(is.na(train))
colSums(is.na(train))

str(test)
any(is.na(test))
colSums(is.na(test))

#Combine both the train and test sets
test$Survived <- 0
titanic.full <- rbind(train, test)
glimpse(titanic.full)

#Feature engineering
#Getting the titles from name
regex <- gregexpr(', [^\\..]+.', titanic.full$Name)
titles <- regmatches(titanic.full$Name, regex)
titles <- gsub(', ', "", titles)
titles <- gsub('.', "", titles, fixed = TRUE)
titanic.full$Title <- titles
#titanic.full$Title <- as.factor(titanic.full$Title)

table(titanic.full$Sex, titanic.full$Title)
table(titanic.full$Title)

#Combine titles with very low count as rare.

rare.title <- names(table(titanic.full$Title))[table(titanic.full$Title) < 10][-9:-11]
rare.title  

#Fix typos
titanic.full$Title[titanic.full$Title == 'Mlle']        <- 'Miss' 
titanic.full$Title[titanic.full$Title == 'Ms']          <- 'Miss'
titanic.full$Title[titanic.full$Title == 'Mme']         <- 'Mrs' 
titanic.full$Title[titanic.full$Title %in% rare.title]  <- 'Rare_Title'

#Get Surname
titanic.full$Surname <- sapply(as.character(titanic.full$Name), function(x) {strsplit(x, split = '[,]')}[[1]][1])
length(unique(titanic.full$Surname))

#Family Size
titanic.full$FamilySize <- 1 + titanic.full$SibSp + titanic.full$Parch
table(titanic.full$FamilySize)

#Family size descretized
titanic.full$FamilySizeDis[titanic.full$FamilySize == 1] <- 'singleton'
titanic.full$FamilySizeDis[titanic.full$FamilySize < 5 & titanic.full$FamilySize > 1] <- 'small'
titanic.full$FamilySizeDis[titanic.full$FamilySize > 4] <- 'large'
table(titanic.full$FamilySizeDis)

#Family
titanic.full$Family <- paste(titanic.full$Surname, titanic.full$FamilySize, sep = '_')
titanic.full$Family

#Cabin.Level
##Getting the first letters of cabin, might represent the level it is situated in the ship.
cabin.level <- substr(titanic.full$Cabin, 1, 1)
table(cabin.level)
titanic.full$Cabin.Level <- cabin.level

#Missing values
#Embarked
titanic.full %>% filter(Embarked == "")
ggplot(titanic.full, aes(x = Embarked, y = Fare)) + geom_boxplot()
titanic.full[c(62,830), 'Embarked'] <- 'C'

#Fare
titanic.full %>% filter(is.na(Fare))
ggplot(titanic.full[titanic.full$Pclass == 3 & titanic.full$Embarked == 'S', ], aes(x = Fare)) + geom_density() + scale_x_continuous()
titanic.full$Fare[1044] <- median(titanic.full[titanic.full$Pclass == 3 & titanic.full$Embarked == 'S', 'Fare'], na.rm = TRUE)                                                                                             

#Data Conversion
factor.vars <- c('PassengerId', 'Pclass', 'Title', 'FamilySize', 'FamilySizeDis', 'Cabin.Level', 'Surname', 'Family')
titanic.full[factor.vars] <- lapply(titanic.full[factor.vars], function(x){as.factor(x)})

#Age
titanic.full.dummy.model <- dummyVars(~. -PassengerId -Name -Ticket -Cabin -Survived -Cabin.Level -Family -Surname, data = titanic.full)
titanic.full.dummy <- predict(titanic.full.dummy.model, newdata = titanic.full)

impute <- preProcess(titanic.full.dummy, method = "bagImpute")
titanic.preprocessed <- predict(impute, newdata = titanic.full.dummy)

titanic.full$Age <- titanic.preprocessed[,'Age']

par(mfrow = c(1,2))
hist(titanic.preprocessed[,'Age'])
hist(titanic.full$Age)

#Child
titanic.full$Child[titanic.full$Age < 18] <- 'Child'
titanic.full$Child[titanic.full$Age >= 18] <- 'Adult'
table(titanic.full$Child, titanic.full$Survived)
table(titanic.full$Child)

#Mother
titanic.full$Mother <- 'Not Mother'
titanic.full$Mother[titanic.full$Sex == 'female' & titanic.full$Parch > 0 & titanic.full$Age > 18 & titanic.full$Title != 'Miss'] <- 'Mother'
table(titanic.full$Mother, titanic.full$Survived)

titanic.full$Child <- as.factor(titanic.full$Child)
titanic.full$Mother <- as.factor(titanic.full$Mother)

#Whether the person has a cabin or not
titanic.full$HasCabin <- ifelse(titanic.full$Cabin == "", 0, 1)
titanic.full$HasCabin <- as.factor(titanic.full$HasCabin)

#Prediction
titanic.train <- slice(titanic.full, 1:891)
titanic.test <- slice(titanic.full, 892:1309)

#Creating Cross Validation Set
index <- createDataPartition(titanic.train$Survived, p = 0.8, times = 1, list = FALSE)
titanic.model.train <- titanic.train[index,]
titanic.model.test <- titanic.train[-index,]


#Random Forest
#Building model with cross validation set
set.seed(754)
rf.model.car <- train(factor(Survived) ~ Title + Age + Sex + 
                        Fare + Embarked + Pclass +
                        FamilySizeDis + Child + Mother, method = 'rf', data = titanic.train)



rf.model.car
varImp(rf.model.car)
plot(rf.model.car)


practice.predictions <- predict(rf.model.car, titanic.model.test)

confusionMatrix(practice.predictions, factor(titanic.model.test$Survived))


#Building the Model
set.seed(754)
rf.model <- randomForest(factor(Survived) ~ Title + Age + Sex + 
                           Fare + Embarked + Pclass + 
                           FamilySizeDis + Child + Mother + HasCabin,
                         data = titanic.train)
rf.model
plot(rf.model)
par(mfrow = c(1,1))
varImpPlot(rf.model)

#Predictions
predictions <- predict(rf.model.car, newdata = titanic.test)
predictions

#Logistic Regression

lr.model <- glm(Survived ~ )

#Creating final submission csv file
submission <- data.frame(PassengerId = titanic.test$PassengerId, Survived = predictions)
write.csv(submission, file = "titanic_submission_36.csv", row.names = FALSE)
