



data=read.csv("song_data.csv")
head(data)
data.numeric <- data[,sapply(data, is.numeric)]
head(data.numeric)

install.packages("pls")
library(pls)
library(psych) 
library(readr)
library(tree)
library(caret)
library(gbm)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)

# 1. verificar NA
sapply(data, function(x) sum(is.na(x)))


head(data)
hist(data$song_popularity)
boxplot(data$song_popularity)

hist(data$song_duration_ms)
boxplot(data$song_duration_ms)

hist(data$acousticness)
boxplot(data$acousticness)

hist(data$danceability)
boxplot(data$sdanceability)

hist(data$energy)
boxplot(data$senergy)

hist(data$instrumentalness)
boxplot(data$instrumentalness)

hist(data$key)
boxplot(data$key)

hist(data$liveness)
boxplot(data$liveness)

hist(data$loudness)
boxplot(data$loudness)

hist(data$audio_mode)
boxplot(data$audio_mode)

hist(data$speechiness)
boxplot(data$speechiness)

hist(data$tempo)
boxplot(data$tempo)

hist(data$time_signature)
boxplot(data$time_signature)

hist(data$audio_valence)
boxplot(data$saudio_valence)


# Matriz de correlação

correlation_matrix <- cor(data.numeric)
correlation_data <- melt(correlation_matrix)

# Criar o heatmap com ggplot2
ggplot(correlation_data, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), 
                       space = "Lab", name="Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Heatmap", x = "Variables", y = "Variables")


# Depois de verificar a matriz de coreelação verificou-se que não há correlações
# significantes, sendo necesário fazer fazer o PC

### PC

pc = prcomp(data.numeric, scale. = TRUE)
pc
summary(pc)



# CATEGORIZAÇÃO DOS DADOS: foi feita de duas formas

# 1ª categorização dos dados

quartiles <- quantile(data$song_popularity, probs = c(0, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
breaks <- c(quartiles[1:5], Inf)
labels <- c("Não_popular", "Pouco_popular", "Rádio_amigável", "Popular", "Viral")
data$song_category <- cut(data$song_popularity, breaks = breaks, labels = labels, include.lowest = TRUE)
summary(data$song_category)
data$song_popularity <- NULL
data$song_name<-NULL
head(data)


# 2ª categorização dos dados

breaks <- c(0, 35, 60, 100)
labels <- c("Not_popular","Radio_friendly","Popular")
data$song_category <- cut(data$song_popularity, breaks = breaks, labels = labels, include.lowest = TRUE)
head(data)
data$song_popularity <- NULL
data$song_name<-NULL
head(data)


# Aplicação do BAGGING

library(randomForest)
head(data.tr)
Adv.bagg=randomForest(song_category~.,data=data.tr,mtry=(ncol(data.tr)-1))
Adv.bagg

pred=predict(Adv.bagg,newdata=data.te,type="class")
pred1=predict(Adv.bagg,newdata=data.te,type="prob")
caret::confusionMatrix(pred,data.te$song_category)


varImpPlot(Adv.bagg)


head(pred1)
roc_curves <- multiclass.roc(data.te$song_category, pred1)
# Calculate AUC
auc_score <- auc(roc_curves)
# Print AUC (curva ROC)
print(auc_score)


## Bagging para 2000 árvores

Adv.bagg2=randomForest(song_popularity ~ ., data = data.tr, mtry = (ncol(data.tr) - 1), ntree = 2000)
Adv.bagg2
