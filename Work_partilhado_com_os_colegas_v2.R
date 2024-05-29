
data=read.csv("song_data.csv")
head(data)
dim(data)

data = data[,sapply(data, is.numeric)] # dados sem a variável song_name
dim(data)


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


# 1. verificar valores ausentes
sapply(data, function(x) sum(is.na(x)))

# 2. Remoção de musicas com tempo 0
data = data[data$tempo!=0, ]
dim(data)

# 3. Identificação e remoção de duplicados
data=data[!duplicated(data),]
dim(data)


# BAGGING sem remover outliers
ind.tr=sample(1:nrow(data),0.6*nrow(data))
data.tr=data[ind.tr,]
data.te=data[-ind.tr,]

library(randomForest)
head(data.tr)
song.bagg1 = randomForest(song_popularity ~ ., data = data.tr, 
                          mtry = (ncol(data.tr) - 1))
song.bagg1

# Prever os valores de popularidade das músicas no conjunto de dados de teste
pred=predict(song.bagg1,newdata=data.te)
# RMSE
sqrt(mean((pred-data.te$song_popularity)^2))


## Bagging para 2000 árvores

song.bagg2 = randomForest(song_popularity ~ ., data = data.tr, 
                          mtry = (ncol(data.tr) - 1), ntree = 2000)
song.bagg2

# Prever os valores de popularidade das músicas no conjunto de dados de teste
pred=predict(song.bagg1,newdata=data.te)
# RMSE
sqrt(mean((pred-data.te$song_popularity)^2))


############### ESTATÍSTICOS BÁSICOS #################

# Aplicamos essas técnicas estatísticas prévias para entender melhor o 
# conjunto de dados e ter alguma intuição sobre como os algoritmos se 
# comportarão.

# Histograma de todas as variáveis
head(data)
hist(data$song_popularity)
# Maior densidade populacional no intervalo 
# [40,80] com caudas mais suaves e um aumento no valor de popularidade 0.
boxplot(data$song_popularity)

hist(data$song_duration_ms)
# A maioria das músicas se concentra no intervalo [0,500000], 
# o que é equivalente a um intervalo de 0 a 8 minutos aproximadamente. 
# Alguns elementos também estão em um intervalo maior do que este, 
# mas com uma densidade extremamente reduzida.
boxplot(data$song_duration_ms)

hist(data$acousticness) # talvez não
# Histograma do acousticness [0,1] x---->1 a música tem elementos acústicos.
##A maioria das músicas não apresentan elementos acústicos
boxplot(data$acousticness)

hist(data$danceability)
boxplot(data$danceability)

hist(data$energy)
boxplot(data$energy)

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
# A maioria das músicas estão concentradas no intervalo [50,200] bpm(beats 
# por minuto)
boxplot(data$tempo)

hist(data$time_signature)
boxplot(data$time_signature)

hist(data$audio_valence)
boxplot(data$audio_valence)


# O proceso é análogo para as demais variáveis, mas os dados mais interessantes
# estão neste conjunto de histogramas.


# Matriz de correlações dos dados e heatmap de correlações

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


# Não são extraídas grandes conclusões do heatmap, pelo menos não novas 
# conclusões em relação a uma intuição prévia. Fenômenos como o fato de que 
# a acústica e a energia da música estejam inversamente relacionadas ou que 
# a energia e a dançabilidade estejam diretamente correlacionadas são 
# fenômenos que poderiam ser presumidos previamente.




############# CATEGORIZAÇÃO DOS DADOS ##################

breaks <- c(0, 35, 60, 100)
labels <- c("Não_popular","Rádio_amizade","Popular")
data$song_category <- cut(data$song_popularity, breaks = breaks, labels = labels, include.lowest = TRUE)
head(data)
data
dim(data)

data$song_popularity <- NULL
data$song_name<-NULL
head(data)
summary(data$song_category)

str(data)
dim(data)
View(data)


##################### BAGGING com os dados organizados ###############

sapply(data, class)
data$song_category=as.factor(data$song_category)
data$time_signature=as.factor(data$time_signature)
data$audio_mode=as.factor(data$audio_mode)

library(readr)
set.seed(1234)
ind.tr=sample(1:nrow(data),0.7*nrow(data))
data.tr=data[ind.tr,]
data.te=data[-ind.tr,]

library(randomForest)
head(data.tr)
song.bagg=randomForest(song_category~.,data=data.tr,mtry=(ncol(data.tr)-1))
song.bagg


pred=predict(song.bagg,newdata=data.te,type="class")
pred1=predict(song.bagg,newdata=data.te,type="prob")
caret::confusionMatrix(pred,data.te$song_category)


varImpPlot(song.bagg)


head(pred1)
roc_curves <- multiclass.roc(data.te$song_category, pred1)
# Calculate AUC
auc_score <- auc(roc_curves)
# Print AUC (curva ROC)
print(auc_score) ## o print diz que em 61% o classificador clasifica bem. 


#JUSTIFICAÇÃO DE RESULTADOS
##os dados são muito semelhantes e as variáveis com mais peso para a 
## classificação são song_duration e acousticness e são semelhantes. 

median(data$song_duration_ms[data$song_category == "Not_popular"])/60000
median(data$song_duration_ms[data$song_category == "Radio_friendly"])/60000
median(data$song_duration_ms[data$song_category == "Popular"])/60000


median(data$acousticness[data$song_category == "Not_popular"])
median(data$acousticness[data$song_category == "Radio_friendly"])
median(data$acousticness[data$song_category == "Popular"])


mean(data$loudness[data$song_category == "Not_popular"])
mean(data$loudness[data$song_category == "Radio_friendly"])
mean(data$loudness[data$song_category == "Popular"])


median(data$loudness[data$song_category == "Not_popular"])
median(data$loudness[data$song_category == "Radio_friendly"])
median(data$loudness[data$song_category == "Popular"])

