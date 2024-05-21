

data=read.csv("data/song_data.csv")
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

#################CATEGORIZAÇÃO DOS DADOS##################

quartiles <- quantile(data$song_popularity, probs = c(0, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE)
breaks <- c(quartiles[1:5], Inf)
labels <- c("Não_popular", "Pouco_popular", "Rádio_amigável", "Popular", "Viral")
data$song_category <- cut(data$song_popularity, breaks = breaks, labels = labels, include.lowest = TRUE)
summary(data$song_category)
data$song_popularity <- NULL
data$song_name<-NULL
head(data)

#################ESTATÍSTICOS BÁSICOS#################

hist(data$song_popularity)
hist(data$song_duration_ms)
hist(data$acousticness)
hist(data$tempo)
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


#################LIMPEZA DE OUTLIERS#################

boxplot(data$song_duration_ms)

boxplot(data$instrumentalness)
ggplot(data, aes(x = song_category, y = song_duration_ms)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "song_duration_ms")


threshold <- 1000000  # limite dos outliers (limite)
data <- data[data$song_duration_ms <= threshold, ]
dim(data)
18835-18831 ##4 outliers eliminados

#####Variavel acousticness

ggplot(data, aes(x = song_category, y = acousticness)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Acousticness")

# Definir o limite (threshold)
threshold1 <- 0.98
data <- data[!(data$song_category == "Popular" & data$acousticness > threshold1), ]
head(data)
18835-18814 ##21 outliers eliminados


#####Variavel Energy
##Não é preciso eliminar outliers

ggplot(data, aes(x = song_category, y = energy
)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Energy")


#####Variavel Speechiness

ggplot(data, aes(x = song_category, y = speechiness
)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Speechiness")

threshold1 <- 0.9  
threshold2 <- 0.9 
threshold3 <- 0.75
threshold4 <- 0.5
data <- data[!(data$song_category == "Not_popular" & data$speechiness > threshold1), ]
data <- data[!(data$song_category == "Low_popularity" & data$speechiness > threshold2), ]
data <- data[!(data$song_category == "Radio_friendly" & data$speechiness > threshold3), ]
data <- data[!(data$song_category == "Popular" & data$speechiness > threshold4), ]

dim(data)
18831-18794 ##37 outliers eliminados

####Variavel Loudness

ggplot(data, aes(x = song_category, y = loudness
)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Loudness")
threshold1=-35
data <- data[!(data$loudness < threshold1), ]
dim(data)
18831-18788 ##43 outliers eliminados

####Variavel Speechiness

ggplot(data, aes(x = song_category, y = speechiness
)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Speechiness")

threshold1=0.85
data <- data[!(data$speechiness > threshold1), ]
dim(data)
18831-18783 ##48 outliers eliminados


##Variavel Tempo
ggplot(data, aes(x = song_category, y = tempo
)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Tempo")
threshold1=230
threshold2=0
data <- data[!(data$tempo > threshold1), ]
data <- data[!(data$tempo <= threshold2), ]
dim(data)
18831-18780 ##51 outliers eliminados


##Variavel Danceability
ggplot(data, aes(x = song_category, y = danceability
)) +
  geom_boxplot() +
  labs(title = "Box plot of Your Variable by Song Category",
       x = "Song Category",
       y = "Danceability")

##Não são eliminados outliers nesta variavel

##51/18831=0.003 percentagem de elementos removidos do dataset. 
##Limpeza muito minuciosa e pormenorizada.


# Visualizar o conjunto de dados limpo
print(data)

#### estamos em condições de aplicar as metodologias. 

df <- data.frame(data)

