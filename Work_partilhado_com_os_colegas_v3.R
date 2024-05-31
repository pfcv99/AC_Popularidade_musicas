
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
library(dplyr)
install.packages("rdist")
library(rdist)
install.packages("dbscan")
library(dbscan)
install.packages("Rlof")
library(Rlof)



# 1. verificar valores ausentes
sapply(data, function(x) sum(is.na(x)))


# 2. Identificação e remoção de duplicados
data=data[!duplicated(data),]
dim(data)
head(data)
df_raw = data
dim(df_raw)


# 3. Remoção de outliers
# Criar o modelo Local Outlier Factor (LOF) para detecção de outliers
x = df_raw
lof_outlier = lof(x, k = 20)

# Obter os escores de outliers para cada ponto de dados
outlier_scores = lof_outlier

# Identificar os índices dos pontos de dados outliers
outlier_indices = outlier_scores > quantile(outlier_scores, probs = 0.99)
print("Outliers:")
print(which(outlier_indices))

# Remover os outliers do dataframe
df_LOF = x[!outlier_indices, ]
dim(df_LOF)
data = df_LOF
dim(data)



################### Aplicação da metodologia BAGGING

# Criação do conjunto de treino e de teste
ind.tr=sample(1:nrow(data),0.8*nrow(data))
data.tr=data[ind.tr,]
data.te=data[-ind.tr,]

head(data.tr)
song.bagg = randomForest(song_popularity ~ ., data = data.tr, 
                          mtry = (ncol(data.tr) - 1))
song.bagg

# Prever os valores de popularidade das músicas no conjunto de teste
pred=predict(song.bagg,newdata=data.te)

# RMSE
sqrt(mean((pred-data.te$song_popularity)^2))

# MAE (média dos valores absolutos)
mean(abs(pred - data.te$song_popularity))

# R-sqrt (Coeficiente de deteminação)
cor(pred, data.te$song_popularity)^2

importance(song.bagg) # verificar as variáveis mais influentes no modelo

varImpPlot(song.bagg)



#################### Bagging para 2000 árvores

song.bagg2 = randomForest(song_popularity ~ ., data = data.tr, 
                          mtry = (ncol(data.tr) - 1), ntree = 2000)
song.bagg2

# Prever os valores de popularidade das músicas no conjunto de teste
pred=predict(song.bagg2,newdata=data.te)

# RMSE
sqrt(mean((pred-data.te$song_popularity)^2))
# MAE (média dos valores absolutos)
mean(abs(pred - data.te$song_popularity))

# R-sqrt (Coeficiente de deteminação)
cor(pred, data.te$song_popularity)^2
importance(song.bagg) # verificar as variáveis mais influentes no modelo
varImpPlot(song.bagg)




############ Aplicação da regularização para melhorar a performance do modelo
library(caret)
library(glmnet)

# Dividir os dados em variáveis preditoras (x) e variável resposta (y)
x_train = as.matrix(data.tr[, -which(names(data.tr) == "song_popularity")])
y_train = data.tr$song_popularity

x_test = as.matrix(data.te[, -which(names(data.tr) == "song_popularity")])
y_test = data.te$song_popularity

x = as.matrix(data[, -ncol(data)])  # todas as colunas menos a variável preditora
y = data[, ncol(data)]  # coluna da variável resposta

# Definir uma sequência de valores de lambda para testar
lambda_seq <- 10^seq(2, -2, by = -0.1)




# 1. Treinar o modelo RIDGE
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq)

# Cross-validation para encontrar o melhor valor de lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, lambda = lambda_seq)
best_lambda_ridge <- cv_ridge$lambda.min

# Predições e avaliação no conjunto de teste
ridge_predictions <- predict(ridge_model, s = best_lambda_ridge, newx = x_test)
ridge_rmse <- sqrt(mean((ridge_predictions - y_test)^2))
print(paste("Ridge RMSE:", ridge_rmse))

# Coeficientes do modelo Ridge
ridge_coefficients <- predict(ridge_model, s = best_lambda_ridge, type = "coefficients")
print("Ridge Coefficients:")
print(ridge_coefficients)




# 2. Treinar o modelo LASSO
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda_seq)

# Cross-validation para encontrar o melhor valor de lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, lambda = lambda_seq)
best_lambda_lasso <- cv_lasso$lambda.min

# Predições e avaliação no conjunto de teste
lasso_predictions <- predict(lasso_model, s = best_lambda_lasso, newx = x_test)
lasso_rmse <- sqrt(mean((lasso_predictions - y_test)^2))
print(paste("Lasso RMSE:", lasso_rmse))

# Coeficientes do modelo Lasso
lasso_coefficients <- predict(lasso_model, s = best_lambda_lasso, type = "coefficients")
print("Lasso Coefficients:")
print(lasso_coefficients)


# Validação cruzada para Ridge Regression
set.seed(123)  # Defina a semente para reprodutibilidade
cv_ridge <- cv.glmnet(X, y, alpha = 0)  # alpha = 0 para Ridge Regression
best_lambda_ridge <- cv_ridge$lambda.min
print(paste("Best lambda for Ridge Regression:", best_lambda_ridge))

# Validação cruzada para Lasso Regression
set.seed(123)  # Defina a semente para reprodutibilidade
cv_lasso = cv.glmnet(X, y, alpha = 1)  # alpha = 1 para Lasso Regression
best_lambda_lasso = cv_lasso$lambda.min
print(paste("Best lambda for Lasso Regression:", best_lambda_lasso))


# Ajustar o modelo final com o melhor lambda
final_ridge_model = glmnet(X, y, alpha = 0, lambda = best_lambda_ridge)
print(coef(final_ridge_model))

final_lasso_model = glmnet(X, y, alpha = 1, lambda = best_lambda_lasso)
print(coef(final_lasso_model))



####################### Regularização Ridge no BAGGING

# Definir o controle de treinamento para validação cruzada
validacruz = trainControl(method = "cv", number = 5)

# Definir a grade de hiperparâmetros para o modelo Ridge
alpha = 0  # Lasso (alpha = 1) ou Ridge (alpha = 0)
lambda = 10^seq(-3, 3, by = 0.1)  # Varie os valores de lambda

# Ajustar o modelo Ridge usando Bagging
set.seed(123)
song.bagg_ridge_model = train(song_popularity ~ ., 
                            data = data.tr, 
                            method = "glmnet", 
                            trControl = validacruz, 
                            tuneGrid = expand.grid(alpha = alpha, lambda = lambda))

# Fazer previsões no conjunto de teste
predictions = predict(song.bagg_ridge_model, newdata = data.te)

# Avaliar o desempenho do modelo
mse = mean((predictions - data.te$song_popularity)^2)
rmse = sqrt(mse)
print(paste("RMSE:", rmse))





############### ESTATÍSTICOS BÁSICOS #################

# Aplicamos essas técnicas estatísticas para entender melhor o 
# conjunto de dados e ter alguma intuição sobre como os algoritmos se 
# comportarão.

# Histograma de todas as variáveis
head(data)
dim(data)
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



# Matriz de correlações dos dados e heatmap de correlações


data.numeric = data[,sapply(data, is.numeric)]
dim(data.numeric)

install.packages("reshape2")
library(reshape2)

correlation_matrix = cor(data.numeric)
correlation_data = melt(correlation_matrix)

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


# Sendo que o Bagging não forneceu grandes resultados, optei por 
# categorizar as variáveis e aplicar a Classificação




############# CATEGORIZAÇÃO DOS DADOS 

breaks <- c(0, 35, 60, 100) # intervalos
labels <- c("Não_popular", "Popular", "Frequência_rádio") # Criar etiquetas para categorias
data$song_category <- cut(data$song_popularity, breaks = breaks, labels = labels, include.lowest = TRUE)
head(data)
data
dim(data)

summary(data$song_category)
data$song_popularity <- NULL
data$song_name<-NULL
head(data)
dim(data)
summary(data$song_category)


dim(data)
View(data)


##################### Bagging com dados categorizados

sapply(data, class)
data$song_category=as.factor(data$song_category)
data$time_signature=as.factor(data$time_signature)
data$audio_mode=as.factor(data$audio_mode)

library(readr)
set.seed(1234)
ind.tr=sample(1:nrow(data),0.8*nrow(data))
data.tr=data[ind.tr,]
data.te=data[-ind.tr,]



library(randomForest)
head(data.tr)
song.bagg3=randomForest(song_category~.,data=data.tr,mtry=(ncol(data.tr)-1))
song.bagg3




pred=predict(song.bagg3,newdata=data.te,type="class")
pred1=predict(song.bagg3,newdata=data.te,type="prob")
caret::confusionMatrix(pred,data.te$song_category)

varImpPlot(song.bagg3)


##  Bagging com dados categorizados para 2000 árvores

song.bagg4 = randomForest(song_category ~ ., data = data.tr, 
                          mtry = (ncol(data.tr) - 1), ntree = 2000)
song.bagg4

pred=predict(song.bagg4,newdata=data.te,type="class")
pred1=predict(song.bagg4,newdata=data.te,type="prob")
caret::confusionMatrix(pred,data.te$song_category)

varImpPlot(song.bagg4)
boxplot(data$song_category)



head(pred1)
pred



######## Curva ROC

if (!requireNamespace("pROC", quietly = TRUE)) {
  install.packages("pROC")
}

# Carregar a biblioteca 'pROC'
library(pROC)



roc_curves <- multiclass.roc(data.te$song_category, pred1)

# Calculate AUC
auc_score <- auc(roc_curves)
# Print AUC (curva ROC)
print(auc_score) ## o print diz que em 59% o classificador clasifica bem. 


####### Algums estatísticos adicionais

# 1. variável song_duartion_ms

# variância
var(data$song_duration_ms[data$song_category == "Não_popular"])
var(data$song_duration_ms[data$song_category == "Popular"])
var(data$song_duration_ms[data$song_category == "Frequência_rádio"])


# média
mean(data$song_duration_ms[data$song_category == "Não_popular"])/60000
mean(data$song_duration_ms[data$song_category == "Popular"])/60000
mean(data$song_duration_ms[data$song_category == "Frequência_rádio"])/60000

# mediana
median(data$song_duration_ms[data$song_category == "Não_popular"])/60000
median(data$song_duration_ms[data$song_category == "Frequência_rádio"])/60000
median(data$song_duration_ms[data$song_category == "Popular"])/60000


# 2. acousticness

# variância
var(data$acousticness[data$song_category == "Não_popular"])
var(data$acousticness[data$song_category == "Frequência_rádio"])
var(data$acousticness[data$song_category == "Popular"])


# média
mean(data$acousticness[data$song_category == "Não_popular"])
mean(data$acousticness[data$song_category == "Frequência_rádio"])
mean(data$acousticness[data$song_category == "Popular"])


# mediana
median(data$acousticness[data$song_category == "Não_popular"])
median(data$acousticness[data$song_category == "Frequência_rádio"])
median(data$acousticness[data$song_category == "Popular"])




