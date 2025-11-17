# Packages ---------------------------------------------------------------------
library(data.table)
library(caTools)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(corrplot)
library(hms)

# Import Data ------------------------------------------------------------------
setwd("~/Desktop/CBA Question Paper")
misinformation.dt <- fread("misinformation2.csv", header = T, na.strings = c("NA", "na", "N/A", "", ".", "m", "M"))

# EDA Before Preprocessing -----------------------------------------------------
## General Info
class(misinformation.dt)
summary(misinformation.dt)
dim(misinformation.dt) # 500 rows & 26 columns
str(misinformation.dt)

## Check duplicated rows
sum(duplicated(misinformation.dt)) # no duplicated rows

## Check NA values
sum(is.na(misinformation.dt)) # no NA

## Categorical variables
unique(misinformation.dt$platform)
unique(misinformation.dt$country)
unique(misinformation.dt$city)
unique(misinformation.dt$timezone)

## Date range
min(misinformation.dt$timestamp)
max(misinformation.dt$timestamp)

# Clean Date & Time ------------------------------------------------------------
misinformation.dt[, cleaned_date := as.Date(date, format = "%d %m %Y")]
misinformation.dt[, cleaned_time := as_hms(time)]
class(misinformation.dt$cleaned_date)
class(misinformation.dt$cleaned_time)

# Factorize --------------------------------------------------------------------
## Binary 0/1
misinformation.dt[, author_verified := factor(author_verified, levels = c(0, 1), labels = c("Not Verified", "Verified"))]
misinformation.dt[, is_misinformation := factor(is_misinformation, levels = c(0, 1), labels = c("Genuine", "Misinformation"))]

## Text / Nominal
factor_cols <- c("platform", "month", "weekday", "country", "city", "timezone")
misinformation.dt[, (factor_cols) := lapply(.SD, factor), .SDcols = factor_cols]

## Order factors for readability
misinformation.dt[, month := factor(month, levels = month.name, ordered = T)]
misinformation.dt[, weekday := factor(weekday, levels = c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"), ordered = T)]

# EDA After Preprocessing ------------------------------------------------------
str(misinformation.dt)
summary(misinformation.dt)
dim(misinformation.dt) # 500 rows and 28 columns

# Early Visualizations ---------------------------------------------------------
## Distribution by country
ggplot(misinformation.dt, aes(x = is_misinformation, fill = is_misinformation)) +
  geom_bar(width = 0.7) +
  labs(title = "Distribution of Misinformation vs Genuine Posts by Country", 
       x = "Post Type",
       y = "Number of Posts",
       fill = "Legend") +
  facet_grid(.~ country)

ggplot(misinformation.dt, aes(x = country, fill = is_misinformation)) +
  geom_bar(position = "fill", width = 0.7) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Proportion of Misinformation vs Genuine Posts by Country",
       x = "Country",
       y = "Proportion of Posts",
       fill = "Legend")

## Quick Insight: ~ 50% are misinformation posts across countries

## Posts by platform (Most prevalent in Twitter)
ggplot(misinformation.dt, aes(x = is_misinformation, fill = is_misinformation)) +
  geom_bar(width = 0.7) +
  labs(title = "Distribution of Misinformation vs Genuine Posts by Platform", 
       x = "Post Type",
       y = "Number of Posts",
       fill = "Post Type") +
  facet_grid(.~ platform)

ggplot(misinformation.dt, aes(x = platform, fill = is_misinformation)) +
  geom_bar(position = "fill", width = 0.7) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Proportion of Misinformation vs Genuine Posts by Platform",
       x = "Platform",
       y = "Proportion of Posts",
       fill = "Post Type")

## Quick Insight: more prevalent & higher proportion in twitter

## Posts by fact check count
ggplot(misinformation.dt, aes(x = is_misinformation, y = external_factchecks_count, fill = is_misinformation)) +
  geom_jitter(aes(color = is_misinformation), width = 0.1, height = 0.5, alpha = 0.5) +
  scale_y_continuous(breaks = 0:5) +
  labs(title = "External Factcheck Count Distribution by Post Type",
       x = "Post Type",
       y = "External Factcheck Count",
       color = "Legend",
       fill = "Legend")

## Quick Insight: most misinformation posts have fewer fact checks

## Posts by source domain reliability
ggplot(misinformation.dt, aes(x = is_misinformation, y = source_domain_reliability, fill = is_misinformation)) +
  geom_boxplot(width = 0.5) +
  labs(title = "Source Domain Reliability by Post Type", 
       x = "Post Type",
       y = "Source Domain Reliability Score",
       fill = "Legend")

## Quick Insight: misinformation posts come from less reliable sources hence lower score

# Distribution of post types ---------------------------------------------------
ggplot(misinformation.dt, aes(x = is_misinformation, fill = is_misinformation)) +
  geom_bar(width = 0.5) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
  labs(title = "Distribution of Genuine vs Misinformation Posts",
       x = "Post Type",
       y = "Number of Posts",
       fill = "Legend") +
  ylim(0, 300)

## Quick Insight: quite evenly distributed for the target y variable

# Drop Columns for Model -------------------------------------------------------
drop_cols <- c("id", "author_id", "timestamp", "date", "time", "cleaned_date", "cleaned_time", "timezone", "city", "month", "weekday")
model.dt <- misinformation.dt[, !drop_cols, with = F]

dim(model.dt)
summary(model.dt)

# Train-test Split -------------------------------------------------------------
set.seed(123)
train <- sample.split(model.dt$is_misinformation, SplitRatio = 0.7)
trainset <- model.dt[train == T]
testset <- model.dt[train == F]

summary(trainset$is_misinformation)

# Regression -------------------------------------------------------------------
## Model 1 (Full)
m1 <- glm(is_misinformation ~ ., data = trainset, family = binomial)
summary(m1)

## Model 2 (Simplified, keep significant - manual)
m2 <- glm(is_misinformation ~ external_factchecks_count + source_domain_reliability, data = trainset, family = binomial)
summary(m2)

## Model confirmation using step function
m.test <- glm(is_misinformation ~ ., data = trainset, family = binomial)
m.step <- step(m.test, direction = "both", trace = T)
summary(m.step)

## AIC
AIC(m1, m2, m.step)

## Confusion Matrix on testset using m2
prob.test <- predict(m2, newdata = testset, type = "response")
threshold1 <- 0.5
m2.predict.test <- ifelse(prob.test > threshold1, "Misinformation", "Genuine")
conf.mtx.lr <- table(Actual = relevel(testset$is_misinformation, ref = "Misinformation"), 
                     Predicted = relevel(as.factor(m2.predict.test), ref = "Misinformation"), deparse.level = 2)
conf.mtx.lr

## Plot Matrix
conf.mtx.lr.df <- as.data.frame(conf.mtx.lr)

ggplot(conf.mtx.lr.df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white", width = 0.9, height = 0.9) +
  geom_text(aes(label = Freq), color = "black", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix (Logistic Regression)",
       x = "Predicted",
       y = "Actual",
       fill = "Frequency") +
  theme(legend.position = "none")

## Calculations
tp.lr <- as.numeric(conf.mtx.lr[1, 1])
fp.lr <- as.numeric(conf.mtx.lr[2, 1])
tn.lr <- as.numeric(conf.mtx.lr[2, 2])
fn.lr <- as.numeric(conf.mtx.lr[1, 2])

accuracy.lr <- (tp.lr + tn.lr) / sum(conf.mtx.lr)
fpr.lr <- fp.lr / (fp.lr + tn.lr)
fnr.lr <- fn.lr / (fn.lr + tp.lr)
overall_error.lr <- 1 - accuracy.lr

# CART -------------------------------------------------------------------------
## Grow Maximal Tree
m3 <- rpart(is_misinformation ~ ., data = trainset, 
            method = "class", control = rpart.control(minsplit = 2, cp = 0))

printcp(m3)
plotcp(m3)
print(m3)

rpart.plot(m3, nn = T, main = "Maximal Tree")

## Compute CP
CVerror.cap <- m3$cptable[which.min(m3$cptable[, "xerror"]), "xerror"] + m3$cptable[which.min(m3$cptable[, "xerror"]), "xstd"]

i <- 1
while (m3$cptable[i, 4] > CVerror.cap) {
  i <- i + 1
}

cp1 <- ifelse(i > 1, sqrt(m3$cptable[i, 1] * m3$cptable[i - 1, 1]), 1)

## Prune Tree
m3.pruned <- prune(m3, cp = cp1)

summary(m3.pruned)
printcp(m3.pruned)
print(m3.pruned)

rpart.plot(m3.pruned, nn = T, main = "Optimal Tree")

## Confusion Matrix on testset using m3.pruned
cart.pred.prob <- predict(m3.pruned, newdata = testset, type = "class")

conf.mtx.cart <- table(Actual = relevel(testset$is_misinformation, ref = "Misinformation"), 
                     Predicted = relevel(cart.pred.prob, ref = "Misinformation"), deparse.level = 2)
conf.mtx.cart

## Plot Matrix
conf.mtx.cart.df <- as.data.frame(conf.mtx.cart)

ggplot(conf.mtx.cart.df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white", width = 0.9, height = 0.9) +
  geom_text(aes(label = Freq), color = "black", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  labs(title = "Confusion Matrix (CART)",
       x = "Predicted",
       y = "Actual",
       fill = "Frequency") +
  theme(legend.position = "none")

## Calculations
tp.cart <- as.numeric(conf.mtx.cart[1, 1])
fp.cart <- as.numeric(conf.mtx.cart[2, 1])
tn.cart <- as.numeric(conf.mtx.cart[2, 2])
fn.cart <- as.numeric(conf.mtx.cart[1, 2])

accuracy.cart <- (tp.cart + tn.cart) / sum(conf.mtx.cart)
fpr.cart <- fp.cart / (fp.cart + tn.cart)
fnr.cart <- fn.cart / (fn.cart + tp.cart)
overall_error.cart <- 1 - accuracy.cart

# Tabulation -------------------------------------------------------------------
## Logistic Regression metrics
lr.data <- data.frame(Model = "Logistic Regression", 
                      Model_Complexity = "2 X-variables", 
                      False_Positive_Rate = round(fpr.lr, 3), 
                      False_Negative_Rate = round(fnr.lr, 3),
                      Overall_Error = round(overall_error.lr, 3))

## CART metrics
cart.data <- data.frame(Model = "CART", 
                        Model_Complexity = "4 terminal nodes",
                        False_Positive_Rate = round(fpr.cart, 3),
                        False_Negative_Rate = round(fnr.cart, 3),
                        Overall_Error = round(overall_error.cart, 3))

## Combine into 1 table
results.table <- rbind(lr.data, cart.data)
results.table

# Variable Importance (CART) ---------------------------------------------------
m3.pruned$variable.importance

par(mar = c(11, 4, 4, 2)) # adjust margin

barplot(m3.pruned$variable.importance,
        main = "Variable Importance (CART)",
        col = "steelblue",
        ylab = "Importance Score",
        las = 2)

## Quick Insight: source_domain_reliability most important

# OR CI ------------------------------------------------------------------------
OR.m2 <- exp(coef(m2))
OR.m2
OR.CI.m2 <- exp(confint(m2))
OR.CI.m2

## Quick Insight: OR < 1, negatively associated. OR CI does not have 1 in range.

# Corr Plot --------------------------------------------------------------------
## Convert factor target to numeric (1 = Misinformation, 0 = Genuine)
misinfo_numeric <- ifelse(misinformation.dt$is_misinformation == "Misinformation", 1, 0)

## Create a data frame for correlation
corr.df <- data.frame(is_misinformation = misinfo_numeric,
                      external_factchecks_count = misinformation.dt$external_factchecks_count,
                      source_domain_reliability = misinformation.dt$source_domain_reliability)

## Compute & plot correlation
corr_matrix <- cor(corr.df, use = "complete.obs")
corr_matrix

par(mar = c(5, 4, 4, 2)) # re adjust margins

corrplot(corr_matrix, method = "circle", type = "upper",
         addCoef.col = "white", tl.col = "black", tl.srt = 45,
         title = "Correlation Between Target & Predictors", mar = c(3,0,2,0))

## Quick Insight: source_domain_reliability higher negative correlation