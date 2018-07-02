---
title: "LDA_QDA"
author: "Julien Jaber"
date: "6/21/2018"
output:
  html_document: 
    keep_md: yes
  pdf_document: default
---

# 1) LDA


```r
library(mvtnorm)
```


```r
set.seed(100)
train_idx <- sample(nrow(iris), 90)
train_set <- iris[train_idx, ]
test_set <- iris[-train_idx, ]
```


**Function lda_fit outputs prior probabilities, group means, and covariance matrix**


```r
lda_fit <- function(X, y) {
  
  nk <- as.vector(table(y))
  n = length(y)
  k <- length(unique(y))
  levs <- levels(y)
  W <- 0
  
  centroids <- matrix(0, ncol(X), k)
  for (j in 1:ncol(X)) {        #Find centroid for every column
  centroids[j,] <- tapply(X[,j], y, FUN=mean)
  }
  
  dimnames(centroids) <- list(colnames(X), levels(y))
  
  for (j in 1:k) { #LOOPING THROUGH all classes
    Xk <- scale(X[y == levs[j], ], scale = FALSE) #Xk are all observations centered
    W <- W + t(Xk) %*% Xk
  }
  
  pi_hat <- nk/n
  mu_hat <- t(centroids)
  sigma_hat <- 1/(n - k) * W
  
  l1 <- list(pi_hat, mu_hat, sigma_hat)
  names(l1) <- c("pi_hat", "mu_hat", "sigma_hat")  

  return(l1)
}

fitter_lda <- lda_fit(train_set[, 1:4], train_set$Species)
```


**Function lda_predict() outputs class and posterior probabilities**


```r
lda_predict <- function(fit, newdata) {
  
  mat <- matrix(0, ncol = 4, nrow = nrow(newdata))
  ThreeSaver <- c()
  
  for (i in 1:nrow(newdata)) { #loop through rows
    for (k in 1:length(fit$pi_hat)) {  #loops through all groups
      
      delta = log(fit$pi_hat[k]) - 0.5 * t(as.matrix(fit$mu_hat[k,])) %*%   solve(as.matrix(fit$sigma_hat)) %*% as.matrix(fit$mu_hat[k,]) + t(as.matrix(fit$mu_hat[k,])) %*% solve(as.matrix(fit$sigma_hat)) %*% t(as.matrix(newdata[i,]))
      
      ThreeSaver <- c(ThreeSaver, delta)
    }
    
    mat[i, ] = c(ThreeSaver, 0)
    ThreeSaver <- c() #make it empty again
  }
  
  classer = mat[, 4]
  
  probMatrix <- matrix(0, ncol = length(fit$pi_hat) , nrow = nrow(newdata)) #creates empty probability matrix
  
  for (i in 1:nrow(newdata)) {  #loops through all rows
    for (k in 1:length(fit$pi_hat)) {   #loops through species
      
    num <- dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[k, ], fit$sigma_hat) * fit$pi_hat[k]
    
    denom <- fit$pi_hat[1] * dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[1, ], fit$sigma_hat) + fit$pi_hat[2] * dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[2, ], fit$sigma_hat) + fit$pi_hat[3] * dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[3, ], fit$sigma_hat)
    
    total = num / denom
    probMatrix[i, k] = total
    }
  }
  largestElem <- apply(probMatrix, 1, which.max) #INDEX OF LARGEST ELEMENT

  largestElem <- gsub(1, "setosa", largestElem)
  largestElem <- gsub(2, "versicolor", largestElem)
  largestElem <- gsub(3, "virginica", largestElem)
  
  l1 <- list(largestElem, probMatrix)
  names(l1) <- c("class", "posterior")
  return(l1)
}


lda_predict(fitter_lda, test_set[, 1:4])
```

```
## $class
##  [1] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
##  [6] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
## [11] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
## [16] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
## [21] "setosa"     "setosa"     "setosa"     "setosa"     "versicolor"
## [26] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
## [31] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
## [36] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
## [41] "versicolor" "virginica"  "virginica"  "virginica"  "virginica" 
## [46] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
## [51] "virginica"  "virginica"  "virginica"  "versicolor" "virginica" 
## [56] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
## 
## $posterior
##               [,1]         [,2]         [,3]
##  [1,] 1.000000e+00 3.769716e-19 3.141057e-37
##  [2,] 1.000000e+00 1.937871e-15 1.605356e-32
##  [3,] 1.000000e+00 1.643005e-14 4.273595e-31
##  [4,] 1.000000e+00 1.675984e-19 1.351472e-37
##  [5,] 1.000000e+00 3.623415e-18 3.803445e-35
##  [6,] 1.000000e+00 1.616963e-17 4.516847e-35
##  [7,] 1.000000e+00 1.963997e-16 4.357481e-34
##  [8,] 1.000000e+00 1.591334e-20 4.773945e-39
##  [9,] 1.000000e+00 3.017050e-16 7.212738e-34
## [10,] 1.000000e+00 5.059445e-17 1.011268e-34
## [11,] 1.000000e+00 4.647495e-21 5.039500e-39
## [12,] 1.000000e+00 4.599811e-18 1.801228e-35
## [13,] 1.000000e+00 2.799797e-21 7.890494e-40
## [14,] 1.000000e+00 8.479053e-19 7.300365e-37
## [15,] 1.000000e+00 1.069540e-14 2.581836e-31
## [16,] 1.000000e+00 2.396470e-15 2.498782e-32
## [17,] 1.000000e+00 2.000926e-18 2.000202e-36
## [18,] 1.000000e+00 8.511598e-18 1.753128e-35
## [19,] 1.000000e+00 1.653690e-18 4.979010e-36
## [20,] 1.000000e+00 4.969807e-16 5.453200e-33
## [21,] 1.000000e+00 4.432742e-13 7.602627e-28
## [22,] 1.000000e+00 4.492056e-14 2.371842e-30
## [23,] 1.000000e+00 3.023084e-20 1.229983e-38
## [24,] 1.000000e+00 1.307534e-17 2.901873e-35
## [25,] 1.664323e-19 9.991072e-01 8.928238e-04
## [26,] 2.075400e-19 9.984676e-01 1.532398e-03
## [27,] 3.502230e-18 9.984625e-01 1.537523e-03
## [28,] 1.354943e-17 9.982931e-01 1.706908e-03
## [29,] 1.501675e-12 9.999910e-01 8.985665e-06
## [30,] 1.308946e-13 9.999984e-01 1.578309e-06
## [31,] 9.421145e-25 8.732669e-01 1.267331e-01
## [32,] 1.182701e-18 9.996998e-01 3.002342e-04
## [33,] 5.295525e-16 9.998659e-01 1.341060e-04
## [34,] 4.050504e-20 9.903970e-01 9.603048e-03
## [35,] 2.454734e-14 9.999888e-01 1.120381e-05
## [36,] 2.383628e-18 9.981832e-01 1.816773e-03
## [37,] 1.087163e-15 9.999723e-01 2.769452e-05
## [38,] 1.407013e-12 9.999993e-01 7.435740e-07
## [39,] 1.310187e-16 9.998005e-01 1.995127e-04
## [40,] 6.135594e-16 9.999233e-01 7.670661e-05
## [41,] 1.620311e-16 9.998415e-01 1.585179e-04
## [42,] 6.929263e-46 3.792669e-08 1.000000e+00
## [43,] 3.509368e-40 8.185561e-06 9.999918e-01
## [44,] 2.881906e-29 4.067540e-02 9.593246e-01
## [45,] 1.745907e-35 1.906362e-03 9.980936e-01
## [46,] 2.785505e-34 5.263818e-04 9.994736e-01
## [47,] 2.413491e-41 1.187906e-06 9.999988e-01
## [48,] 3.412271e-33 9.792262e-04 9.990208e-01
## [49,] 1.155846e-30 2.082888e-02 9.791711e-01
## [50,] 3.826006e-26 1.632384e-01 8.367616e-01
## [51,] 5.391618e-27 4.267053e-01 5.732947e-01
## [52,] 1.138812e-35 1.222240e-03 9.987778e-01
## [53,] 1.642127e-30 6.320417e-03 9.936796e-01
## [54,] 2.589495e-24 8.590630e-01 1.409370e-01
## [55,] 6.466247e-40 1.229023e-05 9.999877e-01
## [56,] 1.319374e-25 2.023759e-01 7.976241e-01
## [57,] 4.000609e-32 1.762510e-03 9.982375e-01
## [58,] 4.026577e-40 4.837705e-06 9.999952e-01
## [59,] 2.472383e-41 7.041958e-07 9.999993e-01
## [60,] 7.316964e-32 7.944641e-03 9.920554e-01
```

```r
p2 <- lda_predict(fitter_lda, test_set[, 1:4])$class
```


**Classification with LDA**


```r
library(MASS)
training <- c(1:47, 51:97, 101:146)
testing <- c(48:50, 98:100, 147:150)
irisDataset = iris
irisDataset$enumerate = 1:150

trained <- subset(iris, irisDataset$enumerate %in% training)
tested <- subset(iris, irisDataset$enumerate %in% testing)

fitter <- lda_fit(trained[, 1:4], trained[, 5])
predictor <- lda_predict(fitter, tested[, 1:4])

fitter
```

```
## $pi_hat
## [1] 0.3357143 0.3357143 0.3285714
## 
## $mu_hat
##            Sepal.Length Sepal.Width Petal.Length Petal.Width
## setosa         5.008511    3.429787     1.463830   0.2489362
## versicolor     5.953191    2.772340     4.289362   1.3319149
## virginica      6.619565    2.973913     5.584783   2.0282609
## 
## $sigma_hat
##              Sepal.Length Sepal.Width Petal.Length Petal.Width
## Sepal.Length   0.27084678  0.09672053   0.16682306  0.03908908
## Sepal.Width    0.09672053  0.11913165   0.05524487  0.03340797
## Petal.Length   0.16682306  0.05524487   0.18140540  0.04254695
## Petal.Width    0.03908908  0.03340797   0.04254695  0.04345135
```

```r
predictor
```

```
## $class
##  [1] "setosa"     "setosa"     "setosa"     "versicolor" "versicolor"
##  [6] "versicolor" "virginica"  "virginica"  "virginica"  "virginica" 
## 
## $posterior
##               [,1]         [,2]         [,3]
##  [1,] 1.000000e+00 1.902934e-18 2.066054e-37
##  [2,] 1.000000e+00 2.710046e-23 1.385488e-43
##  [3,] 1.000000e+00 1.655456e-20 3.601589e-40
##  [4,] 2.098963e-18 9.999561e-01 4.392786e-05
##  [5,] 2.560763e-10 1.000000e+00 1.273872e-08
##  [6,] 7.377643e-19 9.999373e-01 6.272708e-05
##  [7,] 2.150458e-35 9.801042e-03 9.901990e-01
##  [8,] 1.630159e-34 4.840281e-03 9.951597e-01
##  [9,] 5.482963e-40 2.258174e-05 9.999774e-01
## [10,] 5.273542e-33 2.312620e-02 9.768738e-01
```


# QDA

**Function lda_fit outputs prior probabilities, group means, and covariance matrixes for every class**


```r
qda_fit <- function(X, y) {
  
  nk <- as.vector(table(y))
  n = length(y)
  k <- length(unique(y))
  levs <- levels(y)
  W <- 0
  
  centroids <- matrix(0, ncol(X), k)
  for (j in 1:ncol(X)) {                #Find centroid for every column
  centroids[j,] <- tapply(X[,j], y, FUN=mean)
  }
  
  dimnames(centroids) <- list(colnames(X), levels(y))
  
  k = 3; p = ncol(X)
  arr <- array(NA, c(p,p,k)) #array with dim = p x p, and k matrices
  
  for (j in 1:k) { #Loops through all groups
    Xk <- scale(X[y == levs[j], ], scale = FALSE) #Xk are all observations centered
    W <- W + t(Xk) %*% Xk
    arr[ , , j] = 1/(50 - 1) * W
    W = 0
  }
  
  pi_hat <- nk/n
  mu_hat <- t(centroids)
  
  
  l1 <- list(pi_hat, mu_hat, arr)
  names(l1) <- c("pi_hat", "mu_hat", "sigma_hat")  

  return(l1)
}


fitter_qda <- qda_fit(train_set[, 1:4], train_set$Species)
```

**Function lda_predict() outputs class and posterior probabilities**


```r
predict_qda <- function(fit, newdata) { #newdata is X
  
  mat <- matrix(0, ncol = 4, nrow = nrow(newdata))
  ThreeSaver <- c()
  
  for (i in 1:nrow(newdata)) { #loop through rows
    for (k in 1:length(fit$pi_hat)) {  #loops through all groups
      
      delta = log(fit$pi_hat[k]) - 0.5 * t(as.matrix(fit$mu_hat[k,])) %*%   solve(as.matrix(fit$sigma_hat[, , k])) %*% as.matrix(fit$mu_hat[k,]) + t(as.matrix(fit$mu_hat[k,])) %*% solve(as.matrix(fit$sigma_hat[, , k])) %*% t(as.matrix(newdata[i,]))
      
      ThreeSaver <- c(ThreeSaver, delta)
    }
    
    mat[i, ] = c(ThreeSaver, 0)
    ThreeSaver <- c() #make it empty again
  }
  
  classer = mat[, 4]
  
  probMatrix <- matrix(0, ncol = length(fit$pi_hat) , nrow = nrow(newdata)) #creates empty probability matrix
  
  for (i in 1:nrow(newdata)) {  #loops through all rows
    for (k in 1:length(fit$pi_hat)) {   #loops through species
      
    num <- dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[k, ], 
                   fit$sigma_hat[ , , k]) * fit$pi_hat[k]
    
    denom <- fit$pi_hat[1] * dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[1, ], fit$sigma_hat[, , 1]) + fit$pi_hat[2] * dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[2, ], fit$sigma_hat[, , 2]) + fit$pi_hat[3] * dmvnorm(c(newdata[i, 1], newdata[i, 2], newdata[i, 3], newdata[i, 4]), fit$mu_hat[3, ], fit$sigma_hat[, , 3])
    
    total = num / denom
    probMatrix[i, k] = total
    }
  }
  
  largestElem <- apply(probMatrix, 1, which.max) #INDEX OF LARGEST ELEMENT

  largestElem <- gsub(1, "setosa", largestElem)
  largestElem <- gsub(2, "versicolor", largestElem)
  largestElem <- gsub(3, "virginica", largestElem)

  
  #tot = colSums(mat != 0)
  l1 <- list(largestElem, probMatrix)
  names(l1) <- c("class", "posterior")
  return(l1)

}

predict_qda(fitter_qda, test_set[, 1:4])
```

```
## $class
##  [1] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
##  [6] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
## [11] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
## [16] "setosa"     "setosa"     "setosa"     "setosa"     "setosa"    
## [21] "setosa"     "setosa"     "setosa"     "setosa"     "versicolor"
## [26] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
## [31] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
## [36] "versicolor" "versicolor" "versicolor" "versicolor" "versicolor"
## [41] "versicolor" "virginica"  "virginica"  "virginica"  "virginica" 
## [46] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
## [51] "virginica"  "virginica"  "versicolor" "versicolor" "virginica" 
## [56] "virginica"  "virginica"  "virginica"  "virginica"  "virginica" 
## 
## $posterior
##                [,1]         [,2]         [,3]
##  [1,]  1.000000e+00 1.650368e-35 2.772040e-86
##  [2,]  1.000000e+00 2.453380e-25 1.244039e-71
##  [3,]  1.000000e+00 1.552165e-25 3.423533e-64
##  [4,]  1.000000e+00 3.767060e-37 9.198442e-86
##  [5,]  1.000000e+00 2.485326e-36 7.589643e-85
##  [6,]  1.000000e+00 2.676162e-32 5.288721e-79
##  [7,]  1.000000e+00 1.164445e-27 1.777459e-73
##  [8,]  1.000000e+00 1.402891e-39 2.594494e-95
##  [9,]  1.000000e+00 1.846193e-26 3.390783e-72
## [10,]  1.000000e+00 8.889567e-26 7.456940e-67
## [11,]  1.000000e+00 4.905096e-41 8.128892e-96
## [12,]  1.000000e+00 1.970824e-33 3.071067e-82
## [13,]  1.000000e+00 2.228789e-35 3.080030e-82
## [14,]  1.000000e+00 1.123187e-33 1.210288e-86
## [15,]  1.000000e+00 1.054832e-26 1.749556e-65
## [16,]  1.000000e+00 3.002103e-26 8.415256e-71
## [17,]  1.000000e+00 9.319948e-31 9.838150e-83
## [18,]  1.000000e+00 1.638260e-32 2.306660e-81
## [19,]  1.000000e+00 1.020938e-33 5.423469e-82
## [20,]  1.000000e+00 3.055474e-27 1.838429e-65
## [21,]  1.000000e+00 2.346493e-22 9.989980e-59
## [22,]  1.000000e+00 1.275192e-23 6.601509e-66
## [23,]  1.000000e+00 2.164939e-39 8.543797e-93
## [24,]  1.000000e+00 3.153143e-31 9.377805e-80
## [25,] 4.075091e-112 9.999676e-01 3.235434e-05
## [26,] 3.065234e-141 9.998048e-01 1.951510e-04
## [27,] 1.856175e-104 9.995985e-01 4.015199e-04
## [28,] 7.325266e-130 9.999951e-01 4.868711e-06
## [29,]  3.311744e-82 1.000000e+00 3.442282e-10
## [30,] 5.025040e-105 1.000000e+00 6.224805e-10
## [31,] 1.947578e-195 9.514843e-01 4.851570e-02
## [32,] 2.152796e-157 9.998822e-01 1.177506e-04
## [33,] 1.268733e-147 1.000000e+00 4.745455e-12
## [34,] 2.372023e-153 9.999196e-01 8.042918e-05
## [35,] 7.219733e-100 1.000000e+00 5.331951e-11
## [36,] 3.892529e-175 1.000000e+00 3.564525e-09
## [37,] 1.070144e-107 1.000000e+00 9.152515e-10
## [38,]  2.260505e-58 1.000000e+00 8.560882e-11
## [39,] 2.579113e-119 9.999998e-01 2.454717e-07
## [40,] 4.884048e-132 1.000000e+00 2.870679e-10
## [41,] 3.172386e-114 9.999999e-01 1.009099e-07
## [42,]  0.000000e+00 1.164419e-10 1.000000e+00
## [43,] 9.881313e-324 2.692252e-08 1.000000e+00
## [44,] 1.407995e-162 5.305237e-03 9.946948e-01
## [45,]  0.000000e+00 2.624852e-07 9.999997e-01
## [46,] 4.970551e-290 5.567638e-07 9.999994e-01
## [47,] 3.338462e-279 1.737098e-19 1.000000e+00
## [48,] 3.442631e-219 2.331561e-07 9.999998e-01
## [49,]  0.000000e+00 5.163914e-03 9.948361e-01
## [50,] 1.028410e-202 6.582541e-02 9.341746e-01
## [51,] 1.047264e-294 7.166426e-02 9.283357e-01
## [52,]  0.000000e+00 1.757551e-05 9.999824e-01
## [53,]  0.000000e+00 5.785912e-01 4.214088e-01
## [54,] 4.144337e-207 8.609913e-01 1.390087e-01
## [55,]  0.000000e+00 5.511168e-10 1.000000e+00
## [56,] 7.467301e-194 4.365370e-02 9.563463e-01
## [57,] 7.626629e-280 2.765663e-06 9.999972e-01
## [58,]  0.000000e+00 3.223423e-09 1.000000e+00
## [59,]  0.000000e+00 1.149043e-15 1.000000e+00
## [60,] 1.949684e-229 1.024305e-06 9.999990e-01
```

```r
p3 <- predict_qda(fitter_qda, test_set[, 1:4])$class
```

## 2.3) Classification with QDA


```r
training <- c(1:47, 51:97, 101:146)
testing <- c(48:50, 98:100, 147:150)
irisDataset$enumerate = 1:150

trained <- subset(iris, irisDataset$enumerate %in% training)
tested <- subset(iris, irisDataset$enumerate %in% testing)

fitter <- qda_fit(trained[, 1:4], trained[, 5])
predictor <- predict_qda(fitter, tested[, 1:4])

fitter
```

```
## $pi_hat
## [1] 0.3357143 0.3357143 0.3285714
## 
## $mu_hat
##            Sepal.Length Sepal.Width Petal.Length Petal.Width
## setosa         5.008511    3.429787     1.463830   0.2489362
## versicolor     5.953191    2.772340     4.289362   1.3319149
## virginica      6.619565    2.973913     5.584783   2.0282609
## 
## $sigma_hat
## , , 1
## 
##            [,1]        [,2]       [,3]        [,4]
## [1,] 0.11911420 0.095675206 0.01560139 0.010212766
## [2,] 0.09567521 0.140781589 0.01103343 0.009214069
## [3,] 0.01560139 0.011033435 0.02996960 0.005983500
## [4,] 0.01021277 0.009214069 0.00598350 0.010968302
## 
## , , 2
## 
##            [,1]       [,2]       [,3]       [,4]
## [1,] 0.24932696 0.07998263 0.15993053 0.05184108
## [2,] 0.07998263 0.09661311 0.07563613 0.04003040
## [3,] 0.15993053 0.07563613 0.18703430 0.06706036
## [4,] 0.05184108 0.04003040 0.06706036 0.03800261
## 
## , , 3
## 
##            [,1]       [,2]       [,3]       [,4]
## [1,] 0.38882431 0.09476486 0.29089175 0.04723602
## [2,] 0.09476486 0.09568767 0.06779059 0.04416149
## [3,] 0.29089175 0.06779059 0.29019077 0.04591393
## [4,] 0.04723602 0.04416149 0.04591393 0.07251553
```

```r
predictor
```

```
## $class
##  [1] "setosa"     "setosa"     "setosa"     "versicolor" "versicolor"
##  [6] "versicolor" "virginica"  "virginica"  "virginica"  "virginica" 
## 
## $posterior
##                [,1]         [,2]         [,3]
##  [1,]  1.000000e+00 1.813354e-22 2.280380e-35
##  [2,]  1.000000e+00 8.988426e-31 4.462094e-45
##  [3,]  1.000000e+00 4.777289e-25 8.304429e-40
##  [4,]  3.032833e-71 9.999596e-01 4.035974e-05
##  [5,]  6.615329e-27 9.999990e-01 1.004706e-06
##  [6,]  2.719422e-63 9.998484e-01 1.515597e-04
##  [7,] 5.580634e-125 2.335739e-04 9.997664e-01
##  [8,] 1.561695e-134 1.206501e-03 9.987935e-01
##  [9,] 2.680194e-156 1.680356e-06 9.999983e-01
## [10,] 2.359823e-119 6.943433e-02 9.305657e-01
```



## Confusion Matrix


```r
tab1 <- table(Predicted = p2, Actual = test_set[, 5])
tab1
```

```
##             Actual
## Predicted    setosa versicolor virginica
##   setosa         24          0         0
##   versicolor      0         17         1
##   virginica       0          0        18
```

```r
sum(diag(tab1))/sum(tab1) * 100
```

```
## [1] 98.33333
```


```r
tab1 <- table(Predicted = p3, Actual = test_set[, 5])
tab1
```

```
##             Actual
## Predicted    setosa versicolor virginica
##   setosa         24          0         0
##   versicolor      0         17         2
##   virginica       0          0        17
```

```r
sum(diag(tab1))/sum(tab1) * 100
```

```
## [1] 96.66667
```
