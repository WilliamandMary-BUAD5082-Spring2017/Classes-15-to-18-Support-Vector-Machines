if( ! require("e1071") ){ install.packages("e1071") }
library(e1071)
if( ! require("MASS") ){ install.packages("MASS") }
library(MASS)
library(ISLR)
rm(list=ls())
##########################################################################
# A Simple Example
#
# Read data from the file named data.txt and plot the observations,  
# colored according to their class labels.
# Then construct a Support Vector Classification model with various values  
# of C (and default gamma). Compute the error rate for the bestr model and  
# plot the classification again colored according to their class labels.
# 
# Finally, construct an SVM for various values of C and gamma, compute
# the error rate for the best model and plot the classification as above.
##########################################################################
set.seed(5082)
data<-read.table("data.txt",header=T,sep="\t")
data$y<-as.factor(data$y)
plot( data$x1, data$x2,  col=(as.numeric(data$y)+1),pch=19, cex=1.05, xlab='x1', ylab='x2', main='initial data' )
svc.fit.tune.out = tune( svm, y ~ ., data=data, kernel="linear", ranges=list( cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000) ) )
print( summary(svc.fit.tune.out) ) # <- use this output to select the optimal cost value

# The best model is stored in "svc.fit.tune.out$best.model" 
# 
print(svc.fit.tune.out$best.model)
y_hat = predict(svc.fit.tune.out$best.model, newdata=data.frame(x1=data$x1,x2=data$x2) )
y_hat = as.numeric( as.character( y_hat ) ) # convert factor responses into numerical values 
print( paste('Linear SVM training error rate = ',  1 - sum( y_hat == data$y )/length(data$y) ) )
plot( data$x1, data$x2, col=(y_hat+2), pch=19, cex=1.05, xlab='x1', ylab='x2',main='Support Vector Classification' )


svm.fit.tune.out = tune( svm, y ~ ., data=data, kernel="radial",  
                         ranges=list( cost=c(0.001, 0.01, 0.1, 1, 5, 10),
                                      gamma=c(0.5,1,2,3,4,5) ) )
print( summary(svm.fit.tune.out) ) # <- use this output to select the optimal cost value

# The best model is stored in "svm.fit.tune.out$best.model" 
# 
print(svm.fit.tune.out$best.model)
y_hat = predict(svm.fit.tune.out$best.model, newdata=data.frame(x1=data$x1,x2=data$x2) )
y_hat = as.numeric( as.character( y_hat ) ) # convert factor responses into numerical values 
print( paste('Linear SVM training error rate = ',  1 - sum( y_hat == data$y )/length(data$y) ) )
plot( data$x1, data$x2, col=(y_hat+2), pch=19, cex=1.05, xlab='x1', ylab='x2',main='Support Vector Machine/Radial Kernel' )

##########################################################################
# Question 5 (pp.369)
# We have seen that we can fit an SVM with a non-linear kernel in order
# to perform classification using a non-linear decision boundary.We will
# now see that we can also obtain a non-linear decision boundary by
# performing logistic regression using non-linear transformations of the
# features.
##########################################################################
set.seed(5082)
# Part (a):
# Generate a data set with n = 500 and p = 2, such that the observations
# belong to two classes with a quadratic decision boundary
# between them.
n = 1000
p = 2 
x1 = runif(n) - 0.5
x2 = runif(n) - 0.5
y = 1*( x1^2 - x2^2 > 0 )
# or equivalently
#y = 1*( abs(x1) - abs(x2) > 0 )   
DF = data.frame( x1=x1, x2=x2, y=as.factor(y) )
# Part (b):
# Plot the observations, colored according to their class labels.
plot( x1, x2, col=(y+1), pch=19, cex=1.05, xlab='x1', ylab='x2', main='initial data' )

# Part (c):  
# Fit a logistic regression model to the data, using X1 and X2 as
# predictors.
glm.linear.fit = glm( y ~ x1 + x2, data=DF, family=binomial )

# Part (d): 
# Apply this model to the training data in order to obtain a predicted
# class label for each training observation. Plot the observations,
# colored according to the predicted class labels. The
# decision boundary should be linear.
#
# Recall that when type="response" in predict we output probabilities:
# 
y_hat = predict( glm.linear.fit, newdata=data.frame(x1=x1,x2=x2), type="response" )
predicted_class = 1 * ( y_hat > 0.5 ) 
print( paste('Linear logistic regression training error rate = ',  1 - sum( predicted_class == y )/length(y) ) )

plot( x1, x2, col=(predicted_class+1), pch=19, cex=1.05, xlab='x1', ylab='x2', main='logistic regression: y ~ x1 + x2' )

# Part (e): 
# Now fit a logistic regression model to the data using non-linear
# functions of X1 and X2 as predictors (e.g. X1^2, X1*X2, log(X2), etc.)
glm.nonlinear.fit = glm( y ~ x1 + x2 + I(x1^2) + I(x2^2) + I(x1*x2), data=DF, family="binomial" )
# Part (f):
# Apply this model to the training data in order to obtain a predicted
# class label for each training observation. Plot the observations,
# colored according to the predicted class labels. The
# decision boundary should be obviously non-linear. If it is not,
# then repeat (a)-(e) until you come up with an example in which
# the predicted class labels are obviously non-linear.
y_hat = predict( glm.nonlinear.fit, newdata=data.frame(x1=x1,x2=x2), type="response" )
predicted_class = 1 * ( y_hat > 0.5 ) 
print(paste('Non-linear logistic regression training error rate =',  1 - sum( predicted_class == y )/length(y) ) )

plot( x1, x2, col=(predicted_class+1), pch=19, cex=1.05, xlab='x1', ylab='x2' )

# Part (g): 
# Fit a support vector classifier to the data with X1 and X2 as
# predictors. Obtain a class prediction for each training observation.
# Plot the observations, colored according to the predicted
# class labels.
dat = data.frame( x1=x1, x2=x2, y=as.factor(y) )

# Do CV to select the value of cost using the "tune" function:
#
svc.fit.tune.out = tune( svm, y ~ ., data=dat, kernel="linear", ranges=list( cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000) ) )
print( summary(svc.fit.tune.out) ) # <- use this output to select the optimal cost value

# The best model is stored in "svc.fit.tune.out$best.model" 
# 
print(svc.fit.tune.out$best.model)

y_hat = predict(svc.fit.tune.out$best.model, newdata=data.frame(x1=x1,x2=x2) )
y_hat = as.numeric( as.character( y_hat ) ) # convert factor responses into numerical values 
print( paste('Linear SVM training error rate = ',  1 - sum( y_hat == y )/length(y) ) )

plot( x1, x2, col=(y_hat+1), pch=19, cex=1.05, xlab='x1', ylab='x2' )

# Part (h): 
# Fit a SVM using a non-linear kernel to the data. Obtain a class
# prediction for each training observation. Plot the observations,
# colored according to the predicted class labels.
# Do CV to select the value of cost using the "tune" function:
#
svm.fit.tune.out = tune( svm, y ~ ., data=dat, kernel="radial", ranges = 
                   list( cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000),
                         gamma=c(0.5,1,2,3,4) ) )
print( summary(svm.fit.tune.out) ) # <- use this output to select the optimal cost value

# The best model is stored in "svm.fit.tune.out$best.model" 
# 
print(svm.fit.tune.out$best.model)

y_hat = predict(svm.fit.tune.out$best.model, newdata=data.frame(x1=x1,x2=x2) )
y_hat = as.numeric( as.character( y_hat ) ) # convert factor responses into numerical values 
print( paste('Nonlinear SVM training error rate = ',  1 - sum( y_hat == y )/length(y) ) )

plot( x1, x2, col=(y_hat+1), pch=19, cex=1.05, xlab='x1', ylab='x2' )
# Part (i):
# Comment on your results
# In the first part of this problem we use a support vector machine with a linear kernel. We know 
# that this will produce a linear classification region and will therefore not be optimal given the 
# training data. To determine the numerical value of the cost parameter in the svm function call we 
# use the tune function. When we do that and then use the resulting classifier to predict the training 
# data we get the plot for the linear SVM that is predicting all of the samples to be of the same class
# and thus we expect it to have an error rate equal to that of the class with the smaller a-priori 
# probability. 
# Following this we fit a non-linear SVM using the “radial” kernel and also use the tune function to specify 
# both the value of cost and gamma. When we do this and then use the resulting classifier to predict the training 
# data we get the final plot, which comes much closer to the true decision boundary.
#
# In summary, We see that both techniques that result in linear classification boundaries give similar training 
# error rate. The two techniques that result in non-linear boundaries also give similar error rates. One can also 
# see visually (using the plots above) that the two non-linear techniques are estimating the correct decision 
# boundary while the linear techniques are not.
#
##########################################################################
# Question 8 (pp.371-372)
# This problem involves the OJ data set which is part of the ISLR
# package.
##########################################################################
set.seed(5082)
# Part (a): Create train and test sets
#
n = dim(OJ)[1]
n_train = 800
train_inds = sample(1:n,n_train)
test_inds = (1:n)[-train_inds]
n_test = length(test_inds)

# Part (b) Use a linear kernel with cost=0.01 to start with:
#
svm.fit = svm( Purchase ~ ., data=OJ, kernel="linear", cost=0.01 )
print( summary( svm.fit ) )

# Part (c) Use this specific SVM to estimate training/testing error rates:
#
y_hat = predict( svm.fit, newdata=OJ[train_inds,] )
print( table( truth=OJ[train_inds,]$Purchase ,predicted=y_hat) )
print( paste('Linear SVM training error rate (cost=0.01) = ',  1 - sum( y_hat == OJ[train_inds,]$Purchase ) / n_train ) )

y_hat = predict( svm.fit, newdata=OJ[test_inds,] )
print( table( truth=OJ[test_inds,]$Purchase ,predicted=y_hat) )
print( paste('Linear SVM testing error rate (cost=0.01) = ',  1 - sum( y_hat == OJ[test_inds,]$Purchase ) / n_test ) )

# Part (d): Use tune to select an optimal value for cost when we have a linear kernel:
#
svm.linear.tune.out = tune( svm, Purchase ~ ., data=OJ, kernel="linear", ranges=list( cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000) ) )
print( summary(svm.linear.tune.out) ) # <- use this output to select the optimal cost value

# Part (e): Predict the performance on training and testing using the best linear model: 
# 
y_hat = predict( svm.linear.tune.out$best.model, newdata=OJ[train_inds,] )
print( table( truth=OJ[train_inds,]$Purchase ,predicted=y_hat) )
print( paste('Linear SVM training error rate (optimal cost=1) = ',  1 - sum( y_hat == OJ[train_inds,]$Purchase ) / n_train ) )

y_hat = predict( svm.linear.tune.out$best.model, newdata=OJ[test_inds,] )
print( table(truth=OJ[test_inds,]$Purchase , predicted=y_hat) )
print( paste('Linear SVM testing error rate (optimal cost=1) = ',  1 - sum( y_hat == OJ[test_inds,]$Purchase ) / n_test) )

# Part (f): Use a radial kernel: 
# 
svm.nonlinear.tune.out = tune( svm, Purchase ~ ., data=OJ, kernel="radial", ranges = list( cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000), gamma=c(0.5,1,2,3,4) ) )
print( summary(svm.nonlinear.tune.out) ) # <- use this output to select the optimal cost value

y_hat = predict( svm.nonlinear.tune.out$best.model, newdata=OJ[train_inds,] )
print( table( truth=OJ[train_inds,]$Purchase,predicted=y_hat ) )
print( paste('Radial SVM training error rate (optimal) = ',  1 - sum( y_hat == OJ[train_inds,]$Purchase ) / n_train ) )

y_hat = predict( svm.nonlinear.tune.out$best.model, newdata=OJ[test_inds,] )
print( table( truth=OJ[test_inds,]$Purchase ,predicted=y_hat) )
print( paste('Radial SVM testing error rate (optimal) = ',  1 - sum( y_hat == OJ[test_inds,]$Purchase ) / n_test) )

# Part (g): Use a polynomial kernel: try degrees 1 to 3 with various costs
# 
svm.poly.tune.out = tune( svm, Purchase ~ ., data=OJ, kernel="polynomial", ranges = list( cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000), degree=c(1,2,3) ) )
print( summary(svm.poly.tune.out) ) # <- use this output to select the optimal cost value

y_hat = predict( svm.poly.tune.out$best.model, newdata=OJ[train_inds,] )
print( table(truth=OJ[train_inds,]$Purchase, predicted=y_hat ) )
print( paste('Polynomial SVM training error rate (optimal) = ',  1 - sum( y_hat == OJ[train_inds,]$Purchase ) / n_train ) )

y_hat = predict( svm.poly.tune.out$best.model, newdata=OJ[test_inds,] )
print( table(truth=OJ[test_inds,]$Purchase , predicted=y_hat) )
print( paste('Polynomial SVM testing error rate (optimal) = ',  1 - sum( y_hat == OJ[test_inds,]$Purchase ) / n_test) )
# Part (h): From the above test error rates it looks like the radial kernel gives the smallest testing error rate.
