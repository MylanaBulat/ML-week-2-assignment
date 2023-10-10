import numpy as np
import pandas as pd
import statistics 
import matplotlib.pyplot as pl
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import LinearSVC

# Part a)
# i)
# read file and load values into X1, X2 and x, y 
df = pd.read_csv("week2.csv")
print(df.head())
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
x = np.column_stack((X1, X2))
y = df.iloc[:,2]

# plot the data 
pl.rc('font' , size = 14)
pl.rcParams['figure.constrained_layout.use'] = True
pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
pl.xlabel("X1")
pl.ylabel("X2")
pl.legend(["+ 1", "-1"])
pl.show()

# ii)
# train a logistic regression classifier on the model with our x parametrs and y value(-1, or +1)
model = LogisticRegression()
model.fit(x, y )

# parametr values of a trained model: theta0 - intercept; theta1 - coef1, theta2 - coef2
coef = model.coef_[0]
intercept = model.intercept_[0]
print("Coefficients: ", coef)
print("Intercept: ", intercept)

# iii)
# predict the target values in the trained data
ypred = model.predict(x)

# add predictions to the original plot using a different marker and color
# plot training data 
pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
# plot prediction
pl.scatter(X1[ypred==1],  X2[ypred==1], color = 'green', marker = 'x')
pl.scatter(X1[ypred==-1],  X2[ypred==-1], color = 'purple', marker = "*")

pl.xlabel("X1")
pl.ylabel("X2")

# show a decision boundry
# get each coeficient
coef1, coef2 = model.coef_.T
# values 1 and -1 for X1
x_plot = np.array((-1, 1))
# appropriate values for X2
y_plot = ((-coef1/coef2) * x_plot) + (-intercept/coef2)
pl.plot(x_plot, y_plot, color = "black")

pl.legend(["+1", "-1", "predicted +1", "predicted -1", "decision boundary"])
pl.show()

# Part b
# i)
# train the linear SVM classifier on my data with penalty parametr C 
#modelSVC = LinearSVC(C=0.001).fit(x,y)
#modelSVC = LinearSVC(C=1.0).fit(x,y)
modelSVC = LinearSVC(C=100).fit(x,y)
print("intercept%f, slope%f,%f"%(model.intercept_, model.coef_[0][0], model.coef_[0][1]))

# parametr values of each trained model
coefSVC = modelSVC.coef_[0]
interceptSVC = modelSVC.intercept_[0]
print("Coefficients for SVC training: ", coefSVC)
print("Intercept for SVC training: ", interceptSVC)

# ii)
# make predictions
ySVCpred = modelSVC.predict(x)

# plot training data 
pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
# plot prediction
pl.scatter(X1[ySVCpred==1],  X2[ySVCpred==1], color = 'green', marker = 'x')
pl.scatter(X1[ySVCpred==-1],  X2[ySVCpred==-1], color = 'purple', marker = "*")

pl.xlabel("X1")
pl.ylabel("X2")

# plot decision boundry 
coefSVC1, coefSVC2 = modelSVC.coef_.T
ar = np.array((-1, 1))
pl.plot(ar, ((-coefSVC1/coefSVC2) * ar) + (-interceptSVC/coefSVC2), color = "black")
pl.legend(["+1", "-1", "predicted SVC +1", "predicted SVC -1", " decision boundary"])
pl.show()

# Part c
# i)
X1 = df.iloc[:,0]
X2 = df.iloc[:,1]
X1_sq = np.square(X1)
X2_sq = np.square(X2)
x = np.column_stack((X1, X2, X1_sq, X2_sq))
y = df.iloc[:,2]

# ii)
# train a logistic regression classifier on the model with our x parametrs and y value(-1, or +1)
sq_model = LogisticRegression()
sq_model.fit(x, y)

# iii)
# predict the target values in the trained data
ypred = sq_model.predict(x)

# add predictions to the original plot using a different marker and color
# plot training data 
pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
# plot prediction
pl.scatter(X1[ypred==1],  X2[ypred==1], color = 'green', marker = 'x')
pl.scatter(X1[ypred==-1],  X2[ypred==-1], color = 'purple', marker = "*")
pl.xlabel("X1")
pl.ylabel("X2")
pl.legend(["+1", "-1", "predicted +1", "predicted -1"])
pl.show()


# parameter values of a trained model
coef = sq_model.coef_[0]
intercept = sq_model.intercept_[0]
print("Coefficients: ", coef)
print("Intercept: ", intercept)
X1_sorted = np.sort(X1)
X2_sorted = np.sort(X2)
X1_sq = np.square(X1_sorted)
X2_sq = np.square(X2_sorted)

# iii)

#predict something that occurs the most often in the original 
base = np.sign(statistics.mean(y))
#y predictions for the base model
#create an array according to the patern (patern, what do we fill it with)
ypred_base = np.full(len(y), base)
#compare to the actual y
base_acc = metrics.accuracy_score(y, ypred_base)
#metrics classification report gives me 
#precision recall f1 score
#precision is - out of everything we predicted to be positive, how many are actualy positive 
#recall out of everything that is actualy positive how many did i predict correctly 
#f1 score is a messure of inpact of false positives and false negatives on the model
print(metrics.classification_report(y, ypred_base))
print("base model accuracy score: ", base_acc)

pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
# plot prediction
pl.scatter(X1[ypred_base==1],  X2[ypred_base==1], color = 'green', marker = 'x')
pl.scatter(X1[ypred_base==-1],  X2[ypred_base==-1], color = 'purple', marker = "*")
pl.xlabel("X1")
pl.ylabel("X2")
pl.title("Comparison of Logistic Regression to the base model")
pl.legend([ "+1", "-1", "predicted +1", "base model predicted -1"])
pl.show()


# iv)

# θ0 + θ1x1 + θ2x2 + θ3x1^2 + θ4x2^2 = 0
# θ2x2 = -(θ0 + θ1x1 + θ3x1^2 + θ4x2^2)
# x2 = -(θ0 + θ1x1 + θ3x1^2 + θ4x2^2) / θ2
# x2 = -θ0/θ2 - θ1/θ2 * x1 - θ3/θ2 * x1^2 - θ4/θ2 * x2^2

# plot decision boundary 
boundary = (intercept + (coef[0] * X1_sorted) + (coef[2] * X1_sq) + (coef[3] * X2_sq)) / (-coef[1])
print(boundary)
pl.plot(X1_sorted[boundary <= 1], boundary[boundary <= 1], linewidth=2, color='black')
pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
# plot prediction
pl.scatter(X1[ypred==1],  X2[ypred==1], color = 'green', marker = 'x')
pl.scatter(X1[ypred==-1],  X2[ypred==-1], color = 'purple', marker = "*")
pl.xlabel("X1")
pl.ylabel("X2")
pl.legend(["decision boundary", "+1", "-1", "predicted +1", "predicted -1"])
pl.show()