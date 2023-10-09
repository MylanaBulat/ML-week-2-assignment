import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.pyplot as pl2
#import matplotlib.pyplot as pl3
from sklearn.linear_model import LogisticRegression

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
pl.rc('font' , size =18)
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

# show wich feature has more influence on the prediction 
# which feature causes the prediction to increase and to dicrease
# pl2.bar([0,1], model.coef_[0])
# pl2.xlabel("features")
# pl2.ylabel("feature importance")
# pl2.show()

# parametr values of a trained model: feta0 - intercept; feta1, feta2 - coef
coef = model.coef_[0]
intercept = model.intercept_[0]
print("Coeficients: ", coef)
print("Intercept: ", intercept)
#[-0.10780439  3.69806133]

# iii)
# predict the target values in the trained data
ypred = model.predict(x)

# add predictions to the original plot using a different marker and color
# plot training data 
pl.scatter(X1[y==1], X2[y==1], color = 'blue', marker = '+')
pl.scatter(X1[y==-1], X2[y==-1], color = 'orange', marker = 'o')
# plot prediction
pl.scatter(X1[ypred==1],  X2[ypred==1], color = 'green', marker = '^')
pl.scatter(X1[ypred==-1],  X2[ypred==-1], color = 'red', marker = "*")

pl.xlabel("X1")
pl.ylabel("X2")
pl.legend(["+1", "-1", "predicted +1", "predicted -1"])



#show a decision boundry ???
# x_plot = (-1, 1)
# y_plot = tuple(((coef[0][0]*x) + intercept)/coef[0][1] for x in x_plot)
# pl3.plot(list(x_plot), list(y_plot))

# show a decision boundry
coef1, coef2 = model.coef_.T
slope = -coef1 /coef2
c = -intercept/coef2
x_plot = np.array(pl.gca().get_xlim())
y_plot = c + slope * x_plot
pl.plot(x_plot, y_plot, color = "purple")

pl.show()








# Part 2
# i)
from sklearn.svm import LinearSVC
# train the linear SVM classifier on my data with penalty parametr C = 1
modelSVC = LinearSVC(C=1.0).fit(x,y)
# modelSVC = LinearSVC(C=100).fit(x,y)
# modelSVC = LinearSVC(C=0.001).fit(x,y)
print("intercept%f, slope%f,%f"%(model.intercept_, model.coef_[0][0], model.coef_[0][1]))

# parametr values of each trained model?
coefSVC = modelSVC.coef_[0]
interceptSVC = modelSVC.intercept_[0]
print("Coeficients for SVC training: ", coefSVC)
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
pl.legend(["+1", "-1", "predicted SVC +1", "predicted SVC -1"])

# plot decision boundry 
coefSVC1, coefSVC2 = modelSVC.coef_.T
ar = np.array((-1, 1))
pl.plot(ar, ((-coefSVC1/coefSVC2) * ar) + (-interceptSVC/coefSVC2), color = "black")

pl.show()

#[-0.10780439  3.69806133]
# Coeficients for SVC training:  [-0.05581508  1.3546743 ]
# Intercept for SVC training:  -0.773779349645351