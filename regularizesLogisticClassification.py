import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.3,random_state=0)
sc=StandardScaler()
sc.fit(xTrain)
xTrainStd=sc.transform(xTrain)
xTestStd=sc.transform(xTest)

weight,params=[],[]
for c in np.arange(0,5):
    lr=LogisticRegression(C=10**c,random_state=0)
    lr.fit(xTrainStd,yTrain)
    weight.append(lr.coef_[1])
    params.append(10**c)
weight=np.array(weight)

plt.plot(params,weight[:,0],label='petal length')
plt.plot(params,weight[:,1],label='petal width',linestyle='--')
plt.ylabel('weight')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.show()