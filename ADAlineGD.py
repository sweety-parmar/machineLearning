import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

class AdalineGD(object):

    def __init__(self,alpha=0.01,iter=10):
        self.alpha=alpha
        self.iter=iter

    def fit(self,x,y):

        self.weights=np.zeros(1+x.shape[1])
        self.cost=[]

        for i in range(self.iter):
            output = self.predict(x)
            errors=(y-output)
            self.weights[1:]+=self.alpha*x.T.dot(errors)
            self.weights[0]+=self.alpha*errors.sum()
            cost_=(errors**2).sum()/2
            self.cost.append(cost_)
        return self

    def predict(self,x):
        return np.where(self.netInput(x)>=0.0,1,-1)

    def netInput(self,x):
        return np.dot(x,self.weights[1:])+self.weights[0]


# iris=datasets.load_iris()
# x=iris.data[:100,:2]
# y=iris.target
# y=np.where(y==0,-1,1)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(alpha=0.01,iter=10).fit(x, y)
ax[0].plot(range(1, len(ada1.cost) + 1),np.log10(ada1.cost), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = AdalineGD(alpha=0.0001,iter=10).fit(x, y)
ax[1].plot(range(1, len(ada2.cost) + 1),ada2.cost, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()



