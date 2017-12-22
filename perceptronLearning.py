import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

class perceptron(object):

    """ PERCEPTRON CLASSIFIER

    ---------PARAMETERS----------
     eta : float ( learning rate between 0.0 and 0.1 )

     int_iter :int ( number of iterations )

    --------WEIGHTS-------------
    w_:1d array (  weights after fitting )

    errors_:list (  number of misclassification )

     """
    def __init__(self,eta=0.01,int_iter=10):
        self.eta=eta
        self.int_iter=int_iter

    def fit(self,x,y):

        """ FIT TRAINING DATA

        --------PARAMETERS----------
        x:{array-like},shape =[n_samples,n_features] (training vector ,
                                                      n_samples ,no. of samples
                                                      n_features , no. of features )

        y:{array-like},shape=[n_samples](target values)

        -------RETURNS--------------
        self:object """

        self.w_=np.zeros(1+x.shape[1])
        self.errors_=[]


        for _ in range(self.int_iter):
            errors = 0
            for xi ,target in zip(x,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=xi*update
                self.w_[0]+=update
                errors+=int(update != 0.0)
            self.errors_.append(errors)
            print(self.errors_)
        return self

    def net_input(self,x):
        """ calculate net input """
        return np.dot(x,self.w_[1:])+self.w_[0]

    def predict(self,x):
        """ return class label after unit step """
        return np.where(self.net_input(x)>=0.0,1,-1)


# iris=datasets.load_iris()
# x=iris.data[:100,:2]
# y=iris.target
# y=np.where(y==0,-1,1)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values


plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='sentosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal lenght')
plt.legend(loc='upper left')
plt.show()

ppn=perceptron(eta=0.01,int_iter=10)
ppn.fit(x,y)


plt.plot(range(1, len(ppn.errors_) + 1),ppn.errors_,marker='_')
plt.xlabel('epochs')
plt.ylabel('number of classification')
plt.show()

