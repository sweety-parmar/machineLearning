from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(x,y,classifier,text_idx=None,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','green','blue','cyan','gray')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min,x1_max=x[:,0].min()-1,x[:,0].max()+1
    x2_min,x2_max=x[:,1].min()-1,x[:,1].max()+1
    xx1 , xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    print(xx1, xx2)
    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    print(z)
    z=z.reshape(xx1.shape)
    plt.contour(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot all the samples

    x_test,y_test=x[text_idx,:],y[text_idx]
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)


    if text_idx:
        x_test,y_test=x[text_idx,:],y[text_idx]
        plt.scatter(x_test[:,0],x_test[:,1],c='',alpha=1.0,linewidth=1,marker='o',s=55,label='test set')



iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target

xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.3,random_state=0)

sc=StandardScaler()
sc.fit(xTrain)
xTrainStd=sc.transform(xTrain)
xTestStd=sc.transform(xTest)
ppn= Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(xTrainStd,yTrain)
y_pred=ppn.predict(xTestStd)
# print('Misclassified samples: %d' % (yTest != y_pred).sum())
#
# print('accuracy of the model : %f' %accuracy_score(yTest,y_pred))


x_combined_std=np.vstack((xTrainStd,xTestStd))
y_combined=np.hstack((yTrain,yTest))

plot_decision_regions(x_combined_std,y_combined,classifier=ppn,text_idx=range(105,150))
plt.xlabel('petal length [standardised]')
plt.ylabel('petal width [standardised]')
plt.legend('upper left')
plt.show()





