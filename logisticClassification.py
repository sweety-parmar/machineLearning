import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

def plot_decision_regions(x,y,classifier,text_idx,resolution=0.02):
    markers=('s','^','x','o','v')
    colors=('red','green','blue','cyan','gray')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=x[:,0].min(),x[:,0].max()
    x2_min,x2_max=x[:,1].min(),x[:,1].max()
    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,z,alpha=0.2,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    if text_idx:
        x_test,y_test=x[text_idx,:],y[text_idx]
        plt.scatter(x_test[:,0],x_test[:,1],c='',alpha=1.0,linewidth=1,marker='o',s=55,label='test set')



iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target
xTrain,xTest,yTrain,yTest=train_test_split(x,y,random_state=0,test_size=0.3)
sc=StandardScaler()
sc.fit(xTrain)
xTrainStd=sc.transform(xTrain)
xTestStd=sc.transform(xTest)

x_combined_std=np.vstack((xTrainStd,xTestStd))
y_combined=np.hstack((yTrain,yTest))

lr=LogisticRegression(C=1000.0,random_state=0)
lr.fit(xTrainStd,yTrain)
lr.predict_proba(xTestStd[0,:])
# print(' probabilty of test cases %f ' % lr.predict_proba(xTestStd[0,:]))



plot_decision_regions(x_combined_std,y_combined,classifier=lr,text_idx=range(105,150))
plt.xlabel('petal length [standardised]')
plt.ylabel('petal width [standardised]')
plt.legend(loc='upper left')
plt.show()
