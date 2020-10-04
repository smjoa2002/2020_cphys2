import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
iris=datasets.load_iris()

print(iris.target_names)
iris.feature_names
X=[[i] for i in iris.data[:,2]]
y=[1 if i==2 else 0 for i in iris.target]
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(solver='lbfgs')
log_reg.fit(X,y)
print(log_reg.score(X,y))
x=[[i] for i in np.linspace(4,8,100)]
plt.plot(X,y,'o')
plt.plot(x,log_reg.predict(x))


plt.show()
"꽃잎과 꽃받침의 길이와 넓이를 분석해보았다. 꽃받침은 넓이가 2-4cm사이로 다양하게 분포되어 있었고, 길이는 6.3 이상이 되는 경우가 일반적이었다. 꽃잎은 넓이가 1.5cm이상정도였고, 길이가 4.5cm이상으로 분포되어 있었다."   
