from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt

boston=load_boston()
print(boston)
X=boston.data
y=boston.target

reg=LinearRegression()
lasso_reg=Lasso(alpha=0.0000001)
model=reg.fit(X,y)
lasso_reg.fit(X,y)
x=[i[12] for i in X]
	


plt.scatter(x,y)

plt.show()
