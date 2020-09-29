from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso
dia=load_diabetes()
x=dia.data
y=dia.target
reg=LinearRegression()
Lasso_reg=Lasso(alpha=10)
reg.fit(x,y)
Lasso_reg.fit(x,y)
for a, b in zip(dia.feature_names, reg.coef_):
	print(a,b)
for a, b in zip(dia.feature_names, Lasso_reg.coef_):
	print(a,b)
print(reg.intercept_)
import matplotlib.pyplot as plt
x=[i[0] for i in x]
plt.scatter(x,y)

plt.show()

