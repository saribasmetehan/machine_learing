import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.cross_decomposition import  PLSRegression, PLSSVD
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV




"""
ad = pd.read_csv("Advertising.csv")

df_ad = ad.copy()
df_ad = df_ad.iloc[:,1:len(df_ad)]

print(df_ad.head())
print(df_ad.info())
print(df_ad.describe().T)
print(df_ad.isnull().sum())
print(df_ad.corr())

#sns.pairplot(df_ad,kind="reg")
#plt.show()

#sns.jointplot(x = "TV",y="sales",data=df_ad,kind="reg")
#plt.show()

"""
"""
#method 1

x = df_ad[["TV"]]
x = sm.add_constant(x)

y = df_ad["sales"]

lm = sm.OLS(y,x)

model = lm.fit()

print(model.summary())

print(model.params)

print(model.summary().tables[1])
print(model.summary().tables[0])
print(model.conf_int())
print(model.f_pvalue)


sales_prediction = model.predict([1,30])

print(sales_prediction)

new_sales_list = [[1,10],[1,30],[1,100]]

sales_prediction_new = model.predict(new_sales_list)

print(sales_prediction_new)
"""
"""
#method 2

lm = smf.ols("sales ~ TV",df_ad)

model = lm.fit()

print(model.summary())

print(model.params)

print(model.summary().tables[1])

print(model.conf_int())

"""
"""
#method 3

x = df_ad[["TV"]]
y = df_ad["sales"]

reg = LinearRegression()
model = reg.fit(x,y)

print(model.intercept_)


lm = smf.ols("sales ~ TV",df_ad)

model = lm.fit()

mse = mean_squared_error(y,model.fittedvalues)

print(mse)

rmse = np.sqrt(mse)

print(rmse)

print(reg.predict(x)[:10])
print(y[:10])

pd_guess_real = pd.DataFrame({"real" : y[:10] , "guess" : reg.predict(x)[:10] })

print(pd_guess_real)

print(model.resid[0:10])

plt.plot(model.resid)
plt.show()

"""
"""
#multiple linear regression

pd_ad = pd.read_csv("Advertising.csv",usecols=[1,2,3,4])

df = pd_ad.copy()

x = df.drop("sales",axis=1)
y = df["sales"]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=44)

lm = sm.OLS(y_train,x_train)

model = lm.fit()

print(model.summary())


lm = LinearRegression()
model = lm.fit(x_train,y_train)

print(model.intercept_)
print(model.coef_)

new_list = [[30],[10],[40]]

new_list = pd.DataFrame(new_list).T

print(model.predict(new_list))

rmse = np.sqrt(mean_squared_error(y_train,model.predict(x_train)))

print(rmse)

print(cross_val_score(model,x,y,cv=10,scoring="r2").mean())

print(cross_val_score(model,x_train,y_train,cv=10,scoring="r2").mean())
print(np.sqrt(-cross_val_score(model,x_train,y_train,cv=10,scoring="neg_mean_squared_error")).mean())

"""

"""

#PCR

hit = pd.read_csv("Hitters.csv")

df = hit.copy()

df = df.dropna()

print(df.info())
print(df.head())
print(df.describe().T)

dms = pd.get_dummies(df[["League","Division","NewLeague"]]).astype(int)

print(dms.head())

y = df["Salary"]

x_ = df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")

x = pd.concat([x_, dms[["League_N","Division_W","NewLeague_N"]]],axis=1)

print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=44)

training = df.copy()

x_reduced_train = PCA.fit_transform(scale(x_train))

print(np.cumsum(np.round(pca.explained_variance_ratio_,de,4)*100)[0:10])

pcr_model = lm.fit(x_reduced_train,y_train)

y_pred = pcr_model.predic(x_reduced_train)

lm = LinearRegression()
pcr_model = lm.fit(x_reduced_train[:,0:2],y_train)
y_pred = pcr_model.predict(x_reduced_train[:,0:2])

print(np.sqrt(mean_squared_error(y_test,y_pred)))


"""
"""

#PLS

hit = pd.read_csv("Hitters.csv")
df = hit.copy()
df = df.dropna()
ms =pd.get_dummies(df[["League","Division","NewLeague"]])

print(ms.head())

y = df["Salary"]

x_ = df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x = pd.concat([x_,ms[["League_N","Division_W","NewLeague_N"]]],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

pls_model =PLSRegression().fit(x_train,y_train)

print(pls_model.coef_)

pls_model =PLSRegression(n_components=2).fit(x_train,y_train)

print(pls_model.coef_)

print(pls_model)

print(pls_model.predict(x_train))

y_pred = pls_model.predict(x_train)

print(np.sqrt(mean_squared_error(y_train,y_pred)))

print(r2_score(y_train,y_pred))

cv_10 = model_selection.KFold(n_splits=10,shuffle=True,random_state=1)

RMSE = []

for i in np.arange(1,x_train.shape[1]+1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls,x_train,y_train,cv=cv_10,scoring="neg_mean_squared_error"))
    RMSE.append(score)

#plt.plot(np.arange((1,x_train.shape[1]+1),np.array(RMSE),"-v",c= "r"))
#plt.show()


#ridge

ridge_model = Ridge(alpha=0.1).fit(x_train,y_train)

print(ridge_model.coef_)

lambdas = 10**np.linspace(10,-2,100)*0.5

ridge_model = Ridge()

list_ = []

for i in lambdas:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(x_train,y_train)
    list_.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdas,list_)
ax.set_xscale("log")

#plt.show()

y_pred = ridge_model.predict(x_test)
print(np.sqrt(mean_squared_error(y_train,y_pred)))

ridge_cv = RidgeCV(alphas=lambdas,scoring="neg_mean_squared_error",normalize = True)

ridge_cv.fit(x_train,y_train)

ridge_tuned = Ridge(alpha=ridge_cv.alpha_,normalize= True).fit(x_train,y_train)

#Lasso
lasso = Lasso()
lasso_model = Lasso(alpha=0.1).fit(x_train,y_train)
lambdas = 10**np.linspace(10,-2,100)*0.5
list_ = []


for i in lambdas:
    lasso_model.set_params(alpha = i)
    lasso_model.fit(x_train,y_train)
    list_.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdas,list_)
ax.set_xscale("log")

lasso_model.predict(x_test)

lasso_cv_model = LassoCV(alphas=None,cv =10,max_iter =1000,normalize=True)

lasso_cv_model.fit(x_train,y_train)

#eNet

enet_model = ElasticNet().fit(x_train,y_train)

y_pred = enet_model.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,y_pred)))

enet_cv_model = ElasticNetCV(cv=10,random_state=0).fit(x_train,y_train)

enet_tuned = ElasticNet(alpha=enet_cv_model.alpha_).fit(x_train,y_train)
y_pred = enet_tuned.predict(x_test)



"""

