import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#get rid of coliumns with too little data
null = train.isnull().sum() / train.shape[0]
null0 = test.isnull().sum() / test.shape[0]
null.to_csv('null.csv')
null0.to_csv('null0.csv')
train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
#fill the rest of unavailable data with the mean
train = train.fillna(train.mean())
test = test.fillna(train.mean())
#find and get rid of outliers
#for i in range(train.shape[1]):
#    if train.dtypes[i] == 'int64' or train.dtypes[i] == 'float64':
#        sns.regplot(x=train.columns[i], y='SalePrice', data=train)
#        plt.show()
train = train.loc[train['GrLivArea'] < 4000]
train = train.loc[train['SalePrice'] < 700000]
#change object type to int8 type for regresion
train = train.fillna(train.mode())
test = test.fillna(test.mode())
SalePrice = train['SalePrice']
train_test = pd.concat([train.drop('SalePrice', axis=1), test], ignore_index=True)
train_test = pd.get_dummies(data=train_test, drop_first=True)
train = train_test[:train.shape[0]]
train = train.reset_index(drop=True)
test = train_test[train.shape[0]:]
test = test.reset_index(drop=True)
train['SalePrice'] = SalePrice
train = train.fillna(train.mean())
test = test.fillna(test.mean())
#test train split
y = SalePrice
x = list(train.columns[1:train.shape[1]-1])
X = train[x]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
#create object, fit, and predict
#linear regressioin
linReg = LinearRegression()
linReg.fit(X_train, y_train)
yhat = linReg.predict(X_test)
#svr
#svr = Pipeline(steps=[('scalar', StandardScaler()), ('sup_vec_reg', SVR())])
#svr.fit(X_train, y_train)
#yhat0 = svr.predict(X_test)
#display accuracy
print('Linear Regression:')
print('R^2 score:', metrics.r2_score(y_test, yhat))
print('Adjusted R^2 score:', 1 - (1-metrics.r2_score(y_test, yhat))*(len(y)-1)/(len(y)-X.shape[1]-1))
print('RMSE score:', metrics.mean_squared_error(y_test, yhat) ** 0.5)
print('Cross Validation score:', cross_val_score(linReg, X, y, cv=2).mean())
#print('SVR:')
#print('R^2 score:', metrics.r2_score(y_test, yhat0))
#print('Adjusted R^2 score:', 1 - (1-metrics.r2_score(y_test, yhat0))*(len(y)-1)/(len(y)-X.shape[1]-1))
#print('RMSE score:', metrics.mean_squared_error(y_test, yhat0) ** 0.5)
#print('Cross Validation score:', cross_val_score(svr, X, y, cv=2).mean())
#predict test values using best regression
x0 = list(test.columns[1:test.shape[1]])
X0 = test[x0]
linReg.fit(X, y)
yhat = linReg.predict(X0)
yhat_df = pd.DataFrame(yhat).round(2)
yhat_df.columns = ['SalePrice']
yhat_df.index.names = ['Id']
yhat_df.index = yhat_df.index + 1461
yhat_df.to_csv('predict.csv')
#display prediction results
plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
plt.plot(yhat_df, '.')
plt.axis([1400,2950,0,750000])
plt.title('ID vs. Predicted Sale Price')
plt.subplot(1,2,2)
plt.plot(train['Id'], train['SalePrice'],'.')
plt.axis([-50,1500,0,750000])
plt.title('ID vs. Sale Price')
plt.show()