import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import GradientBoostingRegressor

# Data Loading
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
center_data=pd.read_csv("fulfilment_center_info.csv")
meal_data=pd.read_csv("meal_info.csv")

######### Exploratory Data Analysis ##########
train.shape
train.isnull().sum()
train.info()
train.head()
weeks=np.unique(train.week)
centers=np.unique(train.center_id)
len(weeks)
len(centers) 

center_data.isnull().sum()
center_data.head()

meal_data.shape
meal_data.tail()

#checkout_price vs base_price
x=train[train.checkout_price > train.base_price]
len(x)

#Adding new feature
train['price_diff_frac']=(train['base_price']-train['checkout_price'])/(train['base_price'])
test['price_diff_frac']=(test['base_price']-test['checkout_price'])/(test['base_price'])

x=train.groupby(['week'])['num_orders'].mean().reset_index()
sns.lmplot('week','num_orders',data=x,fit_reg=False)
#no proper trend on increasing week

x=train.groupby(['week','price_diff_frac']).mean().reset_index()
sns.lmplot('price_diff_frac','num_orders',data=x,hue='week',fit_reg=False)
#most of the orders are given when checkout_price<base_price

#Correlation Analysis:
corr_all=train.corr()
print(corr_all)
mask=np.zeros_like(corr_all,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
f,ax=plt.subplots(figsize=(11,9))
sns.heatmap(corr_all,mask=mask,square=True,linewidths=.5,ax=ax)
#plot shows none of the attributes are correlated


################### Data Preparation #######################

#Joining train_set with center_dataset
train_center=pd.merge(train,center_data,how='left',on='center_id')
test_center=pd.merge(test,center_data,how='left',on='center_id')

#orders per center_type
train_center.groupby('center_type')['num_orders'].describe()

#Joining the meal_data
train_center_meal=pd.merge(train_center,meal_data,how='left',on='meal_id')
test_center_meal=pd.merge(test_center,meal_data,how='left',on='meal_id')

train_data=train_center_meal.copy()
test_data=test_center_meal.copy()

x=train_data.drop(['num_orders'],1)
y=train_data['num_orders']
x_test=test_data

x=x.drop(['id'],1)
x_test=x_test.drop(['id'],1)

x=pd.get_dummies(x)
x_test=pd.get_dummies(x_test)

def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

#Ensemble Model - Gradient Boost
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01)
y=np.log1p(y)
gbm.fit(x,y)
y_pred=gbm.predict(x_test)
y_pred=np.exp(y_pred)

print(gbm.feature_importances_)

#saving the trained predictions
s=pd.DataFrame({'id':test['id'],'num_orders':y_pred})
s.to_csv("gbm_reg.csv",index=False)