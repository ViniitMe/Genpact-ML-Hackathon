import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb 
from sklearn import model_selection

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
y=np.log1p(y)


def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

x_train,x_valid,y_train,y_valid=model_selection.train_test_split(x,y,test_size=0.25,random_state=23)
d_train=lgb.Dataset(x_train,y_train)
d_valid=lgb.Dataset(x_valid,y_valid)
watchlist=[d_valid]

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

model=lgb.train(params,d_train,valid_sets=d_valid,num_boost_round=4000,early_stopping_rounds=10)
#num_boost_round=n_estimators
#early_stopping_round=will stop training if valid set doesn't improve in these rounds 

y_pred=model.predict(x_test,num_iteration=model.best_iteration)
y_pred=np.exp(y_pred)

#saving the trained predictions
s=pd.DataFrame({'id':test['id'],'num_orders':y_pred})
s.to_csv("lightgbm.csv",index=False)