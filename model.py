import pandas as pd 
import numpy as np 
import keras
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# Data Loading
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
center_data=pd.read_csv("fulfilment_center_info.csv")
meal_data=pd.read_csv("meal_info.csv")

######### Data Analysis ##########
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

################## Data Preparations ###############

#Joining train_set with center_dataset
train_center=pd.merge(train,center_data,how='inner',on='center_id')
test_center=pd.merge(test,center_data,how='inner',on='center_id')

#orders per center_type
train_center.groupby('center_type')['num_orders'].describe()

#Joining the meal_data
train_center_meal=pd.merge(train_center,meal_data,how='left',on='meal_id')
test_center_meal=pd.merge(test_center,meal_data,how='left',on='meal_id')
#change how---inner-left
 
train_data=train_center_meal.copy()
test_data=test_center_meal.copy()

x=train_data.drop(['num_orders'],1)
y=train_data['num_orders']
x_test=test_data

#Dropping attributes
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


# enc=LabelEncoder()
# x['center_type']=enc.fit_transform(x['center_type'])
# x['category']=enc.fit_transform(x['category'])
# x['cuisine']=enc.fit_transform(x['cuisine'])
# x_test['cuisine']=enc.fit_transform(x_test['cuisine'])
# x_test['category']=enc.fit_transform(x_test['category'])
# x_test['center_type']=enc.fit_transform(x_test['center_type'])
 

# Neural Network
model=Sequential()
model.add(Dense(64,input_dim=32,kernel_initializer='normal',activation='relu'))
model.add(Dense(64,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.25))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dense(32,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16,activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='linear'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
train=model.fit(x,y,epochs=10,batch_size=32,verbose=1,validation_split=0.25)

y_pred=model.predict(x_test)
y_pred=np.exp(y_pred)
y_pred=np.reshape(y_pred,-1)

s=pd.DataFrame({'id':test['id'],'num_orders':y_pred})
s.to_csv("ANN_final.csv",index=False)
