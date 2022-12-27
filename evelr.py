import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/Purpl/Desktop/Code projects/Datascience/Eve market data/domain data/domain-history-master2.csv")
data = df.drop(['highest', 'lowest', 'highlow_difference', 'typeName'], axis=1)

mask = data.dtypes == object
categorical_cols = data.columns[mask]

num_ohc_cols = (data[categorical_cols]
               .apply(lambda x: x.nunique())
               .sort_values(ascending=False))

small_num_ohc_cols = num_ohc_cols.loc[num_ohc_cols>1]

small_num_ohc_cols -= 1

from sklearn.preprocessing import OneHotEncoder

data_ohc = data.copy()

ohc = OneHotEncoder()

for col in num_ohc_cols.index:

    new_dat = ohc.fit_transform(data_ohc[[col]])

    data_ohc = data_ohc.drop(col, axis=1)

    cats = ohc.categories_

    new_cols = ['_'.join([col,cat]) for cat in cats[0]]

    new_df = pd.DataFrame(new_dat.toarray(),columns=new_cols)

    data_ohc = pd.concat([data_ohc, new_df], axis=1)


data = data.drop(num_ohc_cols.index, axis=1)



y_col = 'average'

###Split the data that is not one-hot encoded
feature_cols = [x for x in data.columns if x != y_col]
X_data = data[feature_cols]
y_data = data[y_col]

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

#split the data that is one-hot encoded
feature_cols = [x for x in data_ohc.columns if x !=y_col]
X_data_ohc = data_ohc[feature_cols]
y_data_ohc = data_ohc[y_col]

X_train_ohc, X_test_ohc, y_train_ohc, y_test_ohc = train_test_split(X_data_ohc, y_data_ohc, test_size=0.3, random_state=42)

LR = LinearRegression()

#storage for error values
error_df = list()

#Data that have not been one hot encoded
LR = LR.fit(X_train, y_train)
y_train_pred = LR.predict(X_train)
y_test_pred = LR.predict(X_test)

error_df.append(pd.Series({'train': mean_squared_error(y_train, y_train_pred),
                          'test': mean_squared_error(y_test, y_test_pred)},
                         name='no enc'))

#Data that have been one-hot encoded
LR = LR.fit(X_train_ohc, y_train_ohc)
y_train_ohc_pred = LR.predict(X_train_ohc)
y_test_ohc_pred = LR.predict(X_test_ohc)

error_df.append(pd.Series({'train': mean_squared_error(y_train_ohc, y_train_ohc_pred),
                          'test': mean_squared_error(y_test_ohc, y_test_ohc_pred)},
                         name='one-hot enc'))

#Assemble the results
error_df = pd.concat(error_df, axis=1)
print(error_df)


###plot results
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('dark')

ax = plt.axes()
#we are going to use y_test and y_test_pred
ax.scatter(y_test, y_test_pred, alpha=.5)

ax.set(xlabel='Ground truth',
      ylabel='Predictions',
      title='Eve market data Average predictions Vs Truth, using Linear Regression')
plt.show()
