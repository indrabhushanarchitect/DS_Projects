#!/usr/bin/env python
# coding: utf-8

# ### Data Science - Car Prediction Project with Random Forest ML Algorithm 

# ##### Importing the Pandas Library From Python For Data Manipulation

# In[2]:


import pandas as pd


# ##### Loading the Car's Data For Data Manipulation & Prediction

# In[3]:


df=pd.read_csv('E:\Data_Science_Training\Car_Prediction_Project\Car-Price-Prediction-master\Cars_Data.csv')


# ###### Displaying Number of Records & Fields Available in the Cars Data

# In[38]:


print("")
print(" Displaying Few Recrods of the Cars Data")
print("")
print(df)
print("")
print("Number Of Observations & Features : ", df.shape)


# ######  Displaying the UNIQUE Number of Records in the Cars Data

# In[23]:


print("Seller_Type - ", df['Seller_Type'].unique())
print("Fuel_Type   - ", df['Fuel_Type'].unique())
print("Transmission- ", df['Transmission'].unique())
print("Owner       - ", df['Owner'].unique())


# ######  Checking the Sum Of Null Number of Records - Missing Values in the Cars Data

# In[70]:


df.isnull().sum()


# ###### The describe() method returns description of the data in the DataFrame and used for calculating some statistical data like percentile, mean and std of the numerical values of the Series or DataFrame

# In[71]:


df.describe()


# ######  Crating the final features of the Cars Data 

# In[26]:


final_dataset=df[['Year','Selling_Price','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]


# In[27]:


final_dataset.head()


# ######  Finalising the finale data for the Current Year 2023 from Cars Data

# In[32]:


final_dataset['Current Year']=2023


# In[33]:


final_dataset.head()


# ######  Providing the Age of Cars from Cars Data

# In[36]:


final_dataset['no_year']=final_dataset['Current Year']- final_dataset['Year']


# In[37]:


final_dataset.head()


# ######  Dropping the Year Feature From the Cars Data

# In[39]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[40]:


final_dataset.head()


# ######  pandas.get_dummies() is used for data manipulation. It converts categorical data into dummy or indicator variables i.e. 0 and 1

# In[41]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[42]:


final_dataset.head()


# ######  Dropping the Currenty Year Feature From the Cars Data

# In[83]:


final_dataset=final_dataset.drop(['Current Year'],axis=1)


# In[46]:


final_dataset.head()


# ######   Very Important Step to FIND OUT the corelation between the features From the Cars Data

# In[47]:


final_dataset.corr()


# ######  To get more into the Corelation bewteen the Features, We would prefer to use Seaborn for the Cars Data, Where we can plot the Features Corelation

# In[50]:


import seaborn as sns


# In[53]:


sns.pairplot(final_dataset)


# ######  Get correlations of each features in dataset with HEAT MAP the Cars Data

# In[64]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[66]:


final_dataset.head()


# ##### All Independent Feature Assignment to X and Dependent Feature to Y

# In[56]:


X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]


# In[70]:


X.head()


# In[72]:


y.head()


# ##### All Unique Owner Feature from X

# In[57]:


X['Owner'].unique()


# ##### Feature with higer Importance

# In[75]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X,y)


# In[76]:


print(model.feature_importances_)


# ##### Plot Graph of feature importances for better visualization

# In[77]:


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# ##### Train, Test & Split the Model

# In[78]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[79]:


X_train.shape


# In[80]:


X_test.shape


# ##### Random Forest Regresssion

# In[81]:


from sklearn.ensemble import RandomForestRegressor


# In[82]:


regressor=RandomForestRegressor()


# ##### Hyperparameter - Inhance the Model Perfomance

# In[86]:


import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# ##### RandomizedSearchCV - Help us the best parameter to choose

# In[87]:


from sklearn.model_selection import RandomizedSearchCV


# In[88]:


#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# ###### Creat random grid for the parameters

# In[89]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# ###### Use the random grid to search for best hyperparameters
# ###### First create the base model to tune

# In[90]:


rf = RandomForestRegressor()


# ###### Random search of parameters, using 3 fold cross validation, 
# ###### search across 100 different combinations

# In[102]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# ###### Random Forest Fit with Train & Test Observations

# In[ ]:


rf_random.fit(X_train,y_train)


# In[93]:


rf_random.best_params_


# In[94]:


rf_random.best_score_


# ###### Predictions

# In[97]:


predictions=rf_random.predict(X_test)


# ###### Displaying the prediction on Plot

# In[96]:


sns.distplot(y_test-predictions)


# ###### Displaying the prediction on Plot is Linear

# In[98]:


plt.scatter(y_test,predictions)


# ###### Metrics for Accuracy

# In[99]:


from sklearn import metrics


# In[100]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ###### Pickle File Creation

# In[101]:


import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)

