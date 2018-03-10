
# coding: utf-8

# ___
# This notebook tries to do a simple prediction of which team will win in a random match based on seeding

# In[51]:


import pandas as pd
import numpy as np
import scipy
from sklearn import *


# ## Read data

# In[36]:


raw_data = pd.read_csv("input/tour-results-seed.csv")


# ## Data Transformation
# - Differential in seeding
# - winning results

# In[40]:


winning_team_perspective_df = (
    raw_data
    .pipe(lambda x:x.assign(diff_seed = x.L_seed - x.W_seed))
    .pipe(lambda x:x.assign(yhat = 1))
)


# In[41]:


losing_team_perspective_df = (
    raw_data
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    .pipe(lambda x:x.assign(yhat = 0))
)


# In[99]:


prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)


# ## Splitting data into train and test sets
# - train data: <= 2013 Season
# - test data: >= 2014 Season

# In[48]:


train_df = prediction_df.query("Season <= 2013")
test_df = prediction_df.query("Season >= 2014")


# In[82]:


train_data_x = train_df[['diff_seed']]
train_data_y = train_df['yhat']

test_data_x = test_df[['diff_seed']]
test_data_y = test_df['yhat']


# ## Initializing Logistics Regression

# In[83]:


logreg = linear_model.LogisticRegression()


# In[84]:


logreg.fit(train_data_x,train_data_y)


# ## Getting accuracy of logistics regression
# - 0.71 accurate
# - include confusion matrix

# In[103]:


logreg.score(test_data_x,test_data_y)


# In[87]:


logreg.score(test_data_x,test_data_y)


# In[106]:


metrics.confusion_matrix(test_df.yhat,test_df.prediction_results)


# ## Joining prediction to actual dataframe

# In[ ]:


test_results = pd.DataFrame(logreg.predict(test_df[['diff_seed']])).rename(columns={0:"prediction_result"})


# In[93]:


test_df['prediction_results'] = test_results.prediction_result.values


# In[96]:


test_df.tail(20)


# http://blog.yhat.com/posts/roc-curves.html

# # Concluding remarks for logistics regression modelling
# - current approach isnt going to work for predicting results
#     - we are using post results to predict post results
# - will need to use regular season to predict out winning probability

# ## Next Steps
# - combining both regular season and post season for determining winner
#     - refer to images (link)
# - calculate out intermediate variables for prediction
# - all our features into prediction model
# - feature selection will be utilised later to decide which ones remain
# - ensure overfitting doesnt exist
# - try out different models
