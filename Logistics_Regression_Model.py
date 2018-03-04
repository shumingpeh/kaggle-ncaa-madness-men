
# coding: utf-8

# ___
# This notebook tries to do a simple prediction of which team will win in a random match based on seeding



import pandas as pd
import numpy as np
import scipy
from sklearn import *


# ## Read data



raw_data = pd.read_csv("input/tour-results-seed.csv")


# ## Data Transformation
# - Differential in seeding
# - winning results



winning_team_perspective_df = (
    raw_data
    .pipe(lambda x:x.assign(diff_seed = x.L_seed - x.W_seed))
    .pipe(lambda x:x.assign(yhat = 1))
)




losing_team_perspective_df = (
    raw_data
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    .pipe(lambda x:x.assign(yhat = 0))
)




prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)


# ## Splitting data into train and test sets
# - train data: <= 2013 Season
# - test data: >= 2014 Season



train_df = prediction_df.query("Season <= 2013")
test_df = prediction_df.query("Season >= 2014")




train_data_x = train_df[['diff_seed']]
train_data_y = train_df['yhat']

test_data_x = test_df[['diff_seed']]
test_data_y = test_df['yhat']


# ## Initializing Logistics Regression



logreg = linear_model.LogisticRegression()




logreg.fit(train_data_x,train_data_y)


# ## Getting accuracy of logistics regression
# - 0.71 accurate
# - include confusion matrix



logreg.score(test_data_x,test_data_y)




logreg.score(test_data_x,test_data_y)




metrics.confusion_matrix(test_df.yhat,test_df.prediction_results)


# ## Joining prediction to actual dataframe



test_results = pd.DataFrame(logreg.predict(test_df[['diff_seed']])).rename(columns={0:"prediction_result"})




test_df['prediction_results'] = test_results.prediction_result.values




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
