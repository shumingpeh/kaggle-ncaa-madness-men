
___
This notebook tries to do a simple prediction of which team will win in a random match based on seeding


```python
import pandas as pd
import numpy as np
import scipy
from sklearn import *
```

## Read data


```python
raw_data = pd.read_csv("input/tour-results-seed.csv")
```

## Data Transformation
- Differential in seeding
- winning results


```python
winning_team_perspective_df = (
    raw_data
    .pipe(lambda x:x.assign(diff_seed = x.L_seed - x.W_seed))
    .pipe(lambda x:x.assign(yhat = 1))
)
```


```python
losing_team_perspective_df = (
    raw_data
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    .pipe(lambda x:x.assign(yhat = 0))
)
```


```python
prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)
```

## Splitting data into train and test sets
- train data: <= 2013 Season
- test data: >= 2014 Season


```python
train_df = prediction_df.query("Season <= 2013")
test_df = prediction_df.query("Season >= 2014")
```


```python
train_data_x = train_df[['diff_seed']]
train_data_y = train_df['yhat']

test_data_x = test_df[['diff_seed']]
test_data_y = test_df['yhat']
```

## Initializing Logistics Regression


```python
logreg = linear_model.LogisticRegression()
```


```python
logreg.fit(train_data_x,train_data_y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



## Getting accuracy of logistics regression
- 0.71 accurate
- include confusion matrix


```python
logreg.score(test_data_x,test_data_y)
```




    0.70895522388059706




```python
logreg.score(test_data_x,test_data_y)
```




    0.70895522388059706




```python
metrics.confusion_matrix(test_df.yhat,test_df.prediction_results)
```




    array([[180,  88],
           [ 68, 200]])



## Joining prediction to actual dataframe


```python
test_results = pd.DataFrame(logreg.predict(test_df[['diff_seed']])).rename(columns={0:"prediction_result"})
```


```python
test_df['prediction_results'] = test_results.prediction_result.values
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.



```python
test_df.tail(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>WTeamID</th>
      <th>W_seed</th>
      <th>LTeamID</th>
      <th>L_seed</th>
      <th>diff_seed</th>
      <th>yhat</th>
      <th>prediction_results</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2097</th>
      <td>2017</td>
      <td>1276</td>
      <td>7</td>
      <td>1257</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2098</th>
      <td>2017</td>
      <td>1314</td>
      <td>1</td>
      <td>1116</td>
      <td>8</td>
      <td>-7</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2099</th>
      <td>2017</td>
      <td>1332</td>
      <td>3</td>
      <td>1348</td>
      <td>11</td>
      <td>-8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2100</th>
      <td>2017</td>
      <td>1376</td>
      <td>7</td>
      <td>1181</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2101</th>
      <td>2017</td>
      <td>1417</td>
      <td>3</td>
      <td>1153</td>
      <td>6</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2102</th>
      <td>2017</td>
      <td>1211</td>
      <td>1</td>
      <td>1452</td>
      <td>4</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2103</th>
      <td>2017</td>
      <td>1242</td>
      <td>1</td>
      <td>1345</td>
      <td>4</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2104</th>
      <td>2017</td>
      <td>1332</td>
      <td>3</td>
      <td>1276</td>
      <td>7</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2105</th>
      <td>2017</td>
      <td>1462</td>
      <td>11</td>
      <td>1112</td>
      <td>2</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2106</th>
      <td>2017</td>
      <td>1196</td>
      <td>4</td>
      <td>1458</td>
      <td>8</td>
      <td>-4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2107</th>
      <td>2017</td>
      <td>1246</td>
      <td>2</td>
      <td>1417</td>
      <td>3</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2108</th>
      <td>2017</td>
      <td>1314</td>
      <td>1</td>
      <td>1139</td>
      <td>4</td>
      <td>-3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2109</th>
      <td>2017</td>
      <td>1376</td>
      <td>7</td>
      <td>1124</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2110</th>
      <td>2017</td>
      <td>1211</td>
      <td>1</td>
      <td>1462</td>
      <td>11</td>
      <td>-10</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2111</th>
      <td>2017</td>
      <td>1332</td>
      <td>3</td>
      <td>1242</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2112</th>
      <td>2017</td>
      <td>1314</td>
      <td>1</td>
      <td>1246</td>
      <td>2</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2113</th>
      <td>2017</td>
      <td>1376</td>
      <td>7</td>
      <td>1196</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2114</th>
      <td>2017</td>
      <td>1211</td>
      <td>1</td>
      <td>1376</td>
      <td>7</td>
      <td>-6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2115</th>
      <td>2017</td>
      <td>1314</td>
      <td>1</td>
      <td>1332</td>
      <td>3</td>
      <td>-2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2116</th>
      <td>2017</td>
      <td>1314</td>
      <td>1</td>
      <td>1211</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



http://blog.yhat.com/posts/roc-curves.html

# Concluding remarks for logistics regression modelling
- current approach isnt going to work for predicting results
    - we are using post results to predict post results
- will need to use regular season to predict out winning probability

## Next Steps
- combining both regular season and post season for determining winner
    - refer to images (link)
- calculate out intermediate variables for prediction
- all our features into prediction model
- feature selection will be utilised later to decide which ones remain
- ensure overfitting doesnt exist
- try out different models
