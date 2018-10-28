
___
this notebook select features that are important before throwing into the model


```python
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.feature_selection import *
from sklearn import *
from scipy import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel, SelectPercentile,RFECV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from aggregate_function import build_features_table, combine_features_table, coach_stats, win_rate_type_of_location
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/learning_curve.py:22: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the functions are moved. This module will be removed in 0.20
      DeprecationWarning)



```python
coach_file = 'data/DataFiles/TeamCoaches.csv'
regularseason_file = 'data/DataFiles/RegularSeasonDetailedResults.csv'
postseason_file = 'data/DataFiles/NCAATourneyCompactResults.csv'
```


```python
initial_features = build_features_table.BuildFeaturesTable(regularseason_file)
win_rate_features = win_rate_type_of_location.WinRateTypeLocation(regularseason_file)
coach_features = coach_stats.CoachStats(coach_file,regularseason_file,postseason_file)

features = combine_features_table.CombineFeaturesTable(initial_features,win_rate_features,coach_features)
```

## Data Transformation for recency data
- going to apply a flat weightage of 
    - 85% to current year
    - 15% to previous year


```python
features_table = features.final_table

features_table.head()
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
      <th>TeamID</th>
      <th>win_rate</th>
      <th>total_score</th>
      <th>total_opponent_score</th>
      <th>fgp</th>
      <th>fg3p</th>
      <th>ftp</th>
      <th>total_rebounds</th>
      <th>total_off_rebounds</th>
      <th>...</th>
      <th>avg_win_score_by</th>
      <th>win_rate_away</th>
      <th>win_rate_home</th>
      <th>win_rate_neutral</th>
      <th>num_season</th>
      <th>is_playoff</th>
      <th>is_champion</th>
      <th>win_rate_post</th>
      <th>win_rate_regular</th>
      <th>win_rate_overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>1101</td>
      <td>0.095238</td>
      <td>1326.0</td>
      <td>1651.0</td>
      <td>0.405508</td>
      <td>0.373333</td>
      <td>0.746067</td>
      <td>595.0</td>
      <td>168.0</td>
      <td>...</td>
      <td>3.500000</td>
      <td>0.000000</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1101</td>
      <td>0.250000</td>
      <td>1708.0</td>
      <td>2012.0</td>
      <td>0.404858</td>
      <td>0.378026</td>
      <td>0.727924</td>
      <td>781.0</td>
      <td>231.0</td>
      <td>...</td>
      <td>12.142857</td>
      <td>0.142857</td>
      <td>0.222222</td>
      <td>0.666667</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>1101</td>
      <td>0.333333</td>
      <td>1886.0</td>
      <td>2059.0</td>
      <td>0.441621</td>
      <td>0.363458</td>
      <td>0.706985</td>
      <td>829.0</td>
      <td>221.0</td>
      <td>...</td>
      <td>7.555556</td>
      <td>0.285714</td>
      <td>0.333333</td>
      <td>0.500000</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>1101</td>
      <td>0.360000</td>
      <td>1697.0</td>
      <td>1816.0</td>
      <td>0.458365</td>
      <td>0.371069</td>
      <td>0.642241</td>
      <td>761.0</td>
      <td>189.0</td>
      <td>...</td>
      <td>4.666667</td>
      <td>0.444444</td>
      <td>0.312500</td>
      <td>0.000000</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1102</td>
      <td>0.428571</td>
      <td>1603.0</td>
      <td>1596.0</td>
      <td>0.481149</td>
      <td>0.375643</td>
      <td>0.651357</td>
      <td>588.0</td>
      <td>117.0</td>
      <td>...</td>
      <td>15.583333</td>
      <td>0.428571</td>
      <td>0.473684</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
features_table (
    features.final_table
    .pipe(lambda x:x.assign(shifted_team = x.TeamID.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate = x.win_rate.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_score = x.total_score.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_opponent_score = x.total_opponent_score.shift(+1)))
    .pipe(lambda x:x.assign(shifted_fgp = x.fgp.shift(+1)))
    .pipe(lambda x:x.assign(shifted_fg3p = x.fg3p.shift(+1)))
    .pipe(lambda x:x.assign(shifted_ftp = x.ftp.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_rebounds = x.total_rebounds.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_off_rebounds = x.total_off_rebounds.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_def_rebounds = x.total_def_rebounds.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_off_rebounds_percent = x.total_off_rebounds_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_def_rebounds_percent = x.total_def_rebounds_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_rebound_possession_percent = x.total_rebound_possession_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_rebound_possessiongain_percent = x.total_rebound_possessiongain_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_blocks = x.total_blocks.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assists = x.total_assists.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_steals = x.total_steals.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_turnover = x.total_turnover.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_personalfoul = x.total_personalfoul.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_block_opp_FGA_percent = x.total_block_opp_FGA_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assist_per_fgm = x.total_assist_per_fgm.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assist_turnover_ratio = x.total_assist_turnover_ratio.shift(+1)))
    .pipe(lambda x:x.assign(shifted_expectation_per_game = x.expectation_per_game.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assists = x.avg_lose_score_by.shift(+1)))
    .pipe(lambda x:x.assign(shifted_avg_lose_score_by = x.avg_win_score_by.shift(+1)))
    .pipe(lambda x:x.assign(shifted_avg_win_score_by = x.avg_win_score_by.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate_away = x.win_rate_away.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate_home = x.win_rate_home.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate_neutral = x.win_rate_neutral.shift(+1)))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_total_score = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_total_opponent_score = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_fgp = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_fg3p = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_ftp = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_total_rebounds = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
)
```




    Season                                    int64
    TeamID                                    int64
    win_rate                                float64
    total_score                             float64
    total_opponent_score                    float64
    fgp                                     float64
    fg3p                                    float64
    ftp                                     float64
    total_rebounds                          float64
    total_off_rebounds                      float64
    total_def_rebounds                      float64
    total_off_rebounds_percent              float64
    total_def_rebounds_percent              float64
    total_rebound_possession_percent        float64
    total_rebound_possessiongain_percent    float64
    total_blocks                            float64
    total_assists                           float64
    total_steals                            float64
    total_turnover                          float64
    total_personalfoul                      float64
    total_block_opp_FGA_percent             float64
    total_assist_per_fgm                    float64
    total_assist_turnover_ratio             float64
    expectation_per_game                    float64
    avg_lose_score_by                       float64
    avg_win_score_by                        float64
    win_rate_away                           float64
    win_rate_home                           float64
    win_rate_neutral                        float64
    num_season                              float64
    is_playoff                              float64
    is_champion                             float64
    win_rate_post                           float64
    win_rate_regular                        float64
    win_rate_overall                        float64
    dtype: object


