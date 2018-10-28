
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

## Final data transformation for feature table
- post season is only what we care about
- post season match ups will be what we are joining all the features table to
- additional variable of seeding differential


```python
features_table = (
    features.final_table_cum_processed
    .drop(['total_score','total_opponent_score','total_rebounds','total_blocks',
           'total_assist_turnover_ratio','expectation_per_game', 'win_rate','fg3p','win_rate_overall',
           'total_off_rebounds_percent','total_def_rebounds_percent','total_rebound_possession_percent',
           'total_rebound_possessiongain_percent'
          ],1)
    .fillna(0)
)
```


```python
seeding_data = pd.read_csv("input/tour-results-seed.csv")
```


```python
winning_team_perspective_df = (
    seeding_data
    .pipe(lambda x:x.assign(diff_seed = x.L_seed - x.W_seed))
    .pipe(lambda x:x.assign(outcome = 1))
    .merge(features_table,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .merge(features_table,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .pipe(lambda x:x.assign(diff_total_off_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_off_rebounds_x - x.total_off_rebounds_y,
                                                                x.total_off_rebounds_y - x.total_off_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_def_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_def_rebounds_x - x.total_def_rebounds_y,
                                                                x.total_def_rebounds_y - x.total_def_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_assists = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assists_x - x.total_assists_y,
                                                                x.total_assists_y - x.total_assists_x)))
    .pipe(lambda x:x.assign(diff_total_steals = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_steals_x - x.total_steals_y,
                                                                x.total_steals_y - x.total_steals_x)))
    .pipe(lambda x:x.assign(diff_total_turnover = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_turnover_x - x.total_turnover_y,
                                                                x.total_turnover_y - x.total_turnover_x)))
    .pipe(lambda x:x.assign(diff_total_personalfoul = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_personalfoul_x - x.total_personalfoul_y,
                                                                x.total_personalfoul_y - x.total_personalfoul_x)))
    .pipe(lambda x:x.assign(diff_total_assist_per_fgm = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assist_per_fgm_x - x.total_assist_per_fgm_y,
                                                                x.total_assist_per_fgm_y - x.total_assist_per_fgm_x)))
    .pipe(lambda x:x.assign(diff_avg_lose_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_lose_score_by_x - x.avg_lose_score_by_y,
                                                                x.avg_lose_score_by_y - x.avg_lose_score_by_x)))
    .pipe(lambda x:x.assign(diff_avg_win_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_win_score_by_x - x.avg_win_score_by_y,
                                                                x.avg_win_score_by_y - x.avg_win_score_by_x)))
    .pipe(lambda x:x.assign(diff_num_season = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.num_season_x - x.num_season_y,
                                                                x.num_season_y - x.num_season_x)))
    .pipe(lambda x:x.assign(diff_is_playoff = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_playoff_x - x.is_playoff_y,
                                                                x.is_playoff_y - x.is_playoff_x)))
    .pipe(lambda x:x.assign(diff_is_champion = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_champion_x - x.is_champion_y,
                                                                x.is_champion_y - x.is_champion_x)))
    .pipe(lambda x:x.assign(diff_fgp = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.fgp_x - x.fgp_y,
                                                                x.fgp_y - x.fgp_x)))
    .pipe(lambda x:x.assign(diff_total_block_opp_FGA_percent = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_block_opp_FGA_percent_x - x.total_block_opp_FGA_percent_y,
                                                                x.total_block_opp_FGA_percent_y - x.total_block_opp_FGA_percent_x)))
    .pipe(lambda x:x.assign(diff_win_rate_away = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_away_x - x.win_rate_away_y,
                                                                x.win_rate_away_y - x.win_rate_away_x)))
    .pipe(lambda x:x.assign(diff_win_rate_home = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_home_x - x.win_rate_home_y,
                                                                x.win_rate_home_y - x.win_rate_home_x)))
    .pipe(lambda x:x.assign(diff_win_rate_neutral = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_neutral_x - x.win_rate_neutral_y,
                                                                x.win_rate_neutral_y - x.win_rate_neutral_x)))
    .pipe(lambda x:x.assign(diff_win_rate_post = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_post_x - x.win_rate_post_y,
                                                                x.win_rate_post_y - x.win_rate_post_x)))
    .pipe(lambda x:x.assign(diff_win_rate_regular = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_regular_x - x.win_rate_regular_y,
                                                                x.win_rate_regular_y - x.win_rate_regular_x)))
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))

    
)
```


```python
losing_team_perspective_df = (
    seeding_data
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    .pipe(lambda x:x.assign(outcome = 0))
    .merge(features_table,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .merge(features_table,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .pipe(lambda x:x.assign(diff_total_off_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_off_rebounds_x - x.total_off_rebounds_y,
                                                                x.total_off_rebounds_y - x.total_off_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_def_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_def_rebounds_x - x.total_def_rebounds_y,
                                                                x.total_def_rebounds_y - x.total_def_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_assists = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assists_x - x.total_assists_y,
                                                                x.total_assists_y - x.total_assists_x)))
    .pipe(lambda x:x.assign(diff_total_steals = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_steals_x - x.total_steals_y,
                                                                x.total_steals_y - x.total_steals_x)))
    .pipe(lambda x:x.assign(diff_total_turnover = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_turnover_x - x.total_turnover_y,
                                                                x.total_turnover_y - x.total_turnover_x)))
    .pipe(lambda x:x.assign(diff_total_personalfoul = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_personalfoul_x - x.total_personalfoul_y,
                                                                x.total_personalfoul_y - x.total_personalfoul_x)))
    .pipe(lambda x:x.assign(diff_total_assist_per_fgm = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assist_per_fgm_x - x.total_assist_per_fgm_y,
                                                                x.total_assist_per_fgm_y - x.total_assist_per_fgm_x)))
    .pipe(lambda x:x.assign(diff_avg_lose_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_lose_score_by_x - x.avg_lose_score_by_y,
                                                                x.avg_lose_score_by_y - x.avg_lose_score_by_x)))
    .pipe(lambda x:x.assign(diff_avg_win_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_win_score_by_x - x.avg_win_score_by_y,
                                                                x.avg_win_score_by_y - x.avg_win_score_by_x)))
    .pipe(lambda x:x.assign(diff_num_season = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.num_season_x - x.num_season_y,
                                                                x.num_season_y - x.num_season_x)))
    .pipe(lambda x:x.assign(diff_is_playoff = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_playoff_x - x.is_playoff_y,
                                                                x.is_playoff_y - x.is_playoff_x)))
    .pipe(lambda x:x.assign(diff_is_champion = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_champion_x - x.is_champion_y,
                                                                x.is_champion_y - x.is_champion_x)))
    .pipe(lambda x:x.assign(diff_fgp = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.fgp_x - x.fgp_y,
                                                                x.fgp_y - x.fgp_x)))
    .pipe(lambda x:x.assign(diff_total_block_opp_FGA_percent = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_block_opp_FGA_percent_x - x.total_block_opp_FGA_percent_y,
                                                                x.total_block_opp_FGA_percent_y - x.total_block_opp_FGA_percent_x)))
    .pipe(lambda x:x.assign(diff_win_rate_away = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_away_x - x.win_rate_away_y,
                                                                x.win_rate_away_y - x.win_rate_away_x)))
    .pipe(lambda x:x.assign(diff_win_rate_home = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_home_x - x.win_rate_home_y,
                                                                x.win_rate_home_y - x.win_rate_home_x)))
    .pipe(lambda x:x.assign(diff_win_rate_neutral = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_neutral_x - x.win_rate_neutral_y,
                                                                x.win_rate_neutral_y - x.win_rate_neutral_x)))
    .pipe(lambda x:x.assign(diff_win_rate_post = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_post_x - x.win_rate_post_y,
                                                                x.win_rate_post_y - x.win_rate_post_x)))
    .pipe(lambda x:x.assign(diff_win_rate_regular = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_regular_x - x.win_rate_regular_y,
                                                                x.win_rate_regular_y - x.win_rate_regular_x)))
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    
)
```


```python
prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)

train_df = prediction_df.query("Season >= 2003 & Season <= 2016")
test_df = prediction_df.query("Season == 2017")

train_df.head()
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
      <th>outcome</th>
      <th>total_off_rebounds_x</th>
      <th>total_def_rebounds_x</th>
      <th>total_assists_x</th>
      <th>...</th>
      <th>diff_num_season</th>
      <th>diff_is_playoff</th>
      <th>diff_is_champion</th>
      <th>diff_fgp</th>
      <th>diff_total_block_opp_FGA_percent</th>
      <th>diff_win_rate_away</th>
      <th>diff_win_rate_home</th>
      <th>diff_win_rate_neutral</th>
      <th>diff_win_rate_post</th>
      <th>diff_win_rate_regular</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1136</th>
      <td>2003</td>
      <td>1421</td>
      <td>16</td>
      <td>1411</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0.037785</td>
      <td>0.035770</td>
      <td>0.031439</td>
      <td>...</td>
      <td>0.152174</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>-0.018262</td>
      <td>0.012232</td>
      <td>-0.071429</td>
      <td>-0.162281</td>
      <td>0.25</td>
      <td>0.500000</td>
      <td>-0.151724</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>2003</td>
      <td>1112</td>
      <td>1</td>
      <td>1436</td>
      <td>16</td>
      <td>-15</td>
      <td>1</td>
      <td>0.047813</td>
      <td>0.043772</td>
      <td>0.045466</td>
      <td>...</td>
      <td>0.003557</td>
      <td>-0.59375</td>
      <td>-0.2</td>
      <td>-0.016969</td>
      <td>-0.011306</td>
      <td>-0.041667</td>
      <td>-0.370833</td>
      <td>0.60</td>
      <td>-0.641509</td>
      <td>-0.237685</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>2003</td>
      <td>1113</td>
      <td>10</td>
      <td>1272</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>0.043744</td>
      <td>0.036084</td>
      <td>0.040266</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-0.09375</td>
      <td>0.0</td>
      <td>0.040251</td>
      <td>-0.011396</td>
      <td>-0.206349</td>
      <td>-0.111111</td>
      <td>-0.50</td>
      <td>-0.397059</td>
      <td>-0.172414</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>2003</td>
      <td>1141</td>
      <td>11</td>
      <td>1166</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>0.030664</td>
      <td>0.036006</td>
      <td>0.040508</td>
      <td>...</td>
      <td>-0.243478</td>
      <td>-0.15625</td>
      <td>0.0</td>
      <td>0.005763</td>
      <td>-0.011456</td>
      <td>-0.100000</td>
      <td>-0.122024</td>
      <td>0.00</td>
      <td>0.250000</td>
      <td>-0.085684</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>2003</td>
      <td>1143</td>
      <td>8</td>
      <td>1301</td>
      <td>9</td>
      <td>-1</td>
      <td>1</td>
      <td>0.033425</td>
      <td>0.038516</td>
      <td>0.041838</td>
      <td>...</td>
      <td>-0.227668</td>
      <td>-0.12500</td>
      <td>0.0</td>
      <td>-0.009399</td>
      <td>0.010209</td>
      <td>-0.375000</td>
      <td>-0.114706</td>
      <td>0.25</td>
      <td>-0.100000</td>
      <td>-0.124138</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




```python
train_data_x = train_df[['diff_seed','diff_total_off_rebounds','diff_total_def_rebounds','diff_total_assists',
                         'diff_total_steals','diff_total_turnover','diff_total_personalfoul',
                         'diff_total_assist_per_fgm','diff_avg_lose_score_by','diff_avg_win_score_by',
                         'diff_num_season','diff_is_playoff','diff_is_champion','diff_fgp',
                         'diff_total_block_opp_FGA_percent','diff_win_rate_away','diff_win_rate_home',
                         'diff_win_rate_neutral','diff_win_rate_post','diff_win_rate_regular']]
train_data_y = train_df['outcome']

test_data_x = test_df[['diff_seed','diff_total_off_rebounds','diff_total_def_rebounds','diff_total_assists',
                         'diff_total_steals','diff_total_turnover','diff_total_personalfoul',
                         'diff_total_assist_per_fgm','diff_avg_lose_score_by','diff_avg_win_score_by',
                         'diff_num_season','diff_is_playoff','diff_is_champion','diff_fgp',
                         'diff_total_block_opp_FGA_percent','diff_win_rate_away','diff_win_rate_home',
                         'diff_win_rate_neutral','diff_win_rate_post','diff_win_rate_regular']]
test_data_y = test_df['outcome']
```

## Feature selection for logistics regression
- Univariate selection
- SelectFromModel from RF, lassoCV


```python
# univariate selection
percentile_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for i in percentile_list:
    select = SelectPercentile(percentile=i)

    select.fit(train_data_x, train_data_y)

    train_data_x_selected = select.transform(train_data_x)
    test_data_x_selected = select.transform(test_data_x)

    mask = select.get_support()    
#     print(mask)
    logreg = LogisticRegression()
    logreg.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal logreg: {}".format(logreg.score(test_data_x,test_data_y)))

    logreg.fit(train_data_x_selected,train_data_y)
    print("feature selection logreg: {}".format(logreg.score(test_data_x_selected,test_data_y)))
    print(Counter(logreg.predict(test_data_x_selected)[:67]))

    
# 2o percentile is the best FE for logistics regression
```

    
    Which percentile : 10
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 15
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 20
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 25
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 30
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 35
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 40
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 45
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 50
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 55
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 60
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 65
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 70
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 75
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 80
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 85
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 90
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 95
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 100
    normal logreg: 0.8059701492537313
    feature selection logreg: 0.8059701492537313
    Counter({1: 54, 0: 13})



```python
# based on the output of the univariate, we can narrow to 15
select_15 = SelectPercentile(percentile=15)
```


```python
select_15.fit(train_data_x, train_data_y)

train_data_x_selected_15 = select_15.transform(train_data_x)
test_data_x_selected_15 = select_15.transform(test_data_x)

mask = select_15.get_support()    
#     print(mask)
logreg_15 = LogisticRegression()
logreg_15.fit(train_data_x_selected_15,train_data_y)

logreg_15.score(test_data_x_selected_15,test_data_y)
logreg_15.predict(test_data_x_selected_15)[:67]
```




    array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])




```python
rf = RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [5,10,50,100,150,500,1000],
    'max_depth': [1,2,5,10,15,50,100]
}
grid_rf = GridSearchCV(rf, param_grid, scoring='accuracy', cv=5, verbose=0)
grid_rf.fit(train_data_x, train_data_y)

rf_model = grid_rf.best_estimator_
rf_model
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=15, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)




```python
## selectfrommodel RF
select_rf = SelectFromModel(RandomForestClassifier(n_estimators =150, max_depth = 15, random_state=0),threshold=0.05)
```


```python
select_rf.fit(train_data_x,train_data_y)
```




    SelectFromModel(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=15, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=150, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False),
            norm_order=1, prefit=False, threshold=0.05)




```python
train_data_x_selected = select_rf.transform(train_data_x)
test_data_x_selected = select_rf.transform(test_data_x)
```


```python
LogisticRegression().fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)
```




    0.83582089552238803




```python
LogisticRegression().fit(train_data_x_selected,train_data_y).predict(test_data_x_selected)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])




```python
## selectfrommodel lassoCV --> same as univariate
```


```python
select_lcv = SelectFromModel(LassoCV(max_iter=100,n_alphas=10,eps=1e-05),threshold=0.01)

select_lcv.fit(train_data_x,train_data_y)
```




    SelectFromModel(estimator=LassoCV(alphas=None, copy_X=True, cv=None, eps=1e-05, fit_intercept=True,
        max_iter=100, n_alphas=10, n_jobs=1, normalize=False, positive=False,
        precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
        verbose=False),
            norm_order=1, prefit=False, threshold=0.01)




```python
train_data_x_selected_lcv = select_lcv.transform(train_data_x)
test_data_x_selected_lcv = select_lcv.transform(test_data_x)
```


```python
LogisticRegression().fit(train_data_x_selected_lcv,train_data_y).score(test_data_x_selected_lcv,test_data_y)
```




    0.80597014925373134




```python
LogisticRegression().fit(train_data_x_selected_lcv,train_data_y).predict(test_data_x_selected_lcv)[:67]
```




    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])



## Feature selection for RF
- univariate
- SelectFromModel SVM
- RFE(CV)


```python
# univariate selection
percentile_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for i in percentile_list:
    select_rf = SelectPercentile(percentile=i)
    select_rf.fit(train_data_x, train_data_y)
    
    train_data_x_selected = select_rf.transform(train_data_x)
    test_data_x_selected = select_rf.transform(test_data_x)
    
    mask = select_rf.get_support()
#     print(mask)
    rf = RandomForestClassifier(n_estimators = 150, max_depth=15,random_state=0)
    rf.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal rf: {}".format(rf.score(test_data_x,test_data_y)))
    
    rf.fit(train_data_x_selected,train_data_y)
    print("feature selection rf: {}".format(rf.score(test_data_x_selected,test_data_y)))
    print(Counter(rf.predict(test_data_x_selected)[:67]))

```

    
    Which percentile : 10
    normal rf: 0.8582089552238806
    feature selection rf: 0.7985074626865671
    Counter({1: 53, 0: 14})
    
    Which percentile : 15
    normal rf: 0.8582089552238806
    feature selection rf: 0.7985074626865671
    Counter({1: 54, 0: 13})
    
    Which percentile : 20
    normal rf: 0.8582089552238806
    feature selection rf: 0.7985074626865671
    Counter({1: 53, 0: 14})
    
    Which percentile : 25
    normal rf: 0.8582089552238806
    feature selection rf: 0.7761194029850746
    Counter({1: 52, 0: 15})
    
    Which percentile : 30
    normal rf: 0.8582089552238806
    feature selection rf: 0.7985074626865671
    Counter({1: 53, 0: 14})
    
    Which percentile : 35
    normal rf: 0.8582089552238806
    feature selection rf: 0.7985074626865671
    Counter({1: 54, 0: 13})
    
    Which percentile : 40
    normal rf: 0.8582089552238806
    feature selection rf: 0.8283582089552238
    Counter({1: 56, 0: 11})
    
    Which percentile : 45
    normal rf: 0.8582089552238806
    feature selection rf: 0.8283582089552238
    Counter({1: 55, 0: 12})
    
    Which percentile : 50
    normal rf: 0.8582089552238806
    feature selection rf: 0.8432835820895522
    Counter({1: 57, 0: 10})
    
    Which percentile : 55
    normal rf: 0.8582089552238806
    feature selection rf: 0.8134328358208955
    Counter({1: 55, 0: 12})
    
    Which percentile : 60
    normal rf: 0.8582089552238806
    feature selection rf: 0.8283582089552238
    Counter({1: 55, 0: 12})
    
    Which percentile : 65
    normal rf: 0.8582089552238806
    feature selection rf: 0.8507462686567164
    Counter({1: 56, 0: 11})
    
    Which percentile : 70
    normal rf: 0.8582089552238806
    feature selection rf: 0.8432835820895522
    Counter({1: 56, 0: 11})
    
    Which percentile : 75
    normal rf: 0.8582089552238806
    feature selection rf: 0.835820895522388
    Counter({1: 55, 0: 12})
    
    Which percentile : 80
    normal rf: 0.8582089552238806
    feature selection rf: 0.8731343283582089
    Counter({1: 58, 0: 9})
    
    Which percentile : 85
    normal rf: 0.8582089552238806
    feature selection rf: 0.8582089552238806
    Counter({1: 58, 0: 9})
    
    Which percentile : 90
    normal rf: 0.8582089552238806
    feature selection rf: 0.8582089552238806
    Counter({1: 59, 0: 8})
    
    Which percentile : 95
    normal rf: 0.8582089552238806
    feature selection rf: 0.8432835820895522
    Counter({1: 58, 0: 9})
    
    Which percentile : 100
    normal rf: 0.8582089552238806
    feature selection rf: 0.8582089552238806
    Counter({1: 57, 0: 10})



```python
# based on the output of the univariate, we can narrow to 60, 80
select_80_rf = SelectPercentile(percentile=80)
select_90_rf = SelectPercentile(percentile=90)
```


```python
select_80_rf.fit(train_data_x, train_data_y)

train_data_x_selected_80_rf = select_80_rf.transform(train_data_x)
test_data_x_selected_80_rf = select_80_rf.transform(test_data_x)

mask = select_80_rf.get_support()        
# print(mask)
rf_80 = RandomForestClassifier(n_estimators = 150, max_depth=15,random_state=0,warm_start=False)
rf_80.fit(train_data_x_selected_80_rf,train_data_y)

rf_80.score(test_data_x_selected_80_rf,test_data_y)
```




    0.87313432835820892




```python
rf_80.predict(test_data_x_selected_80_rf)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])




```python
select_90_rf.fit(train_data_x, train_data_y)

train_data_x_selected_90_rf = select_90_rf.transform(train_data_x)
test_data_x_selected_90_rf = select_90_rf.transform(test_data_x)

mask = select_90_rf.get_support()        
# print(mask)
rf_90 = RandomForestClassifier(n_estimators = 150, max_depth=15,random_state=0,warm_start=False)
rf_90.fit(train_data_x_selected_90_rf,train_data_y)

rf_90.score(test_data_x_selected_90_rf,test_data_y)
```




    0.85820895522388063




```python
rf_90.predict(test_data_x_selected_90_rf)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])




```python
Counter(rf_80.predict(test_data_x_selected_80_rf)[:67])
```




    Counter({0: 9, 1: 58})




```python
Counter(rf_90.predict(test_data_x_selected_90_rf)[:67])
```




    Counter({0: 8, 1: 59})




```python
## RFE CV from LR
# for i in [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20]:
model_rfe = LogisticRegression()
rfe = RFECV(model_rfe, step=1, cv=5)
fit = rfe.fit(train_data_x, train_data_y)
print("Num Features: "+ str(fit.n_features_))
print("Selected Features: " + str(fit.support_))
print("Feature Ranking: " + str(fit.ranking_))

train_data_x_selected_rfe = fit.transform(train_data_x)
test_data_x_selected_rfe = fit.transform(test_data_x)

model_rfe.fit(train_data_x_selected_rfe,train_data_y).score(test_data_x_selected_rfe,test_data_y)
```

    Num Features: 10
    Selected Features: [False False  True False  True False False  True  True  True False  True
     False False False  True  True False  True  True]
    Feature Ranking: [11  6  1  5  1  9  3  1  1  1  7  1  8  4  2  1  1 10  1  1]





    0.82089552238805974




```python
rf_model = RandomForestClassifier(n_estimators = 150, max_depth=15,random_state=0,warm_start=False)
```


```python
rf_model.fit(train_data_x_selected_rfe,train_data_y).score(test_data_x_selected_rfe,test_data_y)
```




    0.85074626865671643




```python
rf_model.fit(train_data_x_selected_rfe,train_data_y).predict(test_data_x_selected_rfe)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])



## SVM


```python
## SVM
# univariate selection
percentile_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for i in percentile_list:
    select = SelectPercentile(percentile=i)

    select.fit(train_data_x, train_data_y)

    train_data_x_selected = select.transform(train_data_x)
    test_data_x_selected = select.transform(test_data_x)

    mask = select.get_support()    
#     print(mask)
    svm = SVC(probability=True)
    svm.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal svm: {}".format(svm.score(test_data_x,test_data_y)))

    svm.fit(train_data_x_selected,train_data_y)
    print("feature selection svm: {}".format(svm.score(test_data_x_selected,test_data_y)))
    print(Counter(svm.predict(test_data_x_selected)[:67]))
#     print(svm.predict(test_data_x_selected)[:67])
#     print(svm.predict_proba(test_data_x_selected)[0:10])

    
# 2o percentile is the best FE for logistics regression
```

    
    Which percentile : 10
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 15
    normal svm: 0.8955223880597015
    feature selection svm: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 20
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 25
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 30
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 35
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 40
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 45
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 50
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 55
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 60
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 65
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 70
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 75
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 80
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 85
    normal svm: 0.8955223880597015
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 90
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 95
    normal svm: 0.8955223880597015
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 100
    normal svm: 0.8955223880597015
    feature selection svm: 0.8955223880597015
    Counter({1: 60, 0: 7})



```python
test_df.head(10)
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
      <th>outcome</th>
      <th>total_off_rebounds</th>
      <th>total_def_rebounds</th>
      <th>total_assists</th>
      <th>...</th>
      <th>is_playoff</th>
      <th>is_champion</th>
      <th>TeamID</th>
      <th>fgp</th>
      <th>total_block_opp_FGA_percent</th>
      <th>win_rate_away</th>
      <th>win_rate_home</th>
      <th>win_rate_neutral</th>
      <th>win_rate_post</th>
      <th>win_rate_regular</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2053</th>
      <td>2017</td>
      <td>1243</td>
      <td>11</td>
      <td>1448</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0.474946</td>
      <td>0.679408</td>
      <td>0.636210</td>
      <td>...</td>
      <td>0.34375</td>
      <td>0.0</td>
      <td>1243.0</td>
      <td>0.458474</td>
      <td>0.068493</td>
      <td>0.500000</td>
      <td>0.647059</td>
      <td>0.666667</td>
      <td>0.521739</td>
      <td>0.738318</td>
    </tr>
    <tr>
      <th>2054</th>
      <td>2017</td>
      <td>1291</td>
      <td>16</td>
      <td>1309</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0.357298</td>
      <td>0.681874</td>
      <td>0.441624</td>
      <td>...</td>
      <td>0.06250</td>
      <td>0.0</td>
      <td>1291.0</td>
      <td>0.443673</td>
      <td>0.070649</td>
      <td>0.666667</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.530303</td>
    </tr>
    <tr>
      <th>2055</th>
      <td>2017</td>
      <td>1413</td>
      <td>16</td>
      <td>1300</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0.503268</td>
      <td>0.728730</td>
      <td>0.477157</td>
      <td>...</td>
      <td>0.06250</td>
      <td>0.0</td>
      <td>1413.0</td>
      <td>0.432099</td>
      <td>0.061336</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.666667</td>
      <td>0.600000</td>
      <td>0.645161</td>
    </tr>
    <tr>
      <th>2056</th>
      <td>2017</td>
      <td>1425</td>
      <td>11</td>
      <td>1344</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0.566449</td>
      <td>0.763255</td>
      <td>0.654822</td>
      <td>...</td>
      <td>0.09375</td>
      <td>0.0</td>
      <td>1425.0</td>
      <td>0.453718</td>
      <td>0.086307</td>
      <td>0.666667</td>
      <td>0.736842</td>
      <td>0.800000</td>
      <td>0.571429</td>
      <td>0.683673</td>
    </tr>
    <tr>
      <th>2057</th>
      <td>2017</td>
      <td>1112</td>
      <td>2</td>
      <td>1315</td>
      <td>15</td>
      <td>13</td>
      <td>1</td>
      <td>0.553377</td>
      <td>0.843403</td>
      <td>0.593909</td>
      <td>...</td>
      <td>0.31250</td>
      <td>0.0</td>
      <td>1112.0</td>
      <td>0.475707</td>
      <td>0.056555</td>
      <td>0.900000</td>
      <td>0.937500</td>
      <td>0.750000</td>
      <td>0.655172</td>
      <td>0.805471</td>
    </tr>
    <tr>
      <th>2058</th>
      <td>2017</td>
      <td>1139</td>
      <td>4</td>
      <td>1457</td>
      <td>13</td>
      <td>9</td>
      <td>1</td>
      <td>0.383442</td>
      <td>0.622688</td>
      <td>0.492386</td>
      <td>...</td>
      <td>0.09375</td>
      <td>0.0</td>
      <td>1139.0</td>
      <td>0.477586</td>
      <td>0.052378</td>
      <td>0.700000</td>
      <td>0.764706</td>
      <td>0.750000</td>
      <td>0.571429</td>
      <td>0.702128</td>
    </tr>
    <tr>
      <th>2059</th>
      <td>2017</td>
      <td>1196</td>
      <td>4</td>
      <td>1190</td>
      <td>13</td>
      <td>9</td>
      <td>1</td>
      <td>0.596950</td>
      <td>0.750925</td>
      <td>0.483926</td>
      <td>...</td>
      <td>0.03125</td>
      <td>0.0</td>
      <td>1196.0</td>
      <td>0.449551</td>
      <td>0.083874</td>
      <td>0.888889</td>
      <td>0.692308</td>
      <td>0.700000</td>
      <td>0.750000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>2060</th>
      <td>2017</td>
      <td>1199</td>
      <td>3</td>
      <td>1195</td>
      <td>14</td>
      <td>11</td>
      <td>1</td>
      <td>0.638344</td>
      <td>0.789149</td>
      <td>0.620981</td>
      <td>...</td>
      <td>0.25000</td>
      <td>0.0</td>
      <td>1199.0</td>
      <td>0.483268</td>
      <td>0.086777</td>
      <td>1.000000</td>
      <td>0.750000</td>
      <td>0.666667</td>
      <td>0.466667</td>
      <td>0.722222</td>
    </tr>
    <tr>
      <th>2061</th>
      <td>2017</td>
      <td>1211</td>
      <td>1</td>
      <td>1355</td>
      <td>16</td>
      <td>15</td>
      <td>1</td>
      <td>0.464052</td>
      <td>0.992602</td>
      <td>0.695431</td>
      <td>...</td>
      <td>0.56250</td>
      <td>0.0</td>
      <td>1211.0</td>
      <td>0.517829</td>
      <td>0.072989</td>
      <td>0.900000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.590909</td>
      <td>0.834395</td>
    </tr>
    <tr>
      <th>2062</th>
      <td>2017</td>
      <td>1235</td>
      <td>5</td>
      <td>1305</td>
      <td>12</td>
      <td>7</td>
      <td>1</td>
      <td>0.461874</td>
      <td>0.775586</td>
      <td>0.683587</td>
      <td>...</td>
      <td>0.09375</td>
      <td>0.0</td>
      <td>1235.0</td>
      <td>0.468629</td>
      <td>0.057589</td>
      <td>0.625000</td>
      <td>0.666667</td>
      <td>0.857143</td>
      <td>0.571429</td>
      <td>0.763441</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 27 columns</p>
</div>




```python
# based on the output of the univariate, we can narrow to 100
select_100_svm = SelectPercentile(percentile=100)

select_100_svm.fit(train_data_x, train_data_y)

train_data_x_selected_100_svm = select_100_svm.transform(train_data_x)
test_data_x_selected_100_svm = select_100_svm.transform(test_data_x)

mask = select_100_svm.get_support()        
# print(mask)
svm_100 = SVC(probability=True)
svm_100.fit(train_data_x_selected_100_svm,train_data_y)

svm_100.score(test_data_x_selected_100_svm,test_data_y)

svm_100.predict(test_data_x_selected_100_svm)[:67]
```




    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])




```python
rf_fi = RandomForestClassifier(n_estimators = 100, max_depth=10,random_state=0)
```


```python
rf_fi_values = (
    pd.DataFrame(rf_fi.fit(train_data_x,train_data_y).feature_importances_,index=train_data_x.columns)
    .rename(columns={0:"feature_importance_values"})
    .reset_index()
    .rename(columns={"index":"features"})
    .sort_values(['feature_importance_values'],ascending=False)
    .pipe(lambda x:x.assign(fi_cumsum = x.feature_importance_values.cumsum()))
    .query("fi_cumsum <= 0.95")
)
```


```python
rf_fi_values.features.unique()
```




    array(['diff_seed', 'win_rate_post', 'win_rate_home', 'avg_win_score_by',
           'win_rate_regular', 'fgp', 'total_assist_per_fgm',
           'total_block_opp_FGA_percent', 'is_playoff', 'total_steals',
           'total_def_rebounds', 'total_assists', 'total_turnover',
           'total_personalfoul', 'avg_lose_score_by', 'total_off_rebounds',
           'win_rate_away'], dtype=object)




```python
svm_train_data_x = train_df[['diff_seed', 'win_rate_post', 'win_rate_home', 'avg_win_score_by',
       'win_rate_regular', 'fgp', 'total_assist_per_fgm',
       'total_block_opp_FGA_percent', 'is_playoff', 'total_steals',
       'total_def_rebounds', 'total_assists', 'total_turnover',
       'total_personalfoul', 'avg_lose_score_by', 'total_off_rebounds',
       'win_rate_away']]
svm_train_data_y = train_df[['outcome']]

svm_test_data_x = test_df[['diff_seed', 'win_rate_post', 'win_rate_home', 'avg_win_score_by',
       'win_rate_regular', 'fgp', 'total_assist_per_fgm',
       'total_block_opp_FGA_percent', 'is_playoff', 'total_steals',
       'total_def_rebounds', 'total_assists', 'total_turnover',
       'total_personalfoul', 'avg_lose_score_by', 'total_off_rebounds',
       'win_rate_away']]
svm_test_data_y = test_df[['outcome']]
```


```python
svm_fs = SVC()
```


```python
svm_fs.fit(svm_train_data_x,svm_train_data_y)
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py:578: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)





    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
svm_fs.score(svm_test_data_x,svm_test_data_y)
```




    0.77611940298507465




```python
svm_fs_df = pd.DataFrame(svm_100.predict(test_data_x_selected_100_svm)[:67]).rename(columns={0:"svm_100"})
```


```python
log_rf_fs_df = pd.DataFrame(LogisticRegression().fit(train_data_x_selected,train_data_y).predict(test_data_x_selected)[:67]).rename(columns={0:"log_rf_fs_df"})
```


```python
rf_80_df = pd.DataFrame(rf_80.predict(test_data_x_selected_80_rf)[:67]).rename(columns={0:"rf_80_fs"})
```


```python
rf_90_df = pd.DataFrame(rf_90.predict(test_data_x_selected_90_rf)[:67]).rename(columns={0:"rf_90_fs"})
```


```python
rf_rfe_df = pd.DataFrame(rf_model.fit(train_data_x_selected_rfe,train_data_y).predict(test_data_x_selected_rfe)[:67]).rename(columns={0:"rf_rfe"})
```


```python
log_15_df = pd.DataFrame(logreg_25.predict(test_data_x_selected_25)[:67]).rename(columns={0:"log_25_fs"})

```


```python
(
    svm_fs_df
    .merge(log_rf_fs_df,how='outer', left_index=True, right_index=True)
    .merge(rf_80_df,how='outer',left_index=True, right_index=True)
    .merge(rf_90_df,how='outer', left_index=True, right_index=True)
    .merge(rf_rfe_df,how='outer', left_index=True, right_index=True)
    .merge(log_15_df,how='outer', left_index=True, right_index=True)
).to_csv("output/final_results_cumulative_year.csv",index=False)
```


```python
seeding_data = pd.read_csv("data/DataFiles/Stage2UpdatedDataFiles/NCAATourneySeeds.csv")
```


```python
seeding_data.query("Season == 2018").to_csv("output/different_teams.csv")
```


```python
seeding_data_teams = pd.read_csv("output/different_teams.csv")
```


```python

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985.0</td>
      <td>1116</td>
      <td>9.0</td>
      <td>1234.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985.0</td>
      <td>1120</td>
      <td>11.0</td>
      <td>1345.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985.0</td>
      <td>1207</td>
      <td>1.0</td>
      <td>1250.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985.0</td>
      <td>1229</td>
      <td>9.0</td>
      <td>1425.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985.0</td>
      <td>1242</td>
      <td>3.0</td>
      <td>1325.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1985.0</td>
      <td>1246</td>
      <td>12.0</td>
      <td>1449.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1985.0</td>
      <td>1256</td>
      <td>5.0</td>
      <td>1338.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1985.0</td>
      <td>1260</td>
      <td>4.0</td>
      <td>1233.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1985.0</td>
      <td>1314</td>
      <td>2.0</td>
      <td>1292.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1985.0</td>
      <td>1323</td>
      <td>7.0</td>
      <td>1333.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1985.0</td>
      <td>1326</td>
      <td>4.0</td>
      <td>1235.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1985.0</td>
      <td>1328</td>
      <td>1.0</td>
      <td>1299.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1985.0</td>
      <td>1374</td>
      <td>5.0</td>
      <td>1330.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1985.0</td>
      <td>1385</td>
      <td>1.0</td>
      <td>1380.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1985.0</td>
      <td>1396</td>
      <td>8.0</td>
      <td>1439.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985.0</td>
      <td>1424</td>
      <td>4.0</td>
      <td>1361.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985.0</td>
      <td>1104</td>
      <td>7.0</td>
      <td>1112.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985.0</td>
      <td>1130</td>
      <td>11.0</td>
      <td>1403.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1985.0</td>
      <td>1181</td>
      <td>3.0</td>
      <td>1337.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1985.0</td>
      <td>1208</td>
      <td>6.0</td>
      <td>1455.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1985.0</td>
      <td>1210</td>
      <td>2.0</td>
      <td>1273.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1985.0</td>
      <td>1228</td>
      <td>3.0</td>
      <td>1318.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1985.0</td>
      <td>1268</td>
      <td>5.0</td>
      <td>1275.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1985.0</td>
      <td>1272</td>
      <td>2.0</td>
      <td>1335.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1985.0</td>
      <td>1276</td>
      <td>1.0</td>
      <td>1192.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1985.0</td>
      <td>1298</td>
      <td>13.0</td>
      <td>1261.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1985.0</td>
      <td>1301</td>
      <td>3.0</td>
      <td>1305.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1985.0</td>
      <td>1393</td>
      <td>7.0</td>
      <td>1177.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1985.0</td>
      <td>1412</td>
      <td>7.0</td>
      <td>1277.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1985.0</td>
      <td>1431</td>
      <td>11.0</td>
      <td>1409.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4542</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1276.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4543</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1211.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4544</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1326.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4545</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1222.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4546</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1401.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4547</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1281.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4548</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1199.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>4549</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1344.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4550</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1361.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4551</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1355.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>4552</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1422.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4553</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1285.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>4554</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1252.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>4555</th>
      <td>2018.0</td>
      <td>1411</td>
      <td>16.0</td>
      <td>1300.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>4556</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4557</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4558</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4559</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4560</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4561</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4562</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4563</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4564</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4565</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4566</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4567</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4568</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4569</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4570</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4571</th>
      <td>NaN</td>
      <td>_</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>6689 rows × 5 columns</p>
</div>




```python
unique_teams = seeding_data_teams.TeamID.unique()
```


```python
seeding_data_2018 = pd.read_csv("output/match_up_2018.csv")

seeding_data_2018.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018.0</td>
      <td>1437</td>
      <td>1.0</td>
      <td>1345.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018.0</td>
      <td>1437</td>
      <td>1.0</td>
      <td>1403.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018.0</td>
      <td>1437</td>
      <td>1.0</td>
      <td>1455.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018.0</td>
      <td>1437</td>
      <td>1.0</td>
      <td>1452.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018.0</td>
      <td>1437</td>
      <td>1.0</td>
      <td>1196.0</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
seeding_data.head()
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
      <th>Seed</th>
      <th>TeamID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>W01</td>
      <td>1207</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>W02</td>
      <td>1210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>W03</td>
      <td>1228</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>W04</td>
      <td>1260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>W05</td>
      <td>1374</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df.head()
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
      <th>outcome</th>
      <th>total_off_rebounds_x</th>
      <th>total_def_rebounds_x</th>
      <th>total_assists_x</th>
      <th>...</th>
      <th>diff_num_season</th>
      <th>diff_is_playoff</th>
      <th>diff_is_champion</th>
      <th>diff_fgp</th>
      <th>diff_total_block_opp_FGA_percent</th>
      <th>diff_win_rate_away</th>
      <th>diff_win_rate_home</th>
      <th>diff_win_rate_neutral</th>
      <th>diff_win_rate_post</th>
      <th>diff_win_rate_regular</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1136</th>
      <td>2003</td>
      <td>1421</td>
      <td>16</td>
      <td>1411</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
      <td>0.037785</td>
      <td>0.035770</td>
      <td>0.031439</td>
      <td>...</td>
      <td>0.152174</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>-0.018262</td>
      <td>0.012232</td>
      <td>-0.071429</td>
      <td>-0.162281</td>
      <td>0.25</td>
      <td>0.500000</td>
      <td>-0.151724</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>2003</td>
      <td>1112</td>
      <td>1</td>
      <td>1436</td>
      <td>16</td>
      <td>-15</td>
      <td>1</td>
      <td>0.047813</td>
      <td>0.043772</td>
      <td>0.045466</td>
      <td>...</td>
      <td>0.003557</td>
      <td>-0.59375</td>
      <td>-0.2</td>
      <td>-0.016969</td>
      <td>-0.011306</td>
      <td>-0.041667</td>
      <td>-0.370833</td>
      <td>0.60</td>
      <td>-0.641509</td>
      <td>-0.237685</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>2003</td>
      <td>1113</td>
      <td>10</td>
      <td>1272</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>0.043744</td>
      <td>0.036084</td>
      <td>0.040266</td>
      <td>...</td>
      <td>0.000000</td>
      <td>-0.09375</td>
      <td>0.0</td>
      <td>0.040251</td>
      <td>-0.011396</td>
      <td>-0.206349</td>
      <td>-0.111111</td>
      <td>-0.50</td>
      <td>-0.397059</td>
      <td>-0.172414</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>2003</td>
      <td>1141</td>
      <td>11</td>
      <td>1166</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>0.030664</td>
      <td>0.036006</td>
      <td>0.040508</td>
      <td>...</td>
      <td>-0.243478</td>
      <td>-0.15625</td>
      <td>0.0</td>
      <td>0.005763</td>
      <td>-0.011456</td>
      <td>-0.100000</td>
      <td>-0.122024</td>
      <td>0.00</td>
      <td>0.250000</td>
      <td>-0.085684</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>2003</td>
      <td>1143</td>
      <td>8</td>
      <td>1301</td>
      <td>9</td>
      <td>-1</td>
      <td>1</td>
      <td>0.033425</td>
      <td>0.038516</td>
      <td>0.041838</td>
      <td>...</td>
      <td>-0.227668</td>
      <td>-0.12500</td>
      <td>0.0</td>
      <td>-0.009399</td>
      <td>0.010209</td>
      <td>-0.375000</td>
      <td>-0.114706</td>
      <td>0.25</td>
      <td>-0.100000</td>
      <td>-0.124138</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>


