
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

## Final data transformation for feature table
- post season is only what we care about
- post season match ups will be what we are joining all the features table to
- additional variable of seeding differential


```python
features_table = (
    features.final_table_processed
    .drop(['total_score','total_rebounds','total_blocks','total_assist_turnover_ratio','expectation_per_game',
           'win_rate','total_rebound_possession_percent','win_rate_overall','total_off_rebounds_percent','total_def_rebounds_percent',
           'total_opponent_score','total_rebound_possessiongain_percent','fg3p'
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
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed
    ))
)

winning_team_perspective_df.head()
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
      <th>0</th>
      <td>1985</td>
      <td>1116</td>
      <td>9</td>
      <td>1234</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1120</td>
      <td>11</td>
      <td>1345</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1207</td>
      <td>1</td>
      <td>1250</td>
      <td>16</td>
      <td>-15</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1229</td>
      <td>9</td>
      <td>1425</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1242</td>
      <td>3</td>
      <td>1325</td>
      <td>14</td>
      <td>-11</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




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
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed
    ))
)
```


```python
prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)

train_df = prediction_df.query("Season >= 2003 & Season <= 2016")
test_df = prediction_df.query("Season == 2017")

train_df.head()

test_df.head()
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
      <th>2057</th>
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
      <td>0.426087</td>
      <td>0.28125</td>
      <td>0.0</td>
      <td>-0.013661</td>
      <td>0.005368</td>
      <td>-0.166667</td>
      <td>0.068111</td>
      <td>0.166667</td>
      <td>0.521739</td>
      <td>0.122933</td>
    </tr>
    <tr>
      <th>2058</th>
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
      <td>0.000000</td>
      <td>0.03125</td>
      <td>0.0</td>
      <td>-0.017967</td>
      <td>0.017078</td>
      <td>-0.033333</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.333333</td>
      <td>-0.076840</td>
    </tr>
    <tr>
      <th>2059</th>
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
      <td>0.213043</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>-0.022078</td>
      <td>0.006311</td>
      <td>0.153846</td>
      <td>-0.071429</td>
      <td>-0.333333</td>
      <td>0.600000</td>
      <td>-0.141724</td>
    </tr>
    <tr>
      <th>2060</th>
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
      <td>-0.152174</td>
      <td>-0.03125</td>
      <td>0.0</td>
      <td>0.007043</td>
      <td>0.032465</td>
      <td>0.095238</td>
      <td>0.055024</td>
      <td>0.466667</td>
      <td>0.371429</td>
      <td>0.017007</td>
    </tr>
    <tr>
      <th>2061</th>
      <td>2017</td>
      <td>1112</td>
      <td>2</td>
      <td>1315</td>
      <td>15</td>
      <td>-13</td>
      <td>1</td>
      <td>0.553377</td>
      <td>0.843403</td>
      <td>0.593909</td>
      <td>...</td>
      <td>-0.121739</td>
      <td>-0.28125</td>
      <td>0.0</td>
      <td>0.000339</td>
      <td>-0.008089</td>
      <td>-0.122222</td>
      <td>-0.366071</td>
      <td>0.050000</td>
      <td>-0.655172</td>
      <td>-0.126900</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




```python
train_data_x = train_df[['diff_seed','diff_total_off_rebounds','diff_total_def_rebounds','diff_total_assists',
                         'diff_total_steals','diff_total_turnover','diff_total_personalfoul',
                         'diff_total_assist_per_fgm','diff_avg_lose_score_by',
                         'diff_avg_win_score_by','diff_num_season','diff_is_playoff','diff_is_champion',
                         'diff_fgp','diff_total_block_opp_FGA_percent','diff_win_rate_away','diff_win_rate_home',
                         'diff_win_rate_neutral','diff_win_rate_post','diff_win_rate_regular']]
train_data_y = train_df['outcome']

test_data_x = test_df[['diff_seed','diff_total_off_rebounds','diff_total_def_rebounds','diff_total_assists',
                       'diff_total_steals','diff_total_turnover','diff_total_personalfoul',
                       'diff_total_assist_per_fgm','diff_avg_lose_score_by',
                       'diff_avg_win_score_by','diff_num_season','diff_is_playoff','diff_is_champion',
                       'diff_fgp','diff_total_block_opp_FGA_percent','diff_win_rate_away','diff_win_rate_home',
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
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.7761194029850746
    Counter({1: 52, 0: 15})
    
    Which percentile : 15
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 20
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 25
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 30
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 35
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 40
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 45
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 50
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 55
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 60
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 65
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 70
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 75
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 80
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 85
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 90
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 95
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 100
    normal logreg: 0.8507462686567164
    feature selection logreg: 0.8507462686567164
    Counter({1: 57, 0: 10})



```python
# based on the output of the univariate, we can narrow to 10, 25, 80
select_90 = SelectPercentile(percentile=90)
select_85 = SelectPercentile(percentile=85)
```


```python
select_90.fit(train_data_x, train_data_y)

train_data_x_selected_90 = select_90.transform(train_data_x)
test_data_x_selected_90 = select_90.transform(test_data_x)

mask = select_90.get_support()    
#     print(mask)
logreg_90 = LogisticRegression()
logreg_90.fit(train_data_x_selected_90,train_data_y)

logreg_90.score(test_data_x_selected_90,test_data_y)

logreg_90.predict(test_data_x_selected_90)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
select_85.fit(train_data_x, train_data_y)

train_data_x_selected_85 = select_85.transform(train_data_x)
test_data_x_selected_85 = select_85.transform(test_data_x)

mask = select_85.get_support()    
#     print(mask)
logreg_85 = LogisticRegression()
logreg_85.fit(train_data_x_selected_85,train_data_y)

logreg_85.score(test_data_x_selected_85,test_data_y)

logreg_85.predict(test_data_x_selected_85)[:67]
```




    array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
## selectfrommodel RF
select_rf = SelectFromModel(RandomForestClassifier(n_estimators =100, max_depth = 10, random_state=0),threshold=0.04)
```


```python
select_rf.fit(train_data_x,train_data_y)
```




    SelectFromModel(estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False),
            norm_order=1, prefit=False, threshold=0.04)




```python
train_data_x_selected = select_rf.transform(train_data_x)
test_data_x_selected = select_rf.transform(test_data_x)
```


```python
LogisticRegression().fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)
```




    0.83582089552238803




```python
## selectfrommodel lassoCV --> same as univariate
```


```python
select_lcv = SelectFromModel(LassoCV(max_iter=100,n_alphas=10,eps=1e-05),threshold=0.01)

select_lcv.fit(train_data_x,train_data_y)
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)





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




    0.85074626865671643




```python
# select from model lassocv
lassocv = LassoCV(random_state=0)
param_grid_lasso = {
    'n_alphas': [1,5,10,25,50,100,150,200,500,1000],
    'max_iter': [100,500,1000,1500,2000,3000],
    'eps': [0.00001,0.0001,0.001,0.01]
}
grid_lcv = GridSearchCV(lassocv, param_grid_lasso, cv=5, verbose=2)
grid_lcv.fit(train_data_x_selected, train_data_y)

lcv_model = grid_lcv.best_estimator_

```

    Fitting 5 folds for each of 240 candidates, totalling 1200 fits
    [CV] eps=1e-05, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=25 ............................


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s


    [CV] ................... eps=1e-05, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=100, n_alphas=1000 -   0.3s
    [CV] eps=1e-05, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=1e-05, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=1e-05, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=1e-05, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=1e-05, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=2000, n_alphas=1000 -   0.3s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=1e-05, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=1e-05, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=200 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=1e-05, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=1e-05, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=3000, n_alphas=1000 -   0.3s
    [CV] eps=1e-05, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=1e-05, max_iter=3000, n_alphas=1000 -   0.3s
    [CV] eps=0.0001, max_iter=100, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=200 -   0.1s
    [CV] eps=0.0001, max_iter=100, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=100, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=100, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=100, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=100, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=100, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=100, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=100, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=100, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=100, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=100, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=500, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=1 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=5 ............................
    [CV] ................... eps=0.0001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=10 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=25 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=50 ...........................
    [CV] .................. eps=0.0001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=100 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=150 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=200 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=200 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=200 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=500, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=500 ..........................
    [CV] ................. eps=0.0001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=500, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=500, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=500, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=500, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=500, n_alphas=1000 .........................
    [CV] ................ eps=0.0001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=1500, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1500, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1500, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1500, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1500, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=1500, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=2000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=2000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=2000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=2000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=2000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=2000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=5 ...........................
    [CV] .................. eps=0.0001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=10 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=25 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=50 ..........................
    [CV] ................. eps=0.0001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=100 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=150 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=200 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.0001, max_iter=3000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=3000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=3000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=3000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=3000, n_alphas=500 .........................
    [CV] ................ eps=0.0001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.0001, max_iter=3000, n_alphas=1000 ........................
    [CV] ............... eps=0.0001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=100, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=100, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=1 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=5 .............................
    [CV] .................... eps=0.001, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=10 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=25 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=50 ............................
    [CV] ................... eps=0.001, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=100 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=150 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=200 -   0.1s
    [CV] eps=0.001, max_iter=500, n_alphas=200 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=500, n_alphas=500 ...........................
    [CV] .................. eps=0.001, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=500, n_alphas=1000 ..........................
    [CV] ................. eps=0.001, max_iter=500, n_alphas=1000 -   0.3s
    [CV] eps=0.001, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=200 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=200 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=200 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=200 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=200 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1500, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=1500, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=2000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=2000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=1 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=5 ............................
    [CV] ................... eps=0.001, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=10 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=25 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=50 ...........................
    [CV] .................. eps=0.001, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=100 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=150 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=200 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.001, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=3000, n_alphas=500 ..........................
    [CV] ................. eps=0.001, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.001, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.001, max_iter=3000, n_alphas=1000 .........................
    [CV] ................ eps=0.001, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=100, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=100, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=100, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=100, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=100, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=100, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=100, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=100, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=100, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=100, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=100, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=100, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=100, n_alphas=1000 -   0.3s
    [CV] eps=0.01, max_iter=100, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=100, n_alphas=1000 -   0.3s
    [CV] eps=0.01, max_iter=100, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=100, n_alphas=1000 -   0.3s
    [CV] eps=0.01, max_iter=100, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=100, n_alphas=1000 -   0.3s
    [CV] eps=0.01, max_iter=500, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=1 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=5 ..............................
    [CV] ..................... eps=0.01, max_iter=500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=10 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=25 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=50 .............................
    [CV] .................... eps=0.01, max_iter=500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=100 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=150 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=200 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=500, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=500, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=500, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=500, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=500, n_alphas=500 ............................
    [CV] ................... eps=0.01, max_iter=500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=500, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=500, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=500, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=500, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=500, n_alphas=1000 ...........................
    [CV] .................. eps=0.01, max_iter=500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1500, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=1500, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=1500, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=1500, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=200 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=200 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=1500, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=1500, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1500, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1500, n_alphas=1000 -   0.3s
    [CV] eps=0.01, max_iter=1500, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1500, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1500, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=1500, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=1500, n_alphas=1000 -   0.3s
    [CV] eps=0.01, max_iter=2000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=2000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=2000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=2000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=2000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=2000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=2000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=2000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=2000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=2000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=2000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=2000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=2000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=2000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=2000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=3000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=1 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=1 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=5 .............................
    [CV] .................... eps=0.01, max_iter=3000, n_alphas=5 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=10 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=10 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=25 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=25 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=50 ............................
    [CV] ................... eps=0.01, max_iter=3000, n_alphas=50 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=100 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=100 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=150 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=150 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=200 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=200 -   0.0s
    [CV] eps=0.01, max_iter=3000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=3000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=3000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=3000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=3000, n_alphas=500 ...........................
    [CV] .................. eps=0.01, max_iter=3000, n_alphas=500 -   0.1s
    [CV] eps=0.01, max_iter=3000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=3000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=3000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=3000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=3000, n_alphas=1000 -   0.2s
    [CV] eps=0.01, max_iter=3000, n_alphas=1000 ..........................
    [CV] ................. eps=0.01, max_iter=3000, n_alphas=1000 -   0.2s


    [Parallel(n_jobs=1)]: Done 1200 out of 1200 | elapsed:   54.2s finished



```python
lcv_model
```




    LassoCV(alphas=None, copy_X=True, cv=None, eps=1e-05, fit_intercept=True,
        max_iter=100, n_alphas=10, n_jobs=1, normalize=False, positive=False,
        precompute='auto', random_state=0, selection='cyclic', tol=0.0001,
        verbose=False)




```python
select = SelectFromModel(LassoCV(max_iter=100,n_alphas=1,eps=0.001),threshold=0.04)

select.fit(train_data_x,train_data_y)
```




    SelectFromModel(estimator=LassoCV(alphas=None, copy_X=True, cv=None, eps=0.001, fit_intercept=True,
        max_iter=100, n_alphas=1, n_jobs=1, normalize=False, positive=False,
        precompute='auto', random_state=None, selection='cyclic', tol=0.0001,
        verbose=False),
            norm_order=1, prefit=False, threshold=0.04)




```python
train_data_x_selected = select.transform(train_data_x)
train_data_x_selected.shape
```




    (1835, 4)




```python
test_data_x_selected = select.transform(test_data_x)
```


```python
LogisticRegression().fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)
```




    0.71641791044776115




```python
rf = RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [5,10,50,100,150,500,1000],
    'max_depth': [1,2,5,10,15,50,100]
}
grid_rf = GridSearchCV(rf, param_grid, scoring='accuracy', cv=5, verbose=0)
grid_rf.fit(train_data_x_selected, train_data_y)

rf_model = grid_rf.best_estimator_
rf_model
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=10, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)



## Feature selection for RF
- univariate
- SelectFromModel SVM
- RFE(CV)


```python
model = LogisticRegression()
rfe = RFE(model, 9)
fit = rfe.fit(train_data_x, train_data_y)
print("Num Features: "+ str(fit.n_features_))
print("Selected Features: " + str(fit.support_))
print("Feature Ranking: " + str(fit.ranking_))
```

    Num Features: 9
    Selected Features: [False False False False  True  True  True  True False  True False False
      True False False  True False False  True  True]
    Feature Ranking: [ 4  8 10  6  1  1  1  1 12  1  5  3  1 11  9  1  7  2  1  1]



```python
train_data_x_selected = fit.transform(train_data_x)
test_data_x_selected = fit.transform(test_data_x)
```


```python
model.fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)
```




    0.70149253731343286




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
    rf = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0)
    rf.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal rf: {}".format(rf.score(test_data_x,test_data_y)))
    
    rf.fit(train_data_x_selected,train_data_y)
    print("feature selection rf: {}".format(rf.score(test_data_x_selected,test_data_y)))
    print(Counter(rf.predict(test_data_x_selected)[:67]))
    

```

    
    Which percentile : 10
    normal rf: 0.835820895522388
    feature selection rf: 0.753731343283582
    Counter({1: 51, 0: 16})
    
    Which percentile : 15
    normal rf: 0.835820895522388
    feature selection rf: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 20
    normal rf: 0.835820895522388
    feature selection rf: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 25
    normal rf: 0.835820895522388
    feature selection rf: 0.7910447761194029
    Counter({1: 53, 0: 14})
    
    Which percentile : 30
    normal rf: 0.835820895522388
    feature selection rf: 0.7910447761194029
    Counter({1: 53, 0: 14})
    
    Which percentile : 35
    normal rf: 0.835820895522388
    feature selection rf: 0.7910447761194029
    Counter({1: 53, 0: 14})
    
    Which percentile : 40
    normal rf: 0.835820895522388
    feature selection rf: 0.7910447761194029
    Counter({1: 53, 0: 14})
    
    Which percentile : 45
    normal rf: 0.835820895522388
    feature selection rf: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 50
    normal rf: 0.835820895522388
    feature selection rf: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 55
    normal rf: 0.835820895522388
    feature selection rf: 0.7910447761194029
    Counter({1: 53, 0: 14})
    
    Which percentile : 60
    normal rf: 0.835820895522388
    feature selection rf: 0.8134328358208955
    Counter({1: 54, 0: 13})
    
    Which percentile : 65
    normal rf: 0.835820895522388
    feature selection rf: 0.8059701492537313
    Counter({1: 54, 0: 13})
    
    Which percentile : 70
    normal rf: 0.835820895522388
    feature selection rf: 0.835820895522388
    Counter({1: 55, 0: 12})
    
    Which percentile : 75
    normal rf: 0.835820895522388
    feature selection rf: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 80
    normal rf: 0.835820895522388
    feature selection rf: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 85
    normal rf: 0.835820895522388
    feature selection rf: 0.8134328358208955
    Counter({1: 54, 0: 13})
    
    Which percentile : 90
    normal rf: 0.835820895522388
    feature selection rf: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 95
    normal rf: 0.835820895522388
    feature selection rf: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 100
    normal rf: 0.835820895522388
    feature selection rf: 0.835820895522388
    Counter({1: 56, 0: 11})



```python
# based on the output of the univariate, we can narrow to 60, 80
select_70_rf = SelectPercentile(percentile=70)
select_80_rf = SelectPercentile(percentile=80)
```


```python
select_70_rf.fit(train_data_x, train_data_y)

train_data_x_selected_70_rf = select_70_rf.transform(train_data_x)
test_data_x_selected_70_rf = select_70_rf.transform(test_data_x)

mask = select_70_rf.get_support()        
# print(mask)
rf_70 = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0,warm_start=True)
rf_70.fit(train_data_x_selected_70_rf,train_data_y)

rf_70.score(test_data_x_selected_70_rf,test_data_y)
```




    0.83582089552238803




```python
rf_70.predict(test_data_x_selected_70_rf)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])




```python
select_80_rf.fit(train_data_x, train_data_y)

train_data_x_selected_80_rf = select_80_rf.transform(train_data_x)
test_data_x_selected_80_rf = select_80_rf.transform(test_data_x)

mask = select_80_rf.get_support()        
# print(mask)
rf_80 = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0,warm_start=True)
rf_80.fit(train_data_x_selected_80_rf,train_data_y)

rf_80.score(test_data_x_selected_80_rf,test_data_y)
```




    0.83582089552238803




```python
rf_80.predict(test_data_x_selected_80_rf)[:67]
```




    array([0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])




```python
Counter(rf_80.predict(test_data_x_selected_80_rf)[:67])
```




    Counter({0: 11, 1: 56})




```python
Counter(rf_70.predict(test_data_x_selected_70_rf)[:67])
```




    Counter({0: 12, 1: 55})




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
```

    Num Features: 8
    Selected Features: [False False  True  True False False  True  True False False False  True
     False False False  True  True False  True False]
    Feature Ranking: [13  5  1  1  6  4  1  1  3  2 11  1 12  9 10  1  1  8  1  7]



```python
rf_model = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0,warm_start=False)
```


```python
rf_model.fit(train_data_x_selected_rfe,train_data_y).score(test_data_x_selected_rfe,test_data_y)
```




    0.79850746268656714



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
    normal svm: 0.9104477611940298
    feature selection svm: 0.7761194029850746
    Counter({1: 52, 0: 15})
    
    Which percentile : 15
    normal svm: 0.9104477611940298
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 20
    normal svm: 0.9104477611940298
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 25
    normal svm: 0.9104477611940298
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 30
    normal svm: 0.9104477611940298
    feature selection svm: 0.835820895522388
    Counter({1: 56, 0: 11})
    
    Which percentile : 35
    normal svm: 0.9104477611940298
    feature selection svm: 0.8283582089552238
    Counter({1: 56, 0: 11})
    
    Which percentile : 40
    normal svm: 0.9104477611940298
    feature selection svm: 0.8208955223880597
    Counter({1: 55, 0: 12})
    
    Which percentile : 45
    normal svm: 0.9104477611940298
    feature selection svm: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 50
    normal svm: 0.9104477611940298
    feature selection svm: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 55
    normal svm: 0.9104477611940298
    feature selection svm: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 60
    normal svm: 0.9104477611940298
    feature selection svm: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 65
    normal svm: 0.9104477611940298
    feature selection svm: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 70
    normal svm: 0.9104477611940298
    feature selection svm: 0.8507462686567164
    Counter({1: 57, 0: 10})
    
    Which percentile : 75
    normal svm: 0.9104477611940298
    feature selection svm: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 80
    normal svm: 0.9104477611940298
    feature selection svm: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 85
    normal svm: 0.9104477611940298
    feature selection svm: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 90
    normal svm: 0.9104477611940298
    feature selection svm: 0.8656716417910447
    Counter({1: 58, 0: 9})
    
    Which percentile : 95
    normal svm: 0.9104477611940298
    feature selection svm: 0.8805970149253731
    Counter({1: 59, 0: 8})
    
    Which percentile : 100
    normal svm: 0.9104477611940298
    feature selection svm: 0.9104477611940298
    Counter({1: 61, 0: 6})



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




    array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])




```python
rf_fi = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0)
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




    array(['diff_win_rate_home', 'diff_is_playoff', 'diff_win_rate_post',
           'diff_avg_win_score_by', 'diff_total_def_rebounds', 'diff_seed',
           'diff_num_season', 'diff_win_rate_away', 'diff_win_rate_regular',
           'diff_total_assists', 'diff_avg_lose_score_by', 'diff_fgp',
           'diff_total_off_rebounds', 'diff_total_block_opp_FGA_percent',
           'diff_total_steals', 'diff_total_turnover',
           'diff_total_personalfoul'], dtype=object)




```python
svm_train_data_x = train_df[['diff_win_rate_home', 'diff_is_playoff', 'diff_win_rate_post',
       'diff_avg_win_score_by', 'diff_total_def_rebounds', 'diff_seed',
       'diff_num_season', 'diff_win_rate_away', 'diff_win_rate_regular',
       'diff_total_assists', 'diff_avg_lose_score_by', 'diff_fgp',
       'diff_total_off_rebounds', 'diff_total_block_opp_FGA_percent',
       'diff_total_steals', 'diff_total_turnover',
       'diff_total_personalfoul']]
svm_train_data_y = train_df[['outcome']]

svm_test_data_x = test_df[['diff_win_rate_home', 'diff_is_playoff', 'diff_win_rate_post',
       'diff_avg_win_score_by', 'diff_total_def_rebounds', 'diff_seed',
       'diff_num_season', 'diff_win_rate_away', 'diff_win_rate_regular',
       'diff_total_assists', 'diff_avg_lose_score_by', 'diff_fgp',
       'diff_total_off_rebounds', 'diff_total_block_opp_FGA_percent',
       'diff_total_steals', 'diff_total_turnover',
       'diff_total_personalfoul']]
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




    0.89552238805970152




```python
svm_fs_df = pd.DataFrame(svm_fs.predict(svm_test_data_x)[:67]).rename(columns={0:"svm_fs_df"})
```


```python
log_rf_fs_df = pd.DataFrame(LogisticRegression().fit(train_data_x_selected,train_data_y).predict(test_data_x_selected)[:67]).rename(columns={0:"log_rf_fs_df"})
```


```python
rf_70_df = pd.DataFrame(rf_70.predict(test_data_x_selected_70_rf)[:67]).rename(columns={0:"rf_70_fs"})
```


```python
rf_80_df = pd.DataFrame(rf_80.predict(test_data_x_selected_80_rf)[:67]).rename(columns={0:"rf_80_fs"})
```


```python
rf_rfe_df = pd.DataFrame(rf_model.fit(train_data_x_selected_rfe,train_data_y).predict(test_data_x_selected_rfe)[:67]).rename(columns={0:"rf_rfe"})
```


```python
log_85_df = pd.DataFrame(logreg_85.predict(test_data_x_selected_85)[:67]).rename(columns={0:"log_85_fs"})

```


```python
log_90_df = pd.DataFrame(logreg_90.predict(test_data_x_selected_90)[:67]).rename(columns={0:"log_90_fs"})
```


```python
(
    svm_fs_df
    .merge(log_rf_fs_df,how='outer', left_index=True, right_index=True)
    .merge(rf_70_df,how='outer',left_index=True, right_index=True)
    .merge(rf_80_df,how='outer', left_index=True, right_index=True)
    .merge(rf_rfe_df,how='outer', left_index=True, right_index=True)
    .merge(log_85_df,how='outer', left_index=True, right_index=True)
    .merge(log_90_df,how='outer', left_index=True, right_index=True)
).to_csv("output/final_results_static_year_improved.csv",index=False)
```
