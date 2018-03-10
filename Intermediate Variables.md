
___
This notebook decides on the intermediate variables being used


```python
import pandas as pd
import numpy as np
import scipy
from sklearn import *
```

    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)
    /Users/shumingpeh/anaconda3/lib/python3.6/site-packages/sklearn/learning_curve.py:22: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the functions are moved. This module will be removed in 0.20
      DeprecationWarning)


## Read data
- regularseason detailed results
- ~~cities~~
- teams
- coaches
    - there is a problem if a coach is new, so to prevent this from happening
    - coach will have a proxy variables of
        1. number of years of experience up to that year
        1. number of championship
        1. number of playoffs made


```python
raw_data_regularseason = pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv")
```


```python
raw_data_teams = pd.read_csv("data/DataFiles/Teams.csv")
```


```python
raw_data_coaches = pd.read_csv("data/DataFiles/TeamCoaches.csv")
```


```python
raw_data_teams_coaches = (
    raw_data_teams
    .merge(raw_data_coaches, how='left', on=['TeamID'])
)
```


```python
raw_data_regularseason.head()
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
      <th>DayNum</th>
      <th>WTeamID</th>
      <th>WScore</th>
      <th>LTeamID</th>
      <th>LScore</th>
      <th>WLoc</th>
      <th>NumOT</th>
      <th>WFGM</th>
      <th>WFGA</th>
      <th>...</th>
      <th>LFGA3</th>
      <th>LFTM</th>
      <th>LFTA</th>
      <th>LOR</th>
      <th>LDR</th>
      <th>LAst</th>
      <th>LTO</th>
      <th>LStl</th>
      <th>LBlk</th>
      <th>LPF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>10</td>
      <td>1104</td>
      <td>68</td>
      <td>1328</td>
      <td>62</td>
      <td>N</td>
      <td>0</td>
      <td>27</td>
      <td>58</td>
      <td>...</td>
      <td>10</td>
      <td>16</td>
      <td>22</td>
      <td>10</td>
      <td>22</td>
      <td>8</td>
      <td>18</td>
      <td>9</td>
      <td>2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>10</td>
      <td>1272</td>
      <td>70</td>
      <td>1393</td>
      <td>63</td>
      <td>N</td>
      <td>0</td>
      <td>26</td>
      <td>62</td>
      <td>...</td>
      <td>24</td>
      <td>9</td>
      <td>20</td>
      <td>20</td>
      <td>25</td>
      <td>7</td>
      <td>12</td>
      <td>8</td>
      <td>6</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>11</td>
      <td>1266</td>
      <td>73</td>
      <td>1437</td>
      <td>61</td>
      <td>N</td>
      <td>0</td>
      <td>24</td>
      <td>58</td>
      <td>...</td>
      <td>26</td>
      <td>14</td>
      <td>23</td>
      <td>31</td>
      <td>22</td>
      <td>9</td>
      <td>12</td>
      <td>2</td>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>11</td>
      <td>1296</td>
      <td>56</td>
      <td>1457</td>
      <td>50</td>
      <td>N</td>
      <td>0</td>
      <td>18</td>
      <td>38</td>
      <td>...</td>
      <td>22</td>
      <td>8</td>
      <td>15</td>
      <td>17</td>
      <td>20</td>
      <td>9</td>
      <td>19</td>
      <td>4</td>
      <td>3</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>11</td>
      <td>1400</td>
      <td>77</td>
      <td>1208</td>
      <td>71</td>
      <td>N</td>
      <td>0</td>
      <td>30</td>
      <td>61</td>
      <td>...</td>
      <td>16</td>
      <td>17</td>
      <td>27</td>
      <td>21</td>
      <td>15</td>
      <td>12</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
raw_data_regularseason.dtypes
```




    Season      int64
    DayNum      int64
    WTeamID     int64
    WScore      int64
    LTeamID     int64
    LScore      int64
    WLoc       object
    NumOT       int64
    WFGM        int64
    WFGA        int64
    WFGM3       int64
    WFGA3       int64
    WFTM        int64
    WFTA        int64
    WOR         int64
    WDR         int64
    WAst        int64
    WTO         int64
    WStl        int64
    WBlk        int64
    WPF         int64
    LFGM        int64
    LFGA        int64
    LFGM3       int64
    LFGA3       int64
    LFTM        int64
    LFTA        int64
    LOR         int64
    LDR         int64
    LAst        int64
    LTO         int64
    LStl        int64
    LBlk        int64
    LPF         int64
    dtype: object



## Features to be included
- Season year
- winning/losing teamid
- winning/losing score
- winning/losing field goal percentage
- winning/losing field goal 3 point percentage
- winning/losing free throw percentage
- overall win rate


```python
winning_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WScore":"sum","WFGM":"sum","WFGA":"sum","WFGM3":"sum","WFGA3":"sum","WFTM":"sum","WFTA":"sum","LScore":"sum","winning_num_counts":"sum"})
    .reset_index()
    .rename(columns={"LScore":"losing_opponent_score"})
)
```


```python
winning_teams_score_up_to_2013.head()
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
      <th>WScore</th>
      <th>WFGM</th>
      <th>WFGA</th>
      <th>WFGM3</th>
      <th>WFGA3</th>
      <th>WFTM</th>
      <th>WFTA</th>
      <th>losing_opponent_score</th>
      <th>winning_num_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>825</td>
      <td>271</td>
      <td>480</td>
      <td>120</td>
      <td>259</td>
      <td>163</td>
      <td>249</td>
      <td>638</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>1141</td>
      <td>390</td>
      <td>720</td>
      <td>71</td>
      <td>187</td>
      <td>290</td>
      <td>402</td>
      <td>1019</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>1270</td>
      <td>439</td>
      <td>992</td>
      <td>120</td>
      <td>354</td>
      <td>272</td>
      <td>383</td>
      <td>1046</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>556</td>
      <td>179</td>
      <td>433</td>
      <td>64</td>
      <td>157</td>
      <td>134</td>
      <td>180</td>
      <td>465</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>888</td>
      <td>322</td>
      <td>700</td>
      <td>76</td>
      <td>207</td>
      <td>168</td>
      <td>270</td>
      <td>753</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
losing_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"WScore":"sum","LScore":"sum","LFGM":"sum","LFGA":"sum","LFGM3":"sum","LFGA3":"sum","LFTM":"sum","LFTA":"sum","losing_num_counts":"sum"})
    .reset_index()
    .rename(columns={"WScore":"winning_opponent_score"})
)
```


```python
losing_teams_score_up_to_2013.head()
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
      <th>LTeamID</th>
      <th>winning_opponent_score</th>
      <th>LScore</th>
      <th>LFGM</th>
      <th>LFGA</th>
      <th>LFGM3</th>
      <th>LFGA3</th>
      <th>LFTM</th>
      <th>LFTA</th>
      <th>losing_num_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>958</td>
      <td>778</td>
      <td>265</td>
      <td>634</td>
      <td>99</td>
      <td>324</td>
      <td>149</td>
      <td>230</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>1091</td>
      <td>986</td>
      <td>343</td>
      <td>788</td>
      <td>76</td>
      <td>247</td>
      <td>224</td>
      <td>296</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>774</td>
      <td>670</td>
      <td>234</td>
      <td>609</td>
      <td>58</td>
      <td>202</td>
      <td>144</td>
      <td>203</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>1528</td>
      <td>1310</td>
      <td>455</td>
      <td>1169</td>
      <td>133</td>
      <td>383</td>
      <td>267</td>
      <td>388</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>1032</td>
      <td>893</td>
      <td>334</td>
      <td>848</td>
      <td>95</td>
      <td>287</td>
      <td>130</td>
      <td>191</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine_winning_losing_stats_for_year = (
    winning_teams_score_up_to_2013
    .merge(losing_teams_score_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    .pipe(lambda x:x.assign(total_score = x.WScore + x.LScore))
    .pipe(lambda x:x.assign(total_opponent_score = x.winning_opponent_score + x.losing_opponent_score))
    .pipe(lambda x:x.assign(total_fgm = x.WFGM + x.LFGM))
    .pipe(lambda x:x.assign(total_fga = x.WFGA + x.LFGA))
    .pipe(lambda x:x.assign(total_fg3m = x.WFGM3 + x.LFGM3))
    .pipe(lambda x:x.assign(total_fg3a = x.WFGA3 + x.LFGA3))
    .pipe(lambda x:x.assign(total_ftm = x.WFTM + x.LFTM))
    .pipe(lambda x:x.assign(total_fta = x.WFTA + x.LFTA))
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .sort_values(['WTeamID','Season'])
)
```


```python
combine_winning_losing_stats_for_year.head()
combine_winning_losing_stats_for_year.dtypes
```




    Season                      int64
    WTeamID                     int64
    WScore                      int64
    WFGM                        int64
    WFGA                        int64
    WFGM3                       int64
    WFGA3                       int64
    WFTM                        int64
    WFTA                        int64
    losing_opponent_score       int64
    winning_num_counts          int64
    LTeamID                     int64
    winning_opponent_score      int64
    LScore                      int64
    LFGM                        int64
    LFGA                        int64
    LFGM3                       int64
    LFGA3                       int64
    LFTM                        int64
    LFTA                        int64
    losing_num_counts           int64
    total_score                 int64
    total_opponent_score        int64
    total_fgm                   int64
    total_fga                   int64
    total_fg3m                  int64
    total_fg3a                  int64
    total_ftm                   int64
    total_fta                   int64
    win_rate                  float64
    dtype: object




```python
cumulative_stats_for_team_each_year = (
    combine_winning_losing_stats_for_year
    .sort_values(['WTeamID','Season'])
    .groupby(['WTeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(Season = combine_winning_losing_stats_for_year.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_winning_losing_stats_for_year.WTeamID.values))
    .drop(['LTeamID','win_rate'],1)
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .pipe(lambda x:x.assign(WFGP = x.WFGM/x.WFGA))
    .pipe(lambda x:x.assign(WFG3P = x.WFGM3/x.WFGA3))
    .pipe(lambda x:x.assign(WFTP = x.WFTM/x.WFTA))
    .pipe(lambda x:x.assign(LFGP = x.LFGM/x.LFGA))
    .pipe(lambda x:x.assign(LFG3P = x.LFGM3/x.LFGA3))
    .pipe(lambda x:x.assign(LFTP = x.LFTM/x.LFTA))
    .pipe(lambda x:x.assign(fgp = x.total_fgm/x.total_fga))
    .pipe(lambda x:x.assign(fg3p = x.total_fg3m/x.total_fg3a))
    .pipe(lambda x:x.assign(ftp = x.total_ftm/x.total_fta))
)
```


```python
cumulative_stats_for_team_each_year.head()
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
      <th>WScore</th>
      <th>WFGM</th>
      <th>WFGA</th>
      <th>WFGM3</th>
      <th>WFGA3</th>
      <th>WFTM</th>
      <th>WFTA</th>
      <th>losing_opponent_score</th>
      <th>winning_num_counts</th>
      <th>...</th>
      <th>win_rate</th>
      <th>WFGP</th>
      <th>WFG3P</th>
      <th>WFTP</th>
      <th>LFGP</th>
      <th>LFG3P</th>
      <th>LFTP</th>
      <th>fgp</th>
      <th>fg3p</th>
      <th>ftp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>825</td>
      <td>271</td>
      <td>480</td>
      <td>120</td>
      <td>259</td>
      <td>163</td>
      <td>249</td>
      <td>638</td>
      <td>12</td>
      <td>...</td>
      <td>0.428571</td>
      <td>0.564583</td>
      <td>0.463320</td>
      <td>0.654618</td>
      <td>0.417981</td>
      <td>0.305556</td>
      <td>0.647826</td>
      <td>0.481149</td>
      <td>0.375643</td>
      <td>0.651357</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2004</td>
      <td>2229</td>
      <td>737</td>
      <td>1393</td>
      <td>312</td>
      <td>734</td>
      <td>443</td>
      <td>636</td>
      <td>1709</td>
      <td>34</td>
      <td>...</td>
      <td>0.607143</td>
      <td>0.529074</td>
      <td>0.425068</td>
      <td>0.696541</td>
      <td>0.408686</td>
      <td>0.305732</td>
      <td>0.639576</td>
      <td>0.481886</td>
      <td>0.378423</td>
      <td>0.678999</td>
    </tr>
    <tr>
      <th>653</th>
      <td>2005</td>
      <td>3326</td>
      <td>1115</td>
      <td>2180</td>
      <td>458</td>
      <td>1110</td>
      <td>638</td>
      <td>902</td>
      <td>2533</td>
      <td>51</td>
      <td>...</td>
      <td>0.600000</td>
      <td>0.511468</td>
      <td>0.412613</td>
      <td>0.707317</td>
      <td>0.408027</td>
      <td>0.318804</td>
      <td>0.661616</td>
      <td>0.469388</td>
      <td>0.373236</td>
      <td>0.693374</td>
    </tr>
    <tr>
      <th>982</th>
      <td>2006</td>
      <td>4756</td>
      <td>1604</td>
      <td>3171</td>
      <td>659</td>
      <td>1588</td>
      <td>889</td>
      <td>1234</td>
      <td>3676</td>
      <td>73</td>
      <td>...</td>
      <td>0.646018</td>
      <td>0.505834</td>
      <td>0.414987</td>
      <td>0.720421</td>
      <td>0.412921</td>
      <td>0.317597</td>
      <td>0.669456</td>
      <td>0.472430</td>
      <td>0.378968</td>
      <td>0.706192</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2007</td>
      <td>6347</td>
      <td>2135</td>
      <td>4205</td>
      <td>870</td>
      <td>2061</td>
      <td>1207</td>
      <td>1652</td>
      <td>4844</td>
      <td>95</td>
      <td>...</td>
      <td>0.664336</td>
      <td>0.507729</td>
      <td>0.422125</td>
      <td>0.730630</td>
      <td>0.412256</td>
      <td>0.315093</td>
      <td>0.688119</td>
      <td>0.475389</td>
      <td>0.384158</td>
      <td>0.719221</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
cumulative_stats_for_team_each_year.dtypes
```




    Season                      int64
    WScore                      int64
    WFGM                        int64
    WFGA                        int64
    WFGM3                       int64
    WFGA3                       int64
    WFTM                        int64
    WFTA                        int64
    losing_opponent_score       int64
    winning_num_counts          int64
    winning_opponent_score      int64
    LScore                      int64
    LFGM                        int64
    LFGA                        int64
    LFGM3                       int64
    LFGA3                       int64
    LFTM                        int64
    LFTA                        int64
    losing_num_counts           int64
    total_score                 int64
    total_opponent_score        int64
    total_fgm                   int64
    total_fga                   int64
    total_fg3m                  int64
    total_fg3a                  int64
    total_ftm                   int64
    total_fta                   int64
    TeamID                      int64
    win_rate                  float64
    WFGP                      float64
    WFG3P                     float64
    WFTP                      float64
    LFGP                      float64
    LFG3P                     float64
    LFTP                      float64
    fgp                       float64
    fg3p                      float64
    ftp                       float64
    dtype: object



## Some variations to try for features
- separate winning and losing
    - reconcilation of winning and losing will have to be done later
    - could be diff between percentage --> this might give an insight of when they are losing/winning?

## Intermediate Variables
- Coach stats
    - number of years till that season
    - number of championship till that season
    - number of playoffs made till that season
    - win rate of total games till that season
        - consider regular or playoff only?
- ~~win rate for home court~~
- ~~win rate for away court~~
- ~~win rate for neutral court~~
- offensive stats
    - offensive rebounds
    - points scored
    - might try play by play later?
- defensive stats
    - defensive rebounds
    - points scored by opponents
    - turn over from play by play???
    - might try play by play later?
- blocks, steals and personal fouls


#### reconcilation of intermediate variables
- relative scoring method
     - will have a score of between 0 to 1


#### features being throw into prediction model
- test out raw intermediate variables
    - then test out difference in values
    - or something else


```python
#win rate for home court
#need to ensure that the joining is from a bigger table
raw_data_regularseason.head()
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
      <th>DayNum</th>
      <th>WTeamID</th>
      <th>WScore</th>
      <th>LTeamID</th>
      <th>LScore</th>
      <th>WLoc</th>
      <th>NumOT</th>
      <th>WFGM</th>
      <th>WFGA</th>
      <th>...</th>
      <th>LFGA3</th>
      <th>LFTM</th>
      <th>LFTA</th>
      <th>LOR</th>
      <th>LDR</th>
      <th>LAst</th>
      <th>LTO</th>
      <th>LStl</th>
      <th>LBlk</th>
      <th>LPF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>10</td>
      <td>1104</td>
      <td>68</td>
      <td>1328</td>
      <td>62</td>
      <td>N</td>
      <td>0</td>
      <td>27</td>
      <td>58</td>
      <td>...</td>
      <td>10</td>
      <td>16</td>
      <td>22</td>
      <td>10</td>
      <td>22</td>
      <td>8</td>
      <td>18</td>
      <td>9</td>
      <td>2</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>10</td>
      <td>1272</td>
      <td>70</td>
      <td>1393</td>
      <td>63</td>
      <td>N</td>
      <td>0</td>
      <td>26</td>
      <td>62</td>
      <td>...</td>
      <td>24</td>
      <td>9</td>
      <td>20</td>
      <td>20</td>
      <td>25</td>
      <td>7</td>
      <td>12</td>
      <td>8</td>
      <td>6</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>11</td>
      <td>1266</td>
      <td>73</td>
      <td>1437</td>
      <td>61</td>
      <td>N</td>
      <td>0</td>
      <td>24</td>
      <td>58</td>
      <td>...</td>
      <td>26</td>
      <td>14</td>
      <td>23</td>
      <td>31</td>
      <td>22</td>
      <td>9</td>
      <td>12</td>
      <td>2</td>
      <td>5</td>
      <td>23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>11</td>
      <td>1296</td>
      <td>56</td>
      <td>1457</td>
      <td>50</td>
      <td>N</td>
      <td>0</td>
      <td>18</td>
      <td>38</td>
      <td>...</td>
      <td>22</td>
      <td>8</td>
      <td>15</td>
      <td>17</td>
      <td>20</td>
      <td>9</td>
      <td>19</td>
      <td>4</td>
      <td>3</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>11</td>
      <td>1400</td>
      <td>77</td>
      <td>1208</td>
      <td>71</td>
      <td>N</td>
      <td>0</td>
      <td>30</td>
      <td>61</td>
      <td>...</td>
      <td>16</td>
      <td>17</td>
      <td>27</td>
      <td>21</td>
      <td>15</td>
      <td>12</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>




```python
win_test = (
    raw_data_regularseason
    .groupby(['Season','WTeamID','WLoc'])
    .count()
    .reset_index()
    [['Season','WTeamID','WLoc','DayNum']]
)
```


```python
lose_test = (
    raw_data_regularseason
    .groupby(['Season','LTeamID','WLoc'])
    .count()
    .reset_index()
    [['Season','LTeamID','WLoc','DayNum']]
)
```


```python
win_test.head()
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
      <th>WLoc</th>
      <th>DayNum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1102</td>
      <td>H</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1103</td>
      <td>A</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1103</td>
      <td>H</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1104</td>
      <td>A</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
lose_test.head()
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
      <th>LTeamID</th>
      <th>WLoc</th>
      <th>DayNum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>A</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1102</td>
      <td>H</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1102</td>
      <td>N</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1103</td>
      <td>A</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1103</td>
      <td>H</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = (
    lose_test
    .drop(['DayNum'],1)
    .append(win_test.rename(columns={"WTeamID":"LTeamID"}).drop(['DayNum'],1))
    .groupby(['Season','LTeamID','WLoc'])
    .count()
    .reset_index()
)

win_rate_type_of_court = (
    test
    .merge(win_test,how='left',left_on=['Season','LTeamID','WLoc'], right_on=['Season','WTeamID','WLoc'])
    .merge(lose_test,how='left',left_on=['Season','LTeamID','WLoc'],right_on=['Season','LTeamID','WLoc'])
    .fillna(0)
    .rename(columns={"LTeamID":"TeamID","DayNum_x":"games_won","DayNum_y":"games_lost"})
    .drop(['WTeamID'],1)
    .pipe(lambda x:x.assign(win_rate = x.games_won/(x.games_won + x.games_lost)))
)


win_rate_type_of_court.head()
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
      <th>WLoc</th>
      <th>games_won</th>
      <th>games_lost</th>
      <th>win_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>A</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1102</td>
      <td>H</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>0.473684</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1102</td>
      <td>N</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1103</td>
      <td>A</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1103</td>
      <td>H</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
win_rate_away = (
    win_rate_type_of_court
    .query("WLoc == 'A'")
    .rename(columns={"win_rate":"win_rate_away"})
    [['Season','TeamID','win_rate_away']]
)

win_rate_neutral = (
    win_rate_type_of_court
    .query("WLoc == 'N'")
    .rename(columns={"win_rate":"win_rate_neutral"})
    [['Season','TeamID','win_rate_neutral']]
)

win_rate_home = (
    win_rate_type_of_court
    .query("WLoc == 'H'")
    .rename(columns={"win_rate":"win_rate_home"})
    [['Season','TeamID','win_rate_home']]
)

more_testing = win_rate_type_of_court.sort_values(['TeamID','Season']).query("WLoc=='A'").head().groupby(['TeamID']).cumsum()

whatever = win_rate_away.sort_values(['TeamID','Season']).head()

more_testing.pipe(lambda x:x.assign(TeamID = whatever.TeamID.values))
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
      <th>games_won</th>
      <th>games_lost</th>
      <th>win_rate</th>
      <th>TeamID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10783</th>
      <td>2014</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.000000</td>
      <td>1101</td>
    </tr>
    <tr>
      <th>11801</th>
      <td>4029</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.142857</td>
      <td>1101</td>
    </tr>
    <tr>
      <th>12821</th>
      <td>6045</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>0.428571</td>
      <td>1101</td>
    </tr>
    <tr>
      <th>13839</th>
      <td>8062</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>0.873016</td>
      <td>1101</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.428571</td>
      <td>1102</td>
    </tr>
  </tbody>
</table>
</div>




```python
# combine back with cumulative table
cumulative_stats_for_team_each_year.head()
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
      <th>WScore</th>
      <th>WFGM</th>
      <th>WFGA</th>
      <th>WFGM3</th>
      <th>WFGA3</th>
      <th>WFTM</th>
      <th>WFTA</th>
      <th>losing_opponent_score</th>
      <th>winning_num_counts</th>
      <th>...</th>
      <th>win_rate</th>
      <th>WFGP</th>
      <th>WFG3P</th>
      <th>WFTP</th>
      <th>LFGP</th>
      <th>LFG3P</th>
      <th>LFTP</th>
      <th>fgp</th>
      <th>fg3p</th>
      <th>ftp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>825</td>
      <td>271</td>
      <td>480</td>
      <td>120</td>
      <td>259</td>
      <td>163</td>
      <td>249</td>
      <td>638</td>
      <td>12</td>
      <td>...</td>
      <td>0.428571</td>
      <td>0.564583</td>
      <td>0.463320</td>
      <td>0.654618</td>
      <td>0.417981</td>
      <td>0.305556</td>
      <td>0.647826</td>
      <td>0.481149</td>
      <td>0.375643</td>
      <td>0.651357</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2004</td>
      <td>2229</td>
      <td>737</td>
      <td>1393</td>
      <td>312</td>
      <td>734</td>
      <td>443</td>
      <td>636</td>
      <td>1709</td>
      <td>34</td>
      <td>...</td>
      <td>0.607143</td>
      <td>0.529074</td>
      <td>0.425068</td>
      <td>0.696541</td>
      <td>0.408686</td>
      <td>0.305732</td>
      <td>0.639576</td>
      <td>0.481886</td>
      <td>0.378423</td>
      <td>0.678999</td>
    </tr>
    <tr>
      <th>653</th>
      <td>2005</td>
      <td>3326</td>
      <td>1115</td>
      <td>2180</td>
      <td>458</td>
      <td>1110</td>
      <td>638</td>
      <td>902</td>
      <td>2533</td>
      <td>51</td>
      <td>...</td>
      <td>0.600000</td>
      <td>0.511468</td>
      <td>0.412613</td>
      <td>0.707317</td>
      <td>0.408027</td>
      <td>0.318804</td>
      <td>0.661616</td>
      <td>0.469388</td>
      <td>0.373236</td>
      <td>0.693374</td>
    </tr>
    <tr>
      <th>982</th>
      <td>2006</td>
      <td>4756</td>
      <td>1604</td>
      <td>3171</td>
      <td>659</td>
      <td>1588</td>
      <td>889</td>
      <td>1234</td>
      <td>3676</td>
      <td>73</td>
      <td>...</td>
      <td>0.646018</td>
      <td>0.505834</td>
      <td>0.414987</td>
      <td>0.720421</td>
      <td>0.412921</td>
      <td>0.317597</td>
      <td>0.669456</td>
      <td>0.472430</td>
      <td>0.378968</td>
      <td>0.706192</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2007</td>
      <td>6347</td>
      <td>2135</td>
      <td>4205</td>
      <td>870</td>
      <td>2061</td>
      <td>1207</td>
      <td>1652</td>
      <td>4844</td>
      <td>95</td>
      <td>...</td>
      <td>0.664336</td>
      <td>0.507729</td>
      <td>0.422125</td>
      <td>0.730630</td>
      <td>0.412256</td>
      <td>0.315093</td>
      <td>0.688119</td>
      <td>0.475389</td>
      <td>0.384158</td>
      <td>0.719221</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
intermediate_combine_stats_for_team_each_year = (
    cumulative_stats_for_team_each_year
    .merge(win_rate_away,how='left',on=['Season','TeamID'])
    .merge(win_rate_home,how='left',on=['Season','TeamID'])
    .merge(win_rate_neutral,how='left',on=['Season','TeamID'])
)

intermediate_combine_stats_for_team_each_year.head()
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
      <th>WScore</th>
      <th>WFGM</th>
      <th>WFGA</th>
      <th>WFGM3</th>
      <th>WFGA3</th>
      <th>WFTM</th>
      <th>WFTA</th>
      <th>losing_opponent_score</th>
      <th>winning_num_counts</th>
      <th>...</th>
      <th>WFTP</th>
      <th>LFGP</th>
      <th>LFG3P</th>
      <th>LFTP</th>
      <th>fgp</th>
      <th>fg3p</th>
      <th>ftp</th>
      <th>win_rate_away</th>
      <th>win_rate_home</th>
      <th>win_rate_neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>825</td>
      <td>271</td>
      <td>480</td>
      <td>120</td>
      <td>259</td>
      <td>163</td>
      <td>249</td>
      <td>638</td>
      <td>12</td>
      <td>...</td>
      <td>0.654618</td>
      <td>0.417981</td>
      <td>0.305556</td>
      <td>0.647826</td>
      <td>0.481149</td>
      <td>0.375643</td>
      <td>0.651357</td>
      <td>0.428571</td>
      <td>0.473684</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2004</td>
      <td>2229</td>
      <td>737</td>
      <td>1393</td>
      <td>312</td>
      <td>734</td>
      <td>443</td>
      <td>636</td>
      <td>1709</td>
      <td>34</td>
      <td>...</td>
      <td>0.696541</td>
      <td>0.408686</td>
      <td>0.305732</td>
      <td>0.639576</td>
      <td>0.481886</td>
      <td>0.378423</td>
      <td>0.678999</td>
      <td>1.000000</td>
      <td>0.722222</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2005</td>
      <td>3326</td>
      <td>1115</td>
      <td>2180</td>
      <td>458</td>
      <td>1110</td>
      <td>638</td>
      <td>902</td>
      <td>2533</td>
      <td>51</td>
      <td>...</td>
      <td>0.707317</td>
      <td>0.408027</td>
      <td>0.318804</td>
      <td>0.661616</td>
      <td>0.469388</td>
      <td>0.373236</td>
      <td>0.693374</td>
      <td>0.800000</td>
      <td>0.550000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2006</td>
      <td>4756</td>
      <td>1604</td>
      <td>3171</td>
      <td>659</td>
      <td>1588</td>
      <td>889</td>
      <td>1234</td>
      <td>3676</td>
      <td>73</td>
      <td>...</td>
      <td>0.720421</td>
      <td>0.412921</td>
      <td>0.317597</td>
      <td>0.669456</td>
      <td>0.472430</td>
      <td>0.378968</td>
      <td>0.706192</td>
      <td>1.000000</td>
      <td>0.736842</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007</td>
      <td>6347</td>
      <td>2135</td>
      <td>4205</td>
      <td>870</td>
      <td>2061</td>
      <td>1207</td>
      <td>1652</td>
      <td>4844</td>
      <td>95</td>
      <td>...</td>
      <td>0.730630</td>
      <td>0.412256</td>
      <td>0.315093</td>
      <td>0.688119</td>
      <td>0.475389</td>
      <td>0.384158</td>
      <td>0.719221</td>
      <td>0.857143</td>
      <td>0.722222</td>
      <td>0.600000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



## offensive stats


```python
# scored 
# offensive rebounds
# percentage of offensive rebounds to total rebounds
# offensive rebounding percentage, field goal missed
# defensive rebounds
```


```python
# block % from opponent field goal attempted
# assist / turnover ratio
# assist per fgm

# win by how many points
# lose by how many points
```


```python
# normalization on variables
```

## Features selected
- season
- region --> perhaps encode to a number. example: west - east = 1001. west = victor, east = loser
- wteamid
- wscore
- lteamid
- lscore
- wloc
- winning field goal percentage
- winning three point percentage
- winning free throw percentage
- transformed variable for rebounds (offensive and defensive)
- transformed assist
- transformed turnovers
- transformed steals
- transformed blocks
- transformed personal fouls
- repeat for losing team

*transformed variables exclude first


```python
pd.read_csv("data/DataFiles/TeamCoaches.csv").head()
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
      <th>FirstDayNum</th>
      <th>LastDayNum</th>
      <th>CoachName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1102</td>
      <td>0</td>
      <td>154</td>
      <td>reggie_minton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>0</td>
      <td>154</td>
      <td>bob_huggins</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>0</td>
      <td>154</td>
      <td>wimp_sanderson</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>0</td>
      <td>154</td>
      <td>james_oliver</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>0</td>
      <td>154</td>
      <td>davey_whitney</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv("data/DataFiles/Teams.csv")
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
      <th>TeamID</th>
      <th>TeamName</th>
      <th>FirstD1Season</th>
      <th>LastD1Season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1101</td>
      <td>Abilene Chr</td>
      <td>2014</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1102</td>
      <td>Air Force</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1103</td>
      <td>Akron</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1104</td>
      <td>Alabama</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1105</td>
      <td>Alabama A&amp;M</td>
      <td>2000</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1106</td>
      <td>Alabama St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1107</td>
      <td>Albany NY</td>
      <td>2000</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1108</td>
      <td>Alcorn St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1109</td>
      <td>Alliant Intl</td>
      <td>1985</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1110</td>
      <td>American Univ</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1111</td>
      <td>Appalachian St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1112</td>
      <td>Arizona</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1113</td>
      <td>Arizona St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1114</td>
      <td>Ark Little Rock</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1115</td>
      <td>Ark Pine Bluff</td>
      <td>1999</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1116</td>
      <td>Arkansas</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1117</td>
      <td>Arkansas St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1118</td>
      <td>Armstrong St</td>
      <td>1987</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1119</td>
      <td>Army</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1120</td>
      <td>Auburn</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1121</td>
      <td>Augusta</td>
      <td>1985</td>
      <td>1991</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1122</td>
      <td>Austin Peay</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1123</td>
      <td>Ball St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1124</td>
      <td>Baylor</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1125</td>
      <td>Belmont</td>
      <td>2000</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1126</td>
      <td>Bethune-Cookman</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1127</td>
      <td>Binghamton</td>
      <td>2002</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1128</td>
      <td>Birmingham So</td>
      <td>2003</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1129</td>
      <td>Boise St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1130</td>
      <td>Boston College</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>334</th>
      <td>1435</td>
      <td>Vanderbilt</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>335</th>
      <td>1436</td>
      <td>Vermont</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>336</th>
      <td>1437</td>
      <td>Villanova</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>337</th>
      <td>1438</td>
      <td>Virginia</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1439</td>
      <td>Virginia Tech</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>339</th>
      <td>1440</td>
      <td>VMI</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>340</th>
      <td>1441</td>
      <td>W Carolina</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>341</th>
      <td>1442</td>
      <td>W Illinois</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>342</th>
      <td>1443</td>
      <td>WKU</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>343</th>
      <td>1444</td>
      <td>W Michigan</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>344</th>
      <td>1445</td>
      <td>W Salem St</td>
      <td>2007</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>345</th>
      <td>1446</td>
      <td>W Texas A&amp;M</td>
      <td>1985</td>
      <td>1986</td>
    </tr>
    <tr>
      <th>346</th>
      <td>1447</td>
      <td>Wagner</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>347</th>
      <td>1448</td>
      <td>Wake Forest</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>348</th>
      <td>1449</td>
      <td>Washington</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>349</th>
      <td>1450</td>
      <td>Washington St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>350</th>
      <td>1451</td>
      <td>Weber St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>351</th>
      <td>1452</td>
      <td>West Virginia</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>352</th>
      <td>1453</td>
      <td>WI Green Bay</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>353</th>
      <td>1454</td>
      <td>WI Milwaukee</td>
      <td>1991</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>354</th>
      <td>1455</td>
      <td>Wichita St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>355</th>
      <td>1456</td>
      <td>William &amp; Mary</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>356</th>
      <td>1457</td>
      <td>Winthrop</td>
      <td>1987</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>357</th>
      <td>1458</td>
      <td>Wisconsin</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>358</th>
      <td>1459</td>
      <td>Wofford</td>
      <td>1996</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>359</th>
      <td>1460</td>
      <td>Wright St</td>
      <td>1988</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>360</th>
      <td>1461</td>
      <td>Wyoming</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>361</th>
      <td>1462</td>
      <td>Xavier</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>362</th>
      <td>1463</td>
      <td>Yale</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>363</th>
      <td>1464</td>
      <td>Youngstown St</td>
      <td>1985</td>
      <td>2018</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 4 columns</p>
</div>




```python
pd.read_csv("data/DataFiles/RegularSeasonCompactResults.csv").head()
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
      <th>DayNum</th>
      <th>WTeamID</th>
      <th>WScore</th>
      <th>LTeamID</th>
      <th>LScore</th>
      <th>WLoc</th>
      <th>NumOT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>20</td>
      <td>1228</td>
      <td>81</td>
      <td>1328</td>
      <td>64</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>25</td>
      <td>1106</td>
      <td>77</td>
      <td>1354</td>
      <td>70</td>
      <td>H</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>25</td>
      <td>1112</td>
      <td>63</td>
      <td>1223</td>
      <td>56</td>
      <td>H</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>25</td>
      <td>1165</td>
      <td>70</td>
      <td>1432</td>
      <td>54</td>
      <td>H</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>25</td>
      <td>1192</td>
      <td>86</td>
      <td>1447</td>
      <td>74</td>
      <td>H</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.read_csv("data/DataFiles/NCAATourneyCompactResults.csv").query("DayNum==154")
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
      <th>DayNum</th>
      <th>WTeamID</th>
      <th>WScore</th>
      <th>LTeamID</th>
      <th>LScore</th>
      <th>WLoc</th>
      <th>NumOT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>1985</td>
      <td>154</td>
      <td>1437</td>
      <td>66</td>
      <td>1207</td>
      <td>64</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>1986</td>
      <td>154</td>
      <td>1257</td>
      <td>72</td>
      <td>1181</td>
      <td>69</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>188</th>
      <td>1987</td>
      <td>154</td>
      <td>1231</td>
      <td>74</td>
      <td>1393</td>
      <td>73</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>251</th>
      <td>1988</td>
      <td>154</td>
      <td>1242</td>
      <td>83</td>
      <td>1328</td>
      <td>79</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>314</th>
      <td>1989</td>
      <td>154</td>
      <td>1276</td>
      <td>80</td>
      <td>1371</td>
      <td>79</td>
      <td>N</td>
      <td>1</td>
    </tr>
    <tr>
      <th>377</th>
      <td>1990</td>
      <td>154</td>
      <td>1424</td>
      <td>103</td>
      <td>1181</td>
      <td>73</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>1991</td>
      <td>154</td>
      <td>1181</td>
      <td>72</td>
      <td>1242</td>
      <td>65</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>503</th>
      <td>1992</td>
      <td>154</td>
      <td>1181</td>
      <td>71</td>
      <td>1276</td>
      <td>51</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>566</th>
      <td>1993</td>
      <td>154</td>
      <td>1314</td>
      <td>77</td>
      <td>1276</td>
      <td>71</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>629</th>
      <td>1994</td>
      <td>154</td>
      <td>1116</td>
      <td>76</td>
      <td>1181</td>
      <td>72</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>692</th>
      <td>1995</td>
      <td>154</td>
      <td>1417</td>
      <td>89</td>
      <td>1116</td>
      <td>78</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>755</th>
      <td>1996</td>
      <td>154</td>
      <td>1246</td>
      <td>76</td>
      <td>1393</td>
      <td>67</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>818</th>
      <td>1997</td>
      <td>154</td>
      <td>1112</td>
      <td>84</td>
      <td>1246</td>
      <td>79</td>
      <td>N</td>
      <td>1</td>
    </tr>
    <tr>
      <th>881</th>
      <td>1998</td>
      <td>154</td>
      <td>1246</td>
      <td>78</td>
      <td>1428</td>
      <td>69</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>944</th>
      <td>1999</td>
      <td>154</td>
      <td>1163</td>
      <td>77</td>
      <td>1181</td>
      <td>74</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>2000</td>
      <td>154</td>
      <td>1277</td>
      <td>89</td>
      <td>1196</td>
      <td>76</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>2001</td>
      <td>154</td>
      <td>1181</td>
      <td>82</td>
      <td>1112</td>
      <td>72</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>2002</td>
      <td>154</td>
      <td>1268</td>
      <td>64</td>
      <td>1231</td>
      <td>52</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1199</th>
      <td>2003</td>
      <td>154</td>
      <td>1393</td>
      <td>81</td>
      <td>1242</td>
      <td>78</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>2004</td>
      <td>154</td>
      <td>1163</td>
      <td>82</td>
      <td>1210</td>
      <td>73</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>2005</td>
      <td>154</td>
      <td>1314</td>
      <td>75</td>
      <td>1228</td>
      <td>70</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1391</th>
      <td>2006</td>
      <td>154</td>
      <td>1196</td>
      <td>73</td>
      <td>1417</td>
      <td>57</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>2007</td>
      <td>154</td>
      <td>1196</td>
      <td>84</td>
      <td>1326</td>
      <td>75</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1519</th>
      <td>2008</td>
      <td>154</td>
      <td>1242</td>
      <td>75</td>
      <td>1272</td>
      <td>68</td>
      <td>N</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1583</th>
      <td>2009</td>
      <td>154</td>
      <td>1314</td>
      <td>89</td>
      <td>1277</td>
      <td>72</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1647</th>
      <td>2010</td>
      <td>154</td>
      <td>1181</td>
      <td>61</td>
      <td>1139</td>
      <td>59</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1714</th>
      <td>2011</td>
      <td>154</td>
      <td>1163</td>
      <td>53</td>
      <td>1139</td>
      <td>41</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1781</th>
      <td>2012</td>
      <td>154</td>
      <td>1246</td>
      <td>67</td>
      <td>1242</td>
      <td>59</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1848</th>
      <td>2013</td>
      <td>154</td>
      <td>1257</td>
      <td>82</td>
      <td>1276</td>
      <td>76</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>2014</td>
      <td>154</td>
      <td>1163</td>
      <td>60</td>
      <td>1246</td>
      <td>54</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>2015</td>
      <td>154</td>
      <td>1181</td>
      <td>68</td>
      <td>1458</td>
      <td>63</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2049</th>
      <td>2016</td>
      <td>154</td>
      <td>1437</td>
      <td>77</td>
      <td>1314</td>
      <td>74</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2116</th>
      <td>2017</td>
      <td>154</td>
      <td>1314</td>
      <td>71</td>
      <td>1211</td>
      <td>65</td>
      <td>N</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




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
      <th>EventID</th>
      <th>Season</th>
      <th>DayNum</th>
      <th>WTeamID</th>
      <th>LTeamID</th>
      <th>WPoints</th>
      <th>LPoints</th>
      <th>ElapsedSeconds</th>
      <th>EventTeamID</th>
      <th>EventPlayerID</th>
      <th>EventType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2010</td>
      <td>7</td>
      <td>1143</td>
      <td>1293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1143</td>
      <td>600578</td>
      <td>sub_in</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2010</td>
      <td>7</td>
      <td>1143</td>
      <td>1293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1143</td>
      <td>600584</td>
      <td>sub_in</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2010</td>
      <td>7</td>
      <td>1143</td>
      <td>1293</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1143</td>
      <td>600585</td>
      <td>sub_in</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2010</td>
      <td>7</td>
      <td>1143</td>
      <td>1293</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1143</td>
      <td>600581</td>
      <td>miss2_lay</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2010</td>
      <td>7</td>
      <td>1143</td>
      <td>1293</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1143</td>
      <td>600581</td>
      <td>reb_off</td>
    </tr>
  </tbody>
</table>
</div>


