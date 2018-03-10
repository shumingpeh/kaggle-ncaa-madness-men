
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
- ~~expectation to win by how many points in a game~~
- 


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
#win and lose by how many points
```


```python
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
win_rate_df = (
    combine_winning_losing_stats_for_year
    [['Season','WTeamID','winning_num_counts','losing_num_counts','WScore','losing_opponent_score','LScore','winning_opponent_score']]
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
    .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
    .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
    .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
    .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
    .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
    .rename(columns={"WTeamID":"TeamID"})
)

win_rate_df.head()
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
      <th>winning_num_counts</th>
      <th>losing_num_counts</th>
      <th>WScore</th>
      <th>losing_opponent_score</th>
      <th>LScore</th>
      <th>winning_opponent_score</th>
      <th>win_rate</th>
      <th>lose_rate</th>
      <th>win_score_by</th>
      <th>lose_score_by</th>
      <th>expectation_per_game</th>
      <th>avg_win_score_by</th>
      <th>avg_lose_score_by</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>12</td>
      <td>16</td>
      <td>825</td>
      <td>638</td>
      <td>778</td>
      <td>958</td>
      <td>0.428571</td>
      <td>0.571429</td>
      <td>187</td>
      <td>-180</td>
      <td>0.250000</td>
      <td>15.583333</td>
      <td>-11.250000</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2004</td>
      <td>1102</td>
      <td>22</td>
      <td>6</td>
      <td>1404</td>
      <td>1071</td>
      <td>281</td>
      <td>341</td>
      <td>0.785714</td>
      <td>0.214286</td>
      <td>333</td>
      <td>-60</td>
      <td>9.750000</td>
      <td>15.136364</td>
      <td>-10.000000</td>
    </tr>
    <tr>
      <th>653</th>
      <td>2005</td>
      <td>1102</td>
      <td>17</td>
      <td>12</td>
      <td>1097</td>
      <td>824</td>
      <td>679</td>
      <td>775</td>
      <td>0.586207</td>
      <td>0.413793</td>
      <td>273</td>
      <td>-96</td>
      <td>6.103448</td>
      <td>16.058824</td>
      <td>-8.000000</td>
    </tr>
    <tr>
      <th>982</th>
      <td>2006</td>
      <td>1102</td>
      <td>22</td>
      <td>6</td>
      <td>1430</td>
      <td>1143</td>
      <td>348</td>
      <td>385</td>
      <td>0.785714</td>
      <td>0.214286</td>
      <td>287</td>
      <td>-37</td>
      <td>8.928571</td>
      <td>13.045455</td>
      <td>-6.166667</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2007</td>
      <td>1102</td>
      <td>22</td>
      <td>8</td>
      <td>1591</td>
      <td>1168</td>
      <td>464</td>
      <td>539</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>423</td>
      <td>-75</td>
      <td>11.600000</td>
      <td>19.227273</td>
      <td>-9.375000</td>
    </tr>
  </tbody>
</table>
</div>




```python
win_rate_cum_df = (
    cumulative_stats_for_team_each_year
    [['Season','TeamID','winning_num_counts','losing_num_counts','WScore','losing_opponent_score','LScore','winning_opponent_score']]
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
    .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
    .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
    .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
    .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
    .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
)

win_rate_cum_df.head()
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
      <th>winning_num_counts</th>
      <th>losing_num_counts</th>
      <th>WScore</th>
      <th>losing_opponent_score</th>
      <th>LScore</th>
      <th>winning_opponent_score</th>
      <th>win_rate</th>
      <th>lose_rate</th>
      <th>win_score_by</th>
      <th>lose_score_by</th>
      <th>expectation_per_game</th>
      <th>avg_win_score_by</th>
      <th>avg_lose_score_by</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>12</td>
      <td>16</td>
      <td>825</td>
      <td>638</td>
      <td>778</td>
      <td>958</td>
      <td>0.428571</td>
      <td>0.571429</td>
      <td>187</td>
      <td>-180</td>
      <td>0.250000</td>
      <td>15.583333</td>
      <td>-11.250000</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2004</td>
      <td>1102</td>
      <td>34</td>
      <td>22</td>
      <td>2229</td>
      <td>1709</td>
      <td>1059</td>
      <td>1299</td>
      <td>0.607143</td>
      <td>0.392857</td>
      <td>520</td>
      <td>-240</td>
      <td>5.000000</td>
      <td>15.294118</td>
      <td>-10.909091</td>
    </tr>
    <tr>
      <th>653</th>
      <td>2005</td>
      <td>1102</td>
      <td>51</td>
      <td>34</td>
      <td>3326</td>
      <td>2533</td>
      <td>1738</td>
      <td>2074</td>
      <td>0.600000</td>
      <td>0.400000</td>
      <td>793</td>
      <td>-336</td>
      <td>5.376471</td>
      <td>15.549020</td>
      <td>-9.882353</td>
    </tr>
    <tr>
      <th>982</th>
      <td>2006</td>
      <td>1102</td>
      <td>73</td>
      <td>40</td>
      <td>4756</td>
      <td>3676</td>
      <td>2086</td>
      <td>2459</td>
      <td>0.646018</td>
      <td>0.353982</td>
      <td>1080</td>
      <td>-373</td>
      <td>6.256637</td>
      <td>14.794521</td>
      <td>-9.325000</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2007</td>
      <td>1102</td>
      <td>95</td>
      <td>48</td>
      <td>6347</td>
      <td>4844</td>
      <td>2550</td>
      <td>2998</td>
      <td>0.664336</td>
      <td>0.335664</td>
      <td>1503</td>
      <td>-448</td>
      <td>7.377622</td>
      <td>15.821053</td>
      <td>-9.333333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rebounds
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




```python
rebounds_winning_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WOR":"sum","WDR":"sum","WFGA":"sum","WFGM":"sum","LFGM":"sum","LFGA":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(total_winning_rebounds = x.WOR + x.WDR))
    .pipe(lambda x:x.assign(winning_off_rebounds_percent = x.WOR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(winning_def_rebounds_percent = x.WDR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(opp_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_rebound_possession_percent = x.WOR/x.team_missed_attempts))
    .pipe(lambda x:x.assign(winning_rebound_possessiongain_percent = x.WDR/x.opp_team_missed_attempts))
)
```


```python
rebounds_winning_teams_score_up_to_2013.head()
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
      <th>WOR</th>
      <th>WDR</th>
      <th>WFGA</th>
      <th>WFGM</th>
      <th>LFGM</th>
      <th>LFGA</th>
      <th>total_winning_rebounds</th>
      <th>winning_off_rebounds_percent</th>
      <th>winning_def_rebounds_percent</th>
      <th>team_missed_attempts</th>
      <th>opp_team_missed_attempts</th>
      <th>winning_rebound_possession_percent</th>
      <th>winning_rebound_possessiongain_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>46</td>
      <td>232</td>
      <td>480</td>
      <td>271</td>
      <td>228</td>
      <td>560</td>
      <td>278</td>
      <td>0.165468</td>
      <td>0.834532</td>
      <td>209</td>
      <td>332</td>
      <td>0.220096</td>
      <td>0.698795</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>122</td>
      <td>279</td>
      <td>720</td>
      <td>390</td>
      <td>358</td>
      <td>780</td>
      <td>401</td>
      <td>0.304239</td>
      <td>0.695761</td>
      <td>330</td>
      <td>422</td>
      <td>0.369697</td>
      <td>0.661137</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>230</td>
      <td>449</td>
      <td>992</td>
      <td>439</td>
      <td>376</td>
      <td>978</td>
      <td>679</td>
      <td>0.338733</td>
      <td>0.661267</td>
      <td>553</td>
      <td>602</td>
      <td>0.415913</td>
      <td>0.745847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>102</td>
      <td>181</td>
      <td>433</td>
      <td>179</td>
      <td>161</td>
      <td>403</td>
      <td>283</td>
      <td>0.360424</td>
      <td>0.639576</td>
      <td>254</td>
      <td>242</td>
      <td>0.401575</td>
      <td>0.747934</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>166</td>
      <td>364</td>
      <td>700</td>
      <td>322</td>
      <td>244</td>
      <td>702</td>
      <td>530</td>
      <td>0.313208</td>
      <td>0.686792</td>
      <td>378</td>
      <td>458</td>
      <td>0.439153</td>
      <td>0.794760</td>
    </tr>
  </tbody>
</table>
</div>




```python
rebounds_losing_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"LOR":"sum","LDR":"sum","LFGM":"sum","LFGA":"sum","WFGA":"sum","WFGM":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(total_losing_rebounds = x.LOR + x.LDR))
    .pipe(lambda x:x.assign(losing_off_rebounds_percent = x.LOR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_def_rebounds_percent = x.LDR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_opp_team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(losing_rebound_possession_percent = x.LOR/x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(losing_rebound_possessiongain_percent = x.LDR/x.winning_opp_team_missed_attempts))
)

rebounds_losing_teams_score_up_to_2013.head()
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
      <th>LOR</th>
      <th>LDR</th>
      <th>LFGM</th>
      <th>LFGA</th>
      <th>WFGA</th>
      <th>WFGM</th>
      <th>total_losing_rebounds</th>
      <th>losing_off_rebounds_percent</th>
      <th>losing_def_rebounds_percent</th>
      <th>losing_team_missed_attempts</th>
      <th>winning_opp_team_missed_attempts</th>
      <th>losing_rebound_possession_percent</th>
      <th>losing_rebound_possessiongain_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>71</td>
      <td>239</td>
      <td>265</td>
      <td>634</td>
      <td>628</td>
      <td>312</td>
      <td>310</td>
      <td>0.229032</td>
      <td>0.770968</td>
      <td>369</td>
      <td>316</td>
      <td>0.192412</td>
      <td>0.756329</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>142</td>
      <td>259</td>
      <td>343</td>
      <td>788</td>
      <td>759</td>
      <td>392</td>
      <td>401</td>
      <td>0.354115</td>
      <td>0.645885</td>
      <td>445</td>
      <td>367</td>
      <td>0.319101</td>
      <td>0.705722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>150</td>
      <td>221</td>
      <td>234</td>
      <td>609</td>
      <td>576</td>
      <td>275</td>
      <td>371</td>
      <td>0.404313</td>
      <td>0.595687</td>
      <td>375</td>
      <td>301</td>
      <td>0.400000</td>
      <td>0.734219</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>249</td>
      <td>420</td>
      <td>455</td>
      <td>1169</td>
      <td>1130</td>
      <td>541</td>
      <td>669</td>
      <td>0.372197</td>
      <td>0.627803</td>
      <td>714</td>
      <td>589</td>
      <td>0.348739</td>
      <td>0.713073</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>178</td>
      <td>304</td>
      <td>334</td>
      <td>848</td>
      <td>793</td>
      <td>364</td>
      <td>482</td>
      <td>0.369295</td>
      <td>0.630705</td>
      <td>514</td>
      <td>429</td>
      <td>0.346304</td>
      <td>0.708625</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine_winning_losing_rebounds_stats_for_year = (
    rebounds_winning_teams_score_up_to_2013
    .merge(rebounds_losing_teams_score_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    .pipe(lambda x:x.assign(total_rebounds = x.total_winning_rebounds + x.total_losing_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds = x.WOR + x.LOR))
    .pipe(lambda x:x.assign(total_def_rebounds = x.WDR + x.LDR))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_team_missed_attempts = x.team_missed_attempts + x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(total_opp_team_missed_attempts = x.opp_team_missed_attempts + x.winning_opp_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    .rename(columns={"WTeamID":"TeamID"})
    [['Season','TeamID','total_rebounds','total_off_rebounds','total_def_rebounds','total_def_rebounds_percent',
      'total_off_rebounds_percent','total_rebound_possession_percent','total_rebound_possessiongain_percent',
      'total_team_missed_attempts','total_opp_team_missed_attempts']]
)
```


```python
combine_winning_losing_rebounds_stats_for_year.head()
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
      <th>total_rebounds</th>
      <th>total_off_rebounds</th>
      <th>total_def_rebounds</th>
      <th>total_def_rebounds_percent</th>
      <th>total_off_rebounds_percent</th>
      <th>total_rebound_possession_percent</th>
      <th>total_rebound_possessiongain_percent</th>
      <th>total_team_missed_attempts</th>
      <th>total_opp_team_missed_attempts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>588</td>
      <td>117</td>
      <td>471</td>
      <td>0.801020</td>
      <td>0.198980</td>
      <td>0.202422</td>
      <td>0.726852</td>
      <td>578</td>
      <td>648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>802</td>
      <td>264</td>
      <td>538</td>
      <td>0.670823</td>
      <td>0.329177</td>
      <td>0.340645</td>
      <td>0.681876</td>
      <td>775</td>
      <td>789</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>1050</td>
      <td>380</td>
      <td>670</td>
      <td>0.638095</td>
      <td>0.361905</td>
      <td>0.409483</td>
      <td>0.741971</td>
      <td>928</td>
      <td>903</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>952</td>
      <td>351</td>
      <td>601</td>
      <td>0.631303</td>
      <td>0.368697</td>
      <td>0.362603</td>
      <td>0.723225</td>
      <td>968</td>
      <td>831</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>1012</td>
      <td>344</td>
      <td>668</td>
      <td>0.660079</td>
      <td>0.339921</td>
      <td>0.385650</td>
      <td>0.753100</td>
      <td>892</td>
      <td>887</td>
    </tr>
  </tbody>
</table>
</div>




```python
cumulative_winning_losing_rebounds_stats = (
    combine_winning_losing_rebounds_stats_for_year
    .sort_values(['TeamID','Season'])
    .groupby(['TeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    .pipe(lambda x:x.assign(Season = combine_winning_losing_stats_for_year.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_winning_losing_stats_for_year.WTeamID.values))
)
```


```python
# blocks, steals, assists

```


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




```python
bl_sl_topf_winning_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WAst":"sum","WTO":"sum","WStl":"sum","WBlk":"sum","WPF":"sum","LFGA":"sum","WFGM":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(winning_block_opp_FGA_percent = x.WBlk/x.LFGA))
    .pipe(lambda x:x.assign(winning_assist_per_fgm = x.WAst/x.WFGM))
    .pipe(lambda x:x.assign(winning_assist_turnover_ratio = x.WAst/x.WTO))
)

bl_sl_topf_winning_teams_score_up_to_2013.head()
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
      <th>WAst</th>
      <th>WTO</th>
      <th>WStl</th>
      <th>WBlk</th>
      <th>WPF</th>
      <th>LFGA</th>
      <th>WFGM</th>
      <th>winning_block_opp_FGA_percent</th>
      <th>winning_assist_per_fgm</th>
      <th>winning_assist_turnover_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>203</td>
      <td>133</td>
      <td>88</td>
      <td>34</td>
      <td>193</td>
      <td>560</td>
      <td>271</td>
      <td>0.060714</td>
      <td>0.749077</td>
      <td>1.526316</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>230</td>
      <td>163</td>
      <td>95</td>
      <td>24</td>
      <td>266</td>
      <td>780</td>
      <td>390</td>
      <td>0.030769</td>
      <td>0.589744</td>
      <td>1.411043</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>238</td>
      <td>222</td>
      <td>123</td>
      <td>71</td>
      <td>280</td>
      <td>978</td>
      <td>439</td>
      <td>0.072597</td>
      <td>0.542141</td>
      <td>1.072072</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>111</td>
      <td>126</td>
      <td>79</td>
      <td>14</td>
      <td>136</td>
      <td>403</td>
      <td>179</td>
      <td>0.034739</td>
      <td>0.620112</td>
      <td>0.880952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>169</td>
      <td>230</td>
      <td>122</td>
      <td>49</td>
      <td>239</td>
      <td>702</td>
      <td>322</td>
      <td>0.069801</td>
      <td>0.524845</td>
      <td>0.734783</td>
    </tr>
  </tbody>
</table>
</div>




```python
bl_sl_topf_losing_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"LAst":"sum","LTO":"sum","LStl":"sum","LBlk":"sum","LPF":"sum","WFGA":"sum","LFGM":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(losing_block_opp_FGA_percent = x.LBlk/x.WFGA))
    .pipe(lambda x:x.assign(losing_assist_per_fgm = x.LAst/x.LFGM))
    .pipe(lambda x:x.assign(losing_assist_turnover_ratio = x.LAst/x.LTO))
)

bl_sl_topf_losing_teams_score_up_to_2013.head()
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
      <th>LAst</th>
      <th>LTO</th>
      <th>LStl</th>
      <th>LBlk</th>
      <th>LPF</th>
      <th>WFGA</th>
      <th>LFGM</th>
      <th>losing_block_opp_FGA_percent</th>
      <th>losing_assist_per_fgm</th>
      <th>losing_assist_turnover_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>161</td>
      <td>187</td>
      <td>79</td>
      <td>16</td>
      <td>332</td>
      <td>628</td>
      <td>265</td>
      <td>0.025478</td>
      <td>0.607547</td>
      <td>0.860963</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>181</td>
      <td>178</td>
      <td>101</td>
      <td>39</td>
      <td>270</td>
      <td>759</td>
      <td>343</td>
      <td>0.051383</td>
      <td>0.527697</td>
      <td>1.016854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>101</td>
      <td>150</td>
      <td>62</td>
      <td>35</td>
      <td>225</td>
      <td>576</td>
      <td>234</td>
      <td>0.060764</td>
      <td>0.431624</td>
      <td>0.673333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>267</td>
      <td>359</td>
      <td>163</td>
      <td>40</td>
      <td>390</td>
      <td>1130</td>
      <td>455</td>
      <td>0.035398</td>
      <td>0.586813</td>
      <td>0.743733</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>158</td>
      <td>247</td>
      <td>112</td>
      <td>39</td>
      <td>270</td>
      <td>793</td>
      <td>334</td>
      <td>0.049180</td>
      <td>0.473054</td>
      <td>0.639676</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine_winning_losing_other_stats_for_year = (
    bl_sl_topf_winning_teams_score_up_to_2013
    .merge(bl_sl_topf_losing_teams_score_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    .pipe(lambda x:x.assign(total_blocks = x.WBlk + x.LBlk))
    .pipe(lambda x:x.assign(total_assists = x.WAst + x.LAst))
    .pipe(lambda x:x.assign(total_steals = x.WStl + x.LStl))
    .pipe(lambda x:x.assign(total_turnover = x.WTO + x.LTO))
    .pipe(lambda x:x.assign(total_personalfoul = x.WPF + x.LPF))
    .pipe(lambda x:x.assign(total_opp_fga = x.LFGA + x.WFGA))
    .pipe(lambda x:x.assign(total_fgm = x.WFGM + x.LFGM))
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
    .rename(columns={"WTeamID":"TeamID"})
    [['Season','TeamID','total_blocks','total_assists','total_steals','total_turnover','total_personalfoul','total_block_opp_FGA_percent','total_assist_per_fgm','total_assist_turnover_ratio','total_opp_fga','total_fgm']]
)
```


```python
combine_winning_losing_other_stats_for_year.head()
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
      <th>total_blocks</th>
      <th>total_assists</th>
      <th>total_steals</th>
      <th>total_turnover</th>
      <th>total_personalfoul</th>
      <th>total_block_opp_FGA_percent</th>
      <th>total_assist_per_fgm</th>
      <th>total_assist_turnover_ratio</th>
      <th>total_opp_fga</th>
      <th>total_fgm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>50</td>
      <td>364</td>
      <td>167</td>
      <td>320</td>
      <td>525</td>
      <td>0.042088</td>
      <td>0.679104</td>
      <td>1.137500</td>
      <td>1188</td>
      <td>536</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>63</td>
      <td>411</td>
      <td>196</td>
      <td>341</td>
      <td>536</td>
      <td>0.040936</td>
      <td>0.560709</td>
      <td>1.205279</td>
      <td>1539</td>
      <td>733</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>106</td>
      <td>339</td>
      <td>185</td>
      <td>372</td>
      <td>505</td>
      <td>0.068211</td>
      <td>0.503715</td>
      <td>0.911290</td>
      <td>1554</td>
      <td>673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>54</td>
      <td>378</td>
      <td>242</td>
      <td>485</td>
      <td>526</td>
      <td>0.035225</td>
      <td>0.596215</td>
      <td>0.779381</td>
      <td>1533</td>
      <td>634</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>88</td>
      <td>327</td>
      <td>234</td>
      <td>477</td>
      <td>509</td>
      <td>0.058863</td>
      <td>0.498476</td>
      <td>0.685535</td>
      <td>1495</td>
      <td>656</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine_winning_losing_other_stats_for_year.dtypes
```




    Season                           int64
    TeamID                           int64
    total_blocks                     int64
    total_assists                    int64
    total_steals                     int64
    total_turnover                   int64
    total_personalfoul               int64
    total_block_opp_FGA_percent    float64
    total_assist_per_fgm           float64
    total_assist_turnover_ratio    float64
    total_opp_fga                    int64
    total_fgm                        int64
    dtype: object




```python
cumulative_winning_losing_rebounds_stats = (
    combine_winning_losing_other_stats_for_year
    .sort_values(['TeamID','Season'])
    .groupby(['TeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
    .pipe(lambda x:x.assign(Season = combine_winning_losing_stats_for_year.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_winning_losing_stats_for_year.WTeamID.values))
)
```


```python
cumulative_winning_losing_rebounds_stats.head()
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
      <th>total_blocks</th>
      <th>total_assists</th>
      <th>total_steals</th>
      <th>total_turnover</th>
      <th>total_personalfoul</th>
      <th>total_block_opp_FGA_percent</th>
      <th>total_assist_per_fgm</th>
      <th>total_assist_turnover_ratio</th>
      <th>total_opp_fga</th>
      <th>total_fgm</th>
      <th>TeamID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>50</td>
      <td>364</td>
      <td>167</td>
      <td>320</td>
      <td>525</td>
      <td>0.042088</td>
      <td>0.679104</td>
      <td>1.137500</td>
      <td>1188</td>
      <td>536</td>
      <td>1102</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2004</td>
      <td>116</td>
      <td>735</td>
      <td>385</td>
      <td>622</td>
      <td>991</td>
      <td>0.050043</td>
      <td>0.665761</td>
      <td>1.181672</td>
      <td>2318</td>
      <td>1104</td>
      <td>1102</td>
    </tr>
    <tr>
      <th>653</th>
      <td>2005</td>
      <td>166</td>
      <td>1131</td>
      <td>640</td>
      <td>914</td>
      <td>1469</td>
      <td>0.047715</td>
      <td>0.655652</td>
      <td>1.237418</td>
      <td>3479</td>
      <td>1725</td>
      <td>1102</td>
    </tr>
    <tr>
      <th>982</th>
      <td>2006</td>
      <td>225</td>
      <td>1528</td>
      <td>864</td>
      <td>1220</td>
      <td>1900</td>
      <td>0.047249</td>
      <td>0.653271</td>
      <td>1.252459</td>
      <td>4762</td>
      <td>2339</td>
      <td>1102</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2007</td>
      <td>269</td>
      <td>1979</td>
      <td>1052</td>
      <td>1525</td>
      <td>2355</td>
      <td>0.043255</td>
      <td>0.654648</td>
      <td>1.297705</td>
      <td>6219</td>
      <td>3023</td>
      <td>1102</td>
    </tr>
  </tbody>
</table>
</div>




```python
#min max standardization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
minmax_scale = scaler.fit(combine_winning_losing_other_stats_for_year[['total_assists']])
df_minmax = minmax_scale.transform(combine_winning_losing_other_stats_for_year[['total_assists']])
```


```python
winning_games_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts = 1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WScore":"sum","WFGM":"sum","WFGA":"sum","WFGM3":"sum","WFGA3":"sum","WFTM":"sum","WFTA":"sum","LScore":"sum","winning_num_counts":"sum",
          "WOR":"sum","WDR":"sum","LFGM":"sum","LFGA":"sum",
          "WAst":"sum","WTO":"sum","WStl":"sum","WBlk":"sum","WPF":"sum"})
    .reset_index()
    .rename(columns={"LScore":"losing_opponent_score"})
    # rebounds
    .pipe(lambda x:x.assign(total_winning_rebounds = x.WOR + x.WDR))
    .pipe(lambda x:x.assign(winning_off_rebounds_percent = x.WOR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(winning_def_rebounds_percent = x.WDR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(opp_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_rebound_possession_percent = x.WOR/x.team_missed_attempts))
    .pipe(lambda x:x.assign(winning_rebound_possessiongain_percent = x.WDR/x.opp_team_missed_attempts))
    # blocks, steals, assists and turnovers
    .pipe(lambda x:x.assign(winning_block_opp_FGA_percent = x.WBlk/x.LFGA))
    .pipe(lambda x:x.assign(winning_assist_per_fgm = x.WAst/x.WFGM))
    .pipe(lambda x:x.assign(winning_assist_turnover_ratio = x.WAst/x.WTO))
    # rename columns to prevent duplication when joining with losing stats. example: WFGM_x
    .rename(columns={"LFGA":"LFGA_opp","LFGM":"LFGM_opp"})
)
```


```python
winning_games_up_to_2013.head()
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
      <th>...</th>
      <th>total_winning_rebounds</th>
      <th>winning_off_rebounds_percent</th>
      <th>winning_def_rebounds_percent</th>
      <th>team_missed_attempts</th>
      <th>opp_team_missed_attempts</th>
      <th>winning_rebound_possession_percent</th>
      <th>winning_rebound_possessiongain_percent</th>
      <th>winning_block_opp_FGA_percent</th>
      <th>winning_assist_per_fgm</th>
      <th>winning_assist_turnover_ratio</th>
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
      <td>...</td>
      <td>278</td>
      <td>0.165468</td>
      <td>0.834532</td>
      <td>209</td>
      <td>332</td>
      <td>0.220096</td>
      <td>0.698795</td>
      <td>0.060714</td>
      <td>0.749077</td>
      <td>1.526316</td>
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
      <td>...</td>
      <td>401</td>
      <td>0.304239</td>
      <td>0.695761</td>
      <td>330</td>
      <td>422</td>
      <td>0.369697</td>
      <td>0.661137</td>
      <td>0.030769</td>
      <td>0.589744</td>
      <td>1.411043</td>
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
      <td>...</td>
      <td>679</td>
      <td>0.338733</td>
      <td>0.661267</td>
      <td>553</td>
      <td>602</td>
      <td>0.415913</td>
      <td>0.745847</td>
      <td>0.072597</td>
      <td>0.542141</td>
      <td>1.072072</td>
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
      <td>...</td>
      <td>283</td>
      <td>0.360424</td>
      <td>0.639576</td>
      <td>254</td>
      <td>242</td>
      <td>0.401575</td>
      <td>0.747934</td>
      <td>0.034739</td>
      <td>0.620112</td>
      <td>0.880952</td>
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
      <td>...</td>
      <td>530</td>
      <td>0.313208</td>
      <td>0.686792</td>
      <td>378</td>
      <td>458</td>
      <td>0.439153</td>
      <td>0.794760</td>
      <td>0.069801</td>
      <td>0.524845</td>
      <td>0.734783</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
losing_games_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"WScore":"sum","LScore":"sum","LFGM":"sum","LFGA":"sum","LFGM3":"sum","LFGA3":"sum","LFTM":"sum","LFTA":"sum","losing_num_counts":"sum",
          "LOR":"sum","LDR":"sum","WFGA":"sum","WFGM":"sum",
          "LAst":"sum","LTO":"sum","LStl":"sum","LBlk":"sum","LPF":"sum"})
    .reset_index()
    .rename(columns={"WScore":"winning_opponent_score"})
    # rebounds
    .pipe(lambda x:x.assign(total_losing_rebounds = x.LOR + x.LDR))
    .pipe(lambda x:x.assign(losing_off_rebounds_percent = x.LOR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_def_rebounds_percent = x.LDR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_opp_team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(losing_rebound_possession_percent = x.LOR/x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(losing_rebound_possessiongain_percent = x.LDR/x.winning_opp_team_missed_attempts))
    # blocks, steals, assists and turnovers
    .pipe(lambda x:x.assign(losing_block_opp_FGA_percent = x.LBlk/x.WFGA))
    .pipe(lambda x:x.assign(losing_assist_per_fgm = x.LAst/x.LFGM))
    .pipe(lambda x:x.assign(losing_assist_turnover_ratio = x.LAst/x.LTO))
    # rename columns to prevent duplication when joining with losing stats. example: WFGM_x
    .rename(columns={"WFGA":"WFGA_opp","WFGM":"WFGM_opp"})
)

losing_games_up_to_2013.head()
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
      <th>...</th>
      <th>total_losing_rebounds</th>
      <th>losing_off_rebounds_percent</th>
      <th>losing_def_rebounds_percent</th>
      <th>losing_team_missed_attempts</th>
      <th>winning_opp_team_missed_attempts</th>
      <th>losing_rebound_possession_percent</th>
      <th>losing_rebound_possessiongain_percent</th>
      <th>losing_block_opp_FGA_percent</th>
      <th>losing_assist_per_fgm</th>
      <th>losing_assist_turnover_ratio</th>
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
      <td>...</td>
      <td>310</td>
      <td>0.229032</td>
      <td>0.770968</td>
      <td>369</td>
      <td>316</td>
      <td>0.192412</td>
      <td>0.756329</td>
      <td>0.025478</td>
      <td>0.607547</td>
      <td>0.860963</td>
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
      <td>...</td>
      <td>401</td>
      <td>0.354115</td>
      <td>0.645885</td>
      <td>445</td>
      <td>367</td>
      <td>0.319101</td>
      <td>0.705722</td>
      <td>0.051383</td>
      <td>0.527697</td>
      <td>1.016854</td>
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
      <td>...</td>
      <td>371</td>
      <td>0.404313</td>
      <td>0.595687</td>
      <td>375</td>
      <td>301</td>
      <td>0.400000</td>
      <td>0.734219</td>
      <td>0.060764</td>
      <td>0.431624</td>
      <td>0.673333</td>
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
      <td>...</td>
      <td>669</td>
      <td>0.372197</td>
      <td>0.627803</td>
      <td>714</td>
      <td>589</td>
      <td>0.348739</td>
      <td>0.713073</td>
      <td>0.035398</td>
      <td>0.586813</td>
      <td>0.743733</td>
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
      <td>...</td>
      <td>482</td>
      <td>0.369295</td>
      <td>0.630705</td>
      <td>514</td>
      <td>429</td>
      <td>0.346304</td>
      <td>0.708625</td>
      <td>0.049180</td>
      <td>0.473054</td>
      <td>0.639676</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
combine_both_winning_losing_games_stats = (
    winning_games_up_to_2013
    .merge(losing_games_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    # on field goal percentage and winning counts
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
    # on offensive and defensive rebounds
    .pipe(lambda x:x.assign(total_rebounds = x.total_winning_rebounds + x.total_losing_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds = x.WOR + x.LOR))
    .pipe(lambda x:x.assign(total_def_rebounds = x.WDR + x.LDR))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_team_missed_attempts = x.team_missed_attempts + x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(total_opp_team_missed_attempts = x.opp_team_missed_attempts + x.winning_opp_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    # on steals, turnovers, assists, blocks and personal fouls
    .pipe(lambda x:x.assign(total_blocks = x.WBlk + x.LBlk))
    .pipe(lambda x:x.assign(total_assists = x.WAst + x.LAst))
    .pipe(lambda x:x.assign(total_steals = x.WStl + x.LStl))
    .pipe(lambda x:x.assign(total_turnover = x.WTO + x.LTO))
    .pipe(lambda x:x.assign(total_personalfoul = x.WPF + x.LPF))
    .pipe(lambda x:x.assign(total_opp_fga = x.LFGA_opp + x.WFGA_opp))
    .pipe(lambda x:x.assign(total_fgm = x.WFGM + x.LFGM))
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
)

combine_both_winning_losing_games_stats.head()
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
      <th>...</th>
      <th>total_rebound_possessiongain_percent</th>
      <th>total_blocks</th>
      <th>total_assists</th>
      <th>total_steals</th>
      <th>total_turnover</th>
      <th>total_personalfoul</th>
      <th>total_opp_fga</th>
      <th>total_block_opp_FGA_percent</th>
      <th>total_assist_per_fgm</th>
      <th>total_assist_turnover_ratio</th>
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
      <td>...</td>
      <td>0.726852</td>
      <td>50</td>
      <td>364</td>
      <td>167</td>
      <td>320</td>
      <td>525</td>
      <td>1188</td>
      <td>0.042088</td>
      <td>0.679104</td>
      <td>1.137500</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2004</td>
      <td>1102</td>
      <td>1404</td>
      <td>466</td>
      <td>913</td>
      <td>192</td>
      <td>475</td>
      <td>280</td>
      <td>387</td>
      <td>1071</td>
      <td>...</td>
      <td>0.691318</td>
      <td>66</td>
      <td>371</td>
      <td>218</td>
      <td>302</td>
      <td>466</td>
      <td>1130</td>
      <td>0.058407</td>
      <td>0.653169</td>
      <td>1.228477</td>
    </tr>
    <tr>
      <th>653</th>
      <td>2005</td>
      <td>1102</td>
      <td>1097</td>
      <td>378</td>
      <td>787</td>
      <td>146</td>
      <td>376</td>
      <td>195</td>
      <td>266</td>
      <td>824</td>
      <td>...</td>
      <td>0.740066</td>
      <td>50</td>
      <td>396</td>
      <td>255</td>
      <td>292</td>
      <td>478</td>
      <td>1161</td>
      <td>0.043066</td>
      <td>0.637681</td>
      <td>1.356164</td>
    </tr>
    <tr>
      <th>982</th>
      <td>2006</td>
      <td>1102</td>
      <td>1430</td>
      <td>489</td>
      <td>991</td>
      <td>201</td>
      <td>478</td>
      <td>251</td>
      <td>332</td>
      <td>1143</td>
      <td>...</td>
      <td>0.683128</td>
      <td>59</td>
      <td>397</td>
      <td>224</td>
      <td>306</td>
      <td>431</td>
      <td>1283</td>
      <td>0.045986</td>
      <td>0.646580</td>
      <td>1.297386</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>2007</td>
      <td>1102</td>
      <td>1591</td>
      <td>531</td>
      <td>1034</td>
      <td>211</td>
      <td>473</td>
      <td>318</td>
      <td>418</td>
      <td>1168</td>
      <td>...</td>
      <td>0.758788</td>
      <td>44</td>
      <td>451</td>
      <td>188</td>
      <td>305</td>
      <td>455</td>
      <td>1457</td>
      <td>0.030199</td>
      <td>0.659357</td>
      <td>1.478689</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>




```python
cumulative_stats_for_team_each_year.dtypes[0:33]
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
    dtype: object




```python
cumulative_stats_for_team_each_year.dtypes[34:67]
```




    LFTP    float64
    fgp     float64
    fg3p    float64
    ftp     float64
    dtype: object




```python
cumulative_stats_for_team_each_year.dtypes[68:100]
```




    Series([], dtype: object)




```python
cumulative_stats_for_team_each_year = (
    combine_both_winning_losing_games_stats
    .sort_values(['WTeamID','Season'])
    .groupby(['WTeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(Season = combine_both_winning_losing_games_stats.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_both_winning_losing_games_stats.WTeamID.values))
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
    # rebounds cumsum stats
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    # assists, turnovers, steals, blocks and personal fouls
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
    # win or lose by how many points
    .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
    .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
    .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
    .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
    .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
    .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
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
      <th>LFTP</th>
      <th>fgp</th>
      <th>fg3p</th>
      <th>ftp</th>
      <th>lose_rate</th>
      <th>win_score_by</th>
      <th>lose_score_by</th>
      <th>expectation_per_game</th>
      <th>avg_win_score_by</th>
      <th>avg_lose_score_by</th>
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
      <td>0.647826</td>
      <td>0.481149</td>
      <td>0.375643</td>
      <td>0.651357</td>
      <td>0.571429</td>
      <td>187</td>
      <td>-180</td>
      <td>0.250000</td>
      <td>15.583333</td>
      <td>-11.250000</td>
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
      <td>0.639576</td>
      <td>0.481886</td>
      <td>0.378423</td>
      <td>0.678999</td>
      <td>0.392857</td>
      <td>520</td>
      <td>-240</td>
      <td>5.000000</td>
      <td>15.294118</td>
      <td>-10.909091</td>
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
      <td>0.661616</td>
      <td>0.469388</td>
      <td>0.373236</td>
      <td>0.693374</td>
      <td>0.400000</td>
      <td>793</td>
      <td>-336</td>
      <td>5.376471</td>
      <td>15.549020</td>
      <td>-9.882353</td>
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
      <td>0.669456</td>
      <td>0.472430</td>
      <td>0.378968</td>
      <td>0.706192</td>
      <td>0.353982</td>
      <td>1080</td>
      <td>-373</td>
      <td>6.256637</td>
      <td>14.794521</td>
      <td>-9.325000</td>
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
      <td>0.688119</td>
      <td>0.475389</td>
      <td>0.384158</td>
      <td>0.719221</td>
      <td>0.335664</td>
      <td>1503</td>
      <td>-448</td>
      <td>7.377622</td>
      <td>15.821053</td>
      <td>-9.333333</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 100 columns</p>
</div>




```python
from aggregate_function import build_features_table, win_rate_type_of_location
```


```python
test_features = build_features_table.BuildFeaturesTable("data/DataFiles/RegularSeasonDetailedResults.csv")
```


```python
win_rate_location_test = win_rate_type_of_location.WinRateTypeLocation("data/DataFiles/RegularSeasonDetailedResults.csv")
```


```python
win_rate_location_test.win_rate_cum_away_df.head()
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
      <th>win_rate_away</th>
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
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11801</th>
      <td>2015</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>0.142857</td>
      <td>1101</td>
      <td>0.083333</td>
    </tr>
    <tr>
      <th>12821</th>
      <td>2016</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>0.428571</td>
      <td>1101</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>13839</th>
      <td>2017</td>
      <td>7.0</td>
      <td>21.0</td>
      <td>0.873016</td>
      <td>1101</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.428571</td>
      <td>1102</td>
      <td>0.428571</td>
    </tr>
  </tbody>
</table>
</div>




```python
win_rate_location_test.win_rate_cum_home_df.head()
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
      <th>win_rate_home</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10784</th>
      <td>2014</td>
      <td>2.0</td>
      <td>14.0</td>
      <td>0.125000</td>
      <td>1101</td>
      <td>0.125000</td>
    </tr>
    <tr>
      <th>11802</th>
      <td>2015</td>
      <td>6.0</td>
      <td>28.0</td>
      <td>0.347222</td>
      <td>1101</td>
      <td>0.176471</td>
    </tr>
    <tr>
      <th>12822</th>
      <td>2016</td>
      <td>12.0</td>
      <td>40.0</td>
      <td>0.680556</td>
      <td>1101</td>
      <td>0.230769</td>
    </tr>
    <tr>
      <th>13840</th>
      <td>2017</td>
      <td>17.0</td>
      <td>51.0</td>
      <td>0.993056</td>
      <td>1101</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>0.473684</td>
      <td>1102</td>
      <td>0.473684</td>
    </tr>
  </tbody>
</table>
</div>




```python
win_rate_location_test.win_rate_cum_neutral_df.head()
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
      <th>win_rate_neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11803</th>
      <td>2015</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.666667</td>
      <td>1101</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>12823</th>
      <td>2016</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.166667</td>
      <td>1101</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>1102</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>945</th>
      <td>2004</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.500000</td>
      <td>1102</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>1885</th>
      <td>2005</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>1.000000</td>
      <td>1102</td>
      <td>0.375000</td>
    </tr>
  </tbody>
</table>
</div>


