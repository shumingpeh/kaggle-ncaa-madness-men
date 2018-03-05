
___
This notebook decides on the intermediate variables being used


```python
import pandas as pd
import numpy as np
import scipy
from sklearn import *
```

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
      <th>losing_num_counts</th>
      <th>total_score</th>
      <th>total_opponent_score</th>
      <th>total_fgm</th>
      <th>total_fga</th>
      <th>total_fg3m</th>
      <th>total_fg3a</th>
      <th>total_ftm</th>
      <th>total_fta</th>
      <th>win_rate</th>
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
      <td>16</td>
      <td>1603</td>
      <td>1596</td>
      <td>536</td>
      <td>1114</td>
      <td>219</td>
      <td>583</td>
      <td>312</td>
      <td>479</td>
      <td>0.428571</td>
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
      <td>6</td>
      <td>1685</td>
      <td>1412</td>
      <td>568</td>
      <td>1177</td>
      <td>237</td>
      <td>622</td>
      <td>312</td>
      <td>440</td>
      <td>0.785714</td>
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
      <td>12</td>
      <td>1776</td>
      <td>1599</td>
      <td>621</td>
      <td>1384</td>
      <td>258</td>
      <td>708</td>
      <td>276</td>
      <td>379</td>
      <td>0.586207</td>
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
      <td>6</td>
      <td>1778</td>
      <td>1528</td>
      <td>614</td>
      <td>1276</td>
      <td>241</td>
      <td>607</td>
      <td>309</td>
      <td>414</td>
      <td>0.785714</td>
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
      <td>8</td>
      <td>2055</td>
      <td>1707</td>
      <td>684</td>
      <td>1408</td>
      <td>272</td>
      <td>674</td>
      <td>415</td>
      <td>546</td>
      <td>0.733333</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




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
- win rate for home court
- win rate for away court
- win rate for neutral court
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




```python
metrics.confusion_matrix(test_df.yhat,test_df.prediction_results)
```




    array([[180,  88],
           [ 68, 200]])



http://blog.yhat.com/posts/roc-curves.html


```python
pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv").head()
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
pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv").dtypes
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


