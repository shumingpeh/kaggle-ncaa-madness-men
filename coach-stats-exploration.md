
___
this notebook gets the coach stats


```python
import pandas as pd
import numpy as np

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

pd.set_option("display.max_columns",50)
```


<style>.container { width:85% !important; }</style>


## Summary of notebook data manipulation
1. years/season of experience
1. number of playoff made
1. number of championship won
1. win rate
    - regular
    - post
    - overall

## Read data


```python
raw_data_regularseason = pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv")
raw_data_coach = pd.read_csv('data/DataFiles/TeamCoaches.csv')
raw_data_postseason = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')

```

## Get number of season experience


```python
# get season max numbner of days
season_max_days = (
    raw_data_coach
    .groupby(['Season'])
    .agg({"LastDayNum":"max"})
    .reset_index()
    .rename(columns={"LastDayNum":"season_max_days"})
)

# get number of season for each coach for each year
num_days_coach_for_season = (
    raw_data_coach
    .pipe(lambda x:x.assign(daysexp = x.LastDayNum-x.FirstDayNum))
    .merge(season_max_days, how='left',on=['Season'])
    .pipe(lambda x:x.assign(num_season = x.daysexp/x.season_max_days))
    .sort_values(['CoachName','Season'])
)
num_days_coach_for_season.head()
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
      <th>daysexp</th>
      <th>season_max_days</th>
      <th>num_season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>1985</td>
      <td>1224</td>
      <td>0</td>
      <td>154</td>
      <td>a_b_williamson</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>1986</td>
      <td>1224</td>
      <td>0</td>
      <td>154</td>
      <td>a_b_williamson</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>669</th>
      <td>1987</td>
      <td>1224</td>
      <td>0</td>
      <td>154</td>
      <td>a_b_williamson</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>963</th>
      <td>1988</td>
      <td>1224</td>
      <td>0</td>
      <td>154</td>
      <td>a_b_williamson</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>1989</td>
      <td>1224</td>
      <td>0</td>
      <td>154</td>
      <td>a_b_williamson</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Get cumulative number of seasons experience


```python
# get cumulative years of experience
cum_num_days_coach_for_season = (
    num_days_coach_for_season
    .sort_values(['CoachName','Season'])
    .groupby(['CoachName'])
    .cumsum()
    .pipe(lambda x:x.assign(Season = num_days_coach_for_season.Season.values))
    .pipe(lambda x:x.assign(TeamID = num_days_coach_for_season.TeamID.values))
)

cum_num_days_coach_for_season.head()
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
      <th>daysexp</th>
      <th>season_max_days</th>
      <th>num_season</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>95</th>
      <td>1985</td>
      <td>1224</td>
      <td>0</td>
      <td>154</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>1986</td>
      <td>1224</td>
      <td>0</td>
      <td>308</td>
      <td>308</td>
      <td>308</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>669</th>
      <td>1987</td>
      <td>1224</td>
      <td>0</td>
      <td>462</td>
      <td>462</td>
      <td>462</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>963</th>
      <td>1988</td>
      <td>1224</td>
      <td>0</td>
      <td>616</td>
      <td>616</td>
      <td>616</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>1989</td>
      <td>1224</td>
      <td>0</td>
      <td>770</td>
      <td>770</td>
      <td>770</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



## Assign one coach to one season
- check which teams have more than one coach in one season
    - the coach with more days of coaching will be credited for the season


```python
final_coach_for_season = (
    num_days_coach_for_season
    .groupby(['Season','TeamID'])
    .agg({"CoachName":"count"})
    .reset_index()
#     .query("CoachName > 1")
    .rename(columns={"CoachName":"coach_counts"})
    .merge(num_days_coach_for_season,how='left',on=['Season','TeamID'])
    .pipe(lambda x:x.assign(final_coach = np.where(x.num_season >= 0.5, x.CoachName, "ignore")))
    [['Season','TeamID','final_coach']]
)

final_coach_for_season.head()
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
      <th>final_coach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1102</td>
      <td>reggie_minton</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>bob_huggins</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>wimp_sanderson</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>james_oliver</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>davey_whitney</td>
    </tr>
  </tbody>
</table>
</div>



## Get number of playoffs made for coaches
- check if team made to playoff to season
- final coach gets the credit


```python
# get teams for post season
teams_for_postseason = (
    raw_data_postseason
    .groupby(['Season','WTeamID'])
    .agg({"NumOT":"count"})
    .reset_index()
    .append(raw_data_postseason[['Season','LTeamID','NumOT']].rename(columns={"LTeamID":"WTeamID"}))
    .groupby(['Season','WTeamID'])
    .agg({"NumOT":"count"})
    .reset_index()
    .rename(columns={"NumOT":"is_playoff"})
    .pipe(lambda x:x.assign(is_playoff = 1))
    .rename(columns={"WTeamID":"TeamID"})
)

teams_for_postseason.head()
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
      <th>is_playoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1104</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1112</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1116</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1120</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1130</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# join postseason to final coach
final_coach_with_postseason_each_year = (
    final_coach_for_season
    .merge(teams_for_postseason,how='left',on=['Season','TeamID'])
    .fillna(0)
)
```

## Get number of championships won for coaches
- check which team won championship
- final coach gets the credit


```python
championship_team = (
    raw_data_postseason
    .merge(season_max_days,how='left',on=['Season'])
    .query("DayNum == season_max_days")
    .groupby(['Season','WTeamID'])
    .agg({"NumOT":"count"})
    .reset_index()
    .rename(columns={"NumOT":"is_champion","WTeamID":"TeamID"})
#     .merge(final_coach_with_season_each_year,how='left',on=['Season','TeamID'])
)


final_coach_with_postseason_champion_each_year = (
    final_coach_with_postseason_each_year
    .merge(championship_team,how='left',on=['Season','TeamID'])
    .fillna(0)
)

final_coach_with_postseason_champion_each_year.head()
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
      <th>final_coach</th>
      <th>is_playoff</th>
      <th>is_champion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1102</td>
      <td>reggie_minton</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>bob_huggins</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>wimp_sanderson</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>james_oliver</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>davey_whitney</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Get win rate for coach during regular season
- get up till daynum of the coach in the team of the season
- get number of games won and lost, so that reconciling with cumulative table will be okay


```python
# get winning games for coaches
games_won_for_coaches = (
    raw_data_regularseason
    [['Season','DayNum','WTeamID']]
    # merge for winning team
    .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
           how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"FirstDayNum":"FirstDayNum_win","LastDayNum":"LastDayNum_win","CoachName":"CoachName_win","TeamID":"TeamID_win"})
#     # merge for losing team
#     .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
#            how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
#     .rename(columns={"FirstDayNum":"FirstDayNum_lose","LastDayNum":"LastDayNum_lose","CoachName":"CoachName_lose","TeamID":"TeamID_lose"})
    .pipe(lambda x:x.assign(which_coach_for_win = np.where((x.FirstDayNum_win <= x.DayNum) & (x.LastDayNum_win >= x.DayNum),1,0)))
    .query("which_coach_for_win != 0")
    .groupby(['Season','CoachName_win','WTeamID'])
    .agg({"which_coach_for_win":"sum"})
    .reset_index()
)

games_won_for_coaches.head()
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
      <th>CoachName_win</th>
      <th>WTeamID</th>
      <th>which_coach_for_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>al_skinner</td>
      <td>1130</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>al_walker</td>
      <td>1127</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>andy_stoglin</td>
      <td>1238</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>armond_hill</td>
      <td>1162</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>barry_collier</td>
      <td>1304</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get losing games for coaches
games_lose_for_coaches = (
    raw_data_regularseason
    [['Season','DayNum','LTeamID']]
#     # merge for winning team
#     .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
#            how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
#     .rename(columns={"FirstDayNum":"FirstDayNum_win","LastDayNum":"LastDayNum_win","CoachName":"CoachName_win","TeamID":"TeamID_win"})
    # merge for losing team
    .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
           how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"FirstDayNum":"FirstDayNum_lose","LastDayNum":"LastDayNum_lose","CoachName":"CoachName_lose","TeamID":"TeamID_lose"})
    .pipe(lambda x:x.assign(which_coach_for_lose = np.where((x.FirstDayNum_lose <= x.DayNum) & (x.LastDayNum_lose >= x.DayNum),1,0)))
    .query("which_coach_for_lose != 0")
    .groupby(['Season','CoachName_lose','LTeamID'])
    .agg({"which_coach_for_lose":"sum"})
    .reset_index()
)

games_lose_for_coaches.head()
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
      <th>CoachName_lose</th>
      <th>LTeamID</th>
      <th>which_coach_for_lose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>al_skinner</td>
      <td>1130</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>al_walker</td>
      <td>1127</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>andy_stoglin</td>
      <td>1238</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>armond_hill</td>
      <td>1162</td>
      <td>25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>barry_collier</td>
      <td>1304</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
# combine both losing and winning games
combine_regular_games_won_lose = (
    games_lose_for_coaches
    .merge(games_won_for_coaches,how='left',left_on=['Season','LTeamID','CoachName_lose'],right_on=['Season','WTeamID','CoachName_win'])
    .pipe(lambda x:x.assign(win_rate_regular = x.which_coach_for_win/(x.which_coach_for_win + x.which_coach_for_lose)))
    .drop(['CoachName_win','WTeamID'],1)
    .rename(columns={"CoachName_lose":"CoachName","LTeamID":"TeamID","which_coach_for_lose":"games_lost","which_coach_for_win":"games_won"})
)

combine_regular_games_won_lose.head()
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
      <th>CoachName</th>
      <th>TeamID</th>
      <th>games_lost</th>
      <th>games_won</th>
      <th>win_rate_regular</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>al_skinner</td>
      <td>1130</td>
      <td>11</td>
      <td>18.0</td>
      <td>0.620690</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>al_walker</td>
      <td>1127</td>
      <td>13</td>
      <td>14.0</td>
      <td>0.518519</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>andy_stoglin</td>
      <td>1238</td>
      <td>18</td>
      <td>10.0</td>
      <td>0.357143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>armond_hill</td>
      <td>1162</td>
      <td>25</td>
      <td>2.0</td>
      <td>0.074074</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>barry_collier</td>
      <td>1304</td>
      <td>18</td>
      <td>11.0</td>
      <td>0.379310</td>
    </tr>
  </tbody>
</table>
</div>



## Get win rate for coach during post season


```python
# get winning games for coaches
post_games_won_for_coaches = (
    raw_data_postseason
    [['Season','DayNum','WTeamID']]
    # merge for winning team
    .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
           how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"FirstDayNum":"FirstDayNum_win","LastDayNum":"LastDayNum_win","CoachName":"CoachName_win","TeamID":"TeamID_win"})
#     # merge for losing team
#     .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
#            how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
#     .rename(columns={"FirstDayNum":"FirstDayNum_lose","LastDayNum":"LastDayNum_lose","CoachName":"CoachName_lose","TeamID":"TeamID_lose"})
    .pipe(lambda x:x.assign(which_coach_for_win = np.where((x.FirstDayNum_win <= x.DayNum) & (x.LastDayNum_win >= x.DayNum),1,0)))
    .query("which_coach_for_win != 0")
    .groupby(['Season','CoachName_win','WTeamID'])
    .agg({"which_coach_for_win":"sum"})
    .reset_index()
)

post_games_won_for_coaches.head()
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
      <th>CoachName_win</th>
      <th>WTeamID</th>
      <th>which_coach_for_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>andy_russo</td>
      <td>1256</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>bill_frieder</td>
      <td>1276</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>billy_tubbs</td>
      <td>1328</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>bob_donewald</td>
      <td>1229</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>bobby_cremins</td>
      <td>1210</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get losing games for coaches
post_games_lose_for_coaches = (
    raw_data_postseason
    [['Season','DayNum','LTeamID']]
#     # merge for winning team
#     .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
#            how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
#     .rename(columns={"FirstDayNum":"FirstDayNum_win","LastDayNum":"LastDayNum_win","CoachName":"CoachName_win","TeamID":"TeamID_win"})
    # merge for losing team
    .merge(num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
           how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"FirstDayNum":"FirstDayNum_lose","LastDayNum":"LastDayNum_lose","CoachName":"CoachName_lose","TeamID":"TeamID_lose"})
    .pipe(lambda x:x.assign(which_coach_for_lose = np.where((x.FirstDayNum_lose <= x.DayNum) & (x.LastDayNum_lose >= x.DayNum),1,0)))
    .query("which_coach_for_lose != 0")
    .groupby(['Season','CoachName_lose','LTeamID'])
    .agg({"which_coach_for_lose":"sum"})
    .reset_index()
)

post_games_lose_for_coaches.head()
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
      <th>CoachName_lose</th>
      <th>LTeamID</th>
      <th>which_coach_for_lose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>andy_russo</td>
      <td>1256</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>bill_bibb</td>
      <td>1273</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>bill_frieder</td>
      <td>1276</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>billy_tubbs</td>
      <td>1328</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>bob_donewald</td>
      <td>1229</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# combine both losing and winning post games
combine_post_games_won_lose = (
    post_games_lose_for_coaches
    .merge(post_games_won_for_coaches,how='left',left_on=['Season','LTeamID','CoachName_lose'],right_on=['Season','WTeamID','CoachName_win'])
    .pipe(lambda x:x.assign(win_rate_post = x.which_coach_for_win/(x.which_coach_for_win + x.which_coach_for_lose)))
    .drop(['CoachName_win','WTeamID'],1)
    .rename(columns={"CoachName_lose":"CoachName","LTeamID":"TeamID","which_coach_for_lose":"post_games_lost","which_coach_for_win":"post_games_won"})
    .fillna(0)
)

combine_post_games_won_lose.head()
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
      <th>CoachName</th>
      <th>TeamID</th>
      <th>post_games_lost</th>
      <th>post_games_won</th>
      <th>win_rate_post</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>andy_russo</td>
      <td>1256</td>
      <td>1</td>
      <td>2.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>bill_bibb</td>
      <td>1273</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>bill_frieder</td>
      <td>1276</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>billy_tubbs</td>
      <td>1328</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>bob_donewald</td>
      <td>1229</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>



## Get overall win rate for coaches


```python
overall_win_rate_for_coaches = (
    combine_post_games_won_lose
    .merge(combine_regular_games_won_lose,how='left',on=['Season','CoachName','TeamID'])
    .fillna(0)
    .pipe(lambda x:x.assign(overall_games_won = x.post_games_won + x.games_won))
    .pipe(lambda x:x.assign(overall_games_lost = x.post_games_lost + x.games_lost))
    .pipe(lambda x:x.assign(win_rate_overall = x.overall_games_won/(x.overall_games_won + x.overall_games_lost)))
)

overall_win_rate_for_coaches.tail()
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
      <th>CoachName</th>
      <th>TeamID</th>
      <th>post_games_lost</th>
      <th>post_games_won</th>
      <th>win_rate_post</th>
      <th>games_lost</th>
      <th>games_won</th>
      <th>win_rate_regular</th>
      <th>overall_games_won</th>
      <th>overall_games_lost</th>
      <th>win_rate_overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2112</th>
      <td>2017</td>
      <td>tim_cluess</td>
      <td>1233</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>0.647059</td>
      <td>22.0</td>
      <td>13.0</td>
      <td>0.628571</td>
    </tr>
    <tr>
      <th>2113</th>
      <td>2017</td>
      <td>tim_jankovich</td>
      <td>1374</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>30.0</td>
      <td>0.882353</td>
      <td>30.0</td>
      <td>5.0</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>2114</th>
      <td>2017</td>
      <td>tom_izzo</td>
      <td>1277</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>14.0</td>
      <td>19.0</td>
      <td>0.575758</td>
      <td>20.0</td>
      <td>15.0</td>
      <td>0.571429</td>
    </tr>
    <tr>
      <th>2115</th>
      <td>2017</td>
      <td>tony_bennett</td>
      <td>1438</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>10.0</td>
      <td>22.0</td>
      <td>0.687500</td>
      <td>23.0</td>
      <td>11.0</td>
      <td>0.676471</td>
    </tr>
    <tr>
      <th>2116</th>
      <td>2017</td>
      <td>will_wade</td>
      <td>1433</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>0.764706</td>
      <td>26.0</td>
      <td>9.0</td>
      <td>0.742857</td>
    </tr>
  </tbody>
</table>
</div>



## Combine all coach stats into one master table


```python
final_coach_stats_table = (
    num_days_coach_for_season
    .merge(final_coach_with_postseason_champion_each_year,how='left',left_on=['Season','TeamID','CoachName'],right_on=['Season','TeamID','final_coach'])
    .fillna(0)
    .merge(overall_win_rate_for_coaches,how='left',on=['Season','TeamID','CoachName'])
    .fillna(0)
    .drop(['final_coach','FirstDayNum','LastDayNum'],1)
    .sort_values(['CoachName','Season'])
)

final_coach_stats_table.tail()
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
      <th>CoachName</th>
      <th>daysexp</th>
      <th>season_max_days</th>
      <th>num_season</th>
      <th>is_playoff</th>
      <th>is_champion</th>
      <th>post_games_lost</th>
      <th>post_games_won</th>
      <th>win_rate_post</th>
      <th>games_lost</th>
      <th>games_won</th>
      <th>win_rate_regular</th>
      <th>overall_games_won</th>
      <th>overall_games_lost</th>
      <th>win_rate_overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10987</th>
      <td>2014</td>
      <td>1119</td>
      <td>zach_spiker</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10988</th>
      <td>2015</td>
      <td>1119</td>
      <td>zach_spiker</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10989</th>
      <td>2016</td>
      <td>1119</td>
      <td>zach_spiker</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10990</th>
      <td>2017</td>
      <td>1180</td>
      <td>zach_spiker</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10991</th>
      <td>2018</td>
      <td>1180</td>
      <td>zach_spiker</td>
      <td>77</td>
      <td>77</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Cumulative coach stats for master table


```python
cumulative_final_coach_stats_table = (
    final_coach_stats_table
    .groupby(['CoachName'])
    .cumsum()
    .pipe(lambda x:x.assign(Season = final_coach_stats_table.Season.values))
    .pipe(lambda x:x.assign(TeamID = final_coach_stats_table.TeamID.values))
    .pipe(lambda x:x.assign(CoachName = final_coach_stats_table.CoachName.values))
    .pipe(lambda x:x.assign(win_rate_post = x.post_games_won/(x.post_games_won + x.post_games_lost)))
    .fillna(0)
    .pipe(lambda x:x.assign(win_rate_regular = x.games_won/(x.games_won + x.games_lost)))
    .fillna(0)
    .pipe(lambda x:x.assign(win_rate_overall = x.overall_games_won/(x.overall_games_won + x.overall_games_lost)))
    .fillna(0)
)
cumulative_final_coach_stats_table.head()
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
      <th>daysexp</th>
      <th>season_max_days</th>
      <th>num_season</th>
      <th>is_playoff</th>
      <th>is_champion</th>
      <th>post_games_lost</th>
      <th>post_games_won</th>
      <th>win_rate_post</th>
      <th>games_lost</th>
      <th>games_won</th>
      <th>win_rate_regular</th>
      <th>overall_games_won</th>
      <th>overall_games_lost</th>
      <th>win_rate_overall</th>
      <th>CoachName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1224</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1986</td>
      <td>1224</td>
      <td>308</td>
      <td>308</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1987</td>
      <td>1224</td>
      <td>462</td>
      <td>462</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1988</td>
      <td>1224</td>
      <td>616</td>
      <td>616</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1989</td>
      <td>1224</td>
      <td>770</td>
      <td>770</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
  </tbody>
</table>
</div>




```python
coach_file = 'data/DataFiles/TeamCoaches.csv'
regularseason_file = 'data/DataFiles/RegularSeasonDetailedResults.csv'
postseason_file = 'data/DataFiles/NCAATourneyCompactResults.csv'
```


```python
from aggregate_function import coach_stats
```


```python
testing_df = coach_stats.CoachStats(coach_file,regularseason_file,postseason_file)
```


```python
testing_df.cumulative_final_coach_stats_table.head()
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
      <th>daysexp</th>
      <th>season_max_days</th>
      <th>num_season</th>
      <th>is_playoff</th>
      <th>is_champion</th>
      <th>post_games_lost</th>
      <th>post_games_won</th>
      <th>win_rate_post</th>
      <th>games_lost</th>
      <th>games_won</th>
      <th>win_rate_regular</th>
      <th>overall_games_won</th>
      <th>overall_games_lost</th>
      <th>win_rate_overall</th>
      <th>CoachName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1224</td>
      <td>154</td>
      <td>154</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1986</td>
      <td>1224</td>
      <td>308</td>
      <td>308</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1987</td>
      <td>1224</td>
      <td>462</td>
      <td>462</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1988</td>
      <td>1224</td>
      <td>616</td>
      <td>616</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1989</td>
      <td>1224</td>
      <td>770</td>
      <td>770</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>a_b_williamson</td>
    </tr>
  </tbody>
</table>
</div>



## Concluding remarks
- need to clean up the way class file is written, at the moment, its a direct copy and paste into the class file to save time
