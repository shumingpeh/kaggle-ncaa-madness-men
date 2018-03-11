

```python
import pandas as pd
import numpy as np

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

pd.set_option("display.max_columns",50)
```


<style>.container { width:85% !important; }</style>



```python
regular_df = pd.read_csv('data/DataFiles/RegularSeasonDetailedResults.csv')
regular_df.columns = [i.lower() for i in regular_df.columns]
team_df =  pd.read_csv('data/DataFiles/Teams.csv')
team_df.columns = [i.lower() for i in team_df.columns]
teamcoach_df = pd.read_csv('data/DataFiles/TeamCoaches.csv')
teamcoach_df.columns = [i.lower() for i in teamcoach_df.columns]
post_df = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')
post_df.columns = [i.lower() for i in post_df.columns]
```


```python
teamcoach_df.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>coachname</th>
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
combine = teamcoach_df.pipe(lambda x:x.assign(daysexp = x.lastdaynum-x.firstdaynum))
```


```python
combine.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>coachname</th>
      <th>daysexp</th>
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
      <td>154</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>0</td>
      <td>154</td>
      <td>bob_huggins</td>
      <td>154</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>0</td>
      <td>154</td>
      <td>wimp_sanderson</td>
      <td>154</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>0</td>
      <td>154</td>
      <td>james_oliver</td>
      <td>154</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>0</td>
      <td>154</td>
      <td>davey_whitney</td>
      <td>154</td>
    </tr>
  </tbody>
</table>
</div>




```python

```




    10992




```python
post_df.head()
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
      <th>season</th>
      <th>daynum</th>
      <th>wteamid</th>
      <th>wscore</th>
      <th>lteamid</th>
      <th>lscore</th>
      <th>wloc</th>
      <th>numot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>136</td>
      <td>1116</td>
      <td>63</td>
      <td>1234</td>
      <td>54</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>136</td>
      <td>1120</td>
      <td>59</td>
      <td>1345</td>
      <td>58</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>136</td>
      <td>1207</td>
      <td>68</td>
      <td>1250</td>
      <td>43</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>136</td>
      <td>1229</td>
      <td>58</td>
      <td>1425</td>
      <td>55</td>
      <td>N</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>136</td>
      <td>1242</td>
      <td>49</td>
      <td>1325</td>
      <td>38</td>
      <td>N</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
post_df.daynum.max()
```




    154




```python
champions_df = post_df.query('daynum==154')[['season','daynum','wteamid']]
```


```python
champions_df.columns = [['season','lastdaynum','teamid']]
champions_df['champions'] = [1 for i in champions_df.teamid]
```


```python
champions_df.head()
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
      <th>season</th>
      <th>lastdaynum</th>
      <th>teamid</th>
      <th>champions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>1985</td>
      <td>154</td>
      <td>1437</td>
      <td>1</td>
    </tr>
    <tr>
      <th>125</th>
      <td>1986</td>
      <td>154</td>
      <td>1257</td>
      <td>1</td>
    </tr>
    <tr>
      <th>188</th>
      <td>1987</td>
      <td>154</td>
      <td>1231</td>
      <td>1</td>
    </tr>
    <tr>
      <th>251</th>
      <td>1988</td>
      <td>154</td>
      <td>1242</td>
      <td>1</td>
    </tr>
    <tr>
      <th>314</th>
      <td>1989</td>
      <td>154</td>
      <td>1276</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine2 = pd.merge(combine, champions_df, how='left',on=['season','lastdaynum','teamid'])
```


```python
combine2.champions = combine2.champions.fillna(0)
```


```python
combine2.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>coachname</th>
      <th>daysexp</th>
      <th>champions</th>
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
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>0</td>
      <td>154</td>
      <td>bob_huggins</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>0</td>
      <td>154</td>
      <td>wimp_sanderson</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>0</td>
      <td>154</td>
      <td>james_oliver</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>0</td>
      <td>154</td>
      <td>davey_whitney</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
intermediate = pd.DataFrame({'season':post_df.groupby(['season'])['daynum'].min().index.values, 'playoff_startday':[v for i,v in enumerate(post_df.groupby(['season'])['daynum'].min())]})
```


```python
postmod_df = pd.merge(post_df,intermediate,how='left',on='season')
```


```python
postmod_df.head()
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
      <th>season</th>
      <th>daynum</th>
      <th>wteamid</th>
      <th>wscore</th>
      <th>lteamid</th>
      <th>lscore</th>
      <th>wloc</th>
      <th>numot</th>
      <th>playoff_startday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>136</td>
      <td>1116</td>
      <td>63</td>
      <td>1234</td>
      <td>54</td>
      <td>N</td>
      <td>0</td>
      <td>136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>136</td>
      <td>1120</td>
      <td>59</td>
      <td>1345</td>
      <td>58</td>
      <td>N</td>
      <td>0</td>
      <td>136</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>136</td>
      <td>1207</td>
      <td>68</td>
      <td>1250</td>
      <td>43</td>
      <td>N</td>
      <td>0</td>
      <td>136</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>136</td>
      <td>1229</td>
      <td>58</td>
      <td>1425</td>
      <td>55</td>
      <td>N</td>
      <td>0</td>
      <td>136</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>136</td>
      <td>1242</td>
      <td>49</td>
      <td>1325</td>
      <td>38</td>
      <td>N</td>
      <td>0</td>
      <td>136</td>
    </tr>
  </tbody>
</table>
</div>




```python
postmod_df.shape
```




    (2117, 9)




```python
intermediate = postmod_df.groupby(['season','wteamid'])['daynum'].count().reset_index()
intermediate['count'] = [1 for i in intermediate.daynum]
```


```python
intermediate2 = postmod_df.groupby(['season','lteamid'])['daynum'].count().reset_index()
intermediate2['count'] = [1 for i in intermediate2.daynum]
intermediate2.columns = [['season','wteamid','daynum','count']]
```


```python
intermediate3 = pd.concat([intermediate, intermediate2], axis =0)
```


```python
intermediate3.head()
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
      <th>season</th>
      <th>wteamid</th>
      <th>daynum</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1104</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1116</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1120</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1130</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1181</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
intermediate3.drop_duplicates(subset=['wteamid','season'], inplace = True)
```


```python
intermediate3.drop('daynum', axis=1, inplace=True)
```


```python
intermediate3.columns = [['season','teamid','playoff']]
```


```python
intermediate3.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>playoff</th>
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
      <td>1116</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1120</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1130</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1181</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine2.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>coachname</th>
      <th>daysexp</th>
      <th>champions</th>
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
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>0</td>
      <td>154</td>
      <td>bob_huggins</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>0</td>
      <td>154</td>
      <td>wimp_sanderson</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>0</td>
      <td>154</td>
      <td>james_oliver</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>0</td>
      <td>154</td>
      <td>davey_whitney</td>
      <td>154</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine3 = pd.merge(combine2, intermediate3, how='left', on=['season','teamid'])
```


```python
combine3.playoff = combine3.playoff.fillna(0)
```


```python
regular_df.head()
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
      <th>season</th>
      <th>daynum</th>
      <th>wteamid</th>
      <th>wscore</th>
      <th>lteamid</th>
      <th>lscore</th>
      <th>wloc</th>
      <th>numot</th>
      <th>wfgm</th>
      <th>wfga</th>
      <th>wfgm3</th>
      <th>wfga3</th>
      <th>wftm</th>
      <th>wfta</th>
      <th>wor</th>
      <th>wdr</th>
      <th>wast</th>
      <th>wto</th>
      <th>wstl</th>
      <th>wblk</th>
      <th>wpf</th>
      <th>lfgm</th>
      <th>lfga</th>
      <th>lfgm3</th>
      <th>lfga3</th>
      <th>lftm</th>
      <th>lfta</th>
      <th>lor</th>
      <th>ldr</th>
      <th>last</th>
      <th>lto</th>
      <th>lstl</th>
      <th>lblk</th>
      <th>lpf</th>
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
      <td>3</td>
      <td>14</td>
      <td>11</td>
      <td>18</td>
      <td>14</td>
      <td>24</td>
      <td>13</td>
      <td>23</td>
      <td>7</td>
      <td>1</td>
      <td>22</td>
      <td>22</td>
      <td>53</td>
      <td>2</td>
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
      <td>8</td>
      <td>20</td>
      <td>10</td>
      <td>19</td>
      <td>15</td>
      <td>28</td>
      <td>16</td>
      <td>13</td>
      <td>4</td>
      <td>4</td>
      <td>18</td>
      <td>24</td>
      <td>67</td>
      <td>6</td>
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
      <td>8</td>
      <td>18</td>
      <td>17</td>
      <td>29</td>
      <td>17</td>
      <td>26</td>
      <td>15</td>
      <td>10</td>
      <td>5</td>
      <td>2</td>
      <td>25</td>
      <td>22</td>
      <td>73</td>
      <td>3</td>
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
      <td>3</td>
      <td>9</td>
      <td>17</td>
      <td>31</td>
      <td>6</td>
      <td>19</td>
      <td>11</td>
      <td>12</td>
      <td>14</td>
      <td>2</td>
      <td>18</td>
      <td>18</td>
      <td>49</td>
      <td>6</td>
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
      <td>6</td>
      <td>14</td>
      <td>11</td>
      <td>13</td>
      <td>17</td>
      <td>22</td>
      <td>12</td>
      <td>14</td>
      <td>4</td>
      <td>4</td>
      <td>20</td>
      <td>24</td>
      <td>62</td>
      <td>6</td>
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
</div>




```python
teamcoach_df2.query('lastdaynum<154 & season==2005')
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
      <th>season</th>
      <th>wteamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>wcoachname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6240</th>
      <td>2005</td>
      <td>1236</td>
      <td>0</td>
      <td>72</td>
      <td>doug_noll</td>
    </tr>
    <tr>
      <th>6397</th>
      <td>2005</td>
      <td>1409</td>
      <td>0</td>
      <td>53</td>
      <td>john_phillips</td>
    </tr>
    <tr>
      <th>6414</th>
      <td>2005</td>
      <td>1425</td>
      <td>0</td>
      <td>34</td>
      <td>henry_bibby</td>
    </tr>
  </tbody>
</table>
</div>




```python
teamcoach_df2 = teamcoach_df
teamcoach_df2.columns = [['season','wteamid','firstdaynum','lastdaynum','wcoachname']]
```


```python
regularmod_df = pd.merge(regular_df,teamcoach_df2,how='left',on=['season','wteamid'])
regularmod_df.head()
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
      <th>season</th>
      <th>daynum</th>
      <th>wteamid</th>
      <th>wscore</th>
      <th>lteamid</th>
      <th>lscore</th>
      <th>wloc</th>
      <th>numot</th>
      <th>wfgm</th>
      <th>wfga</th>
      <th>wfgm3</th>
      <th>wfga3</th>
      <th>wftm</th>
      <th>wfta</th>
      <th>wor</th>
      <th>wdr</th>
      <th>wast</th>
      <th>wto</th>
      <th>wstl</th>
      <th>wblk</th>
      <th>wpf</th>
      <th>lfgm</th>
      <th>lfga</th>
      <th>lfgm3</th>
      <th>lfga3</th>
      <th>lftm</th>
      <th>lfta</th>
      <th>lor</th>
      <th>ldr</th>
      <th>last</th>
      <th>lto</th>
      <th>lstl</th>
      <th>lblk</th>
      <th>lpf</th>
      <th>index</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>wcoachname</th>
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
      <td>3</td>
      <td>14</td>
      <td>11</td>
      <td>18</td>
      <td>14</td>
      <td>24</td>
      <td>13</td>
      <td>23</td>
      <td>7</td>
      <td>1</td>
      <td>22</td>
      <td>22</td>
      <td>53</td>
      <td>2</td>
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
      <td>0</td>
      <td>0</td>
      <td>154</td>
      <td>mark_gottfried</td>
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
      <td>8</td>
      <td>20</td>
      <td>10</td>
      <td>19</td>
      <td>15</td>
      <td>28</td>
      <td>16</td>
      <td>13</td>
      <td>4</td>
      <td>4</td>
      <td>18</td>
      <td>24</td>
      <td>67</td>
      <td>6</td>
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
      <td>1</td>
      <td>0</td>
      <td>154</td>
      <td>john_calipari</td>
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
      <td>8</td>
      <td>18</td>
      <td>17</td>
      <td>29</td>
      <td>17</td>
      <td>26</td>
      <td>15</td>
      <td>10</td>
      <td>5</td>
      <td>2</td>
      <td>25</td>
      <td>22</td>
      <td>73</td>
      <td>3</td>
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
      <td>2</td>
      <td>0</td>
      <td>154</td>
      <td>tom_crean</td>
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
      <td>3</td>
      <td>9</td>
      <td>17</td>
      <td>31</td>
      <td>6</td>
      <td>19</td>
      <td>11</td>
      <td>12</td>
      <td>14</td>
      <td>2</td>
      <td>18</td>
      <td>18</td>
      <td>49</td>
      <td>6</td>
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
      <td>3</td>
      <td>0</td>
      <td>154</td>
      <td>rob_judson</td>
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
      <td>6</td>
      <td>14</td>
      <td>11</td>
      <td>13</td>
      <td>17</td>
      <td>22</td>
      <td>12</td>
      <td>14</td>
      <td>4</td>
      <td>4</td>
      <td>20</td>
      <td>24</td>
      <td>62</td>
      <td>6</td>
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
      <td>4</td>
      <td>0</td>
      <td>154</td>
      <td>rick_barnes</td>
    </tr>
  </tbody>
</table>
</div>




```python
regular_df.season.unique().shape
```




    (15L,)




```python
reg_int = regular_df.groupby(['season','wteamid'])['index'].count().reset_index()
reg_int.columns = [['season','teamid','wcount']]
reg_int.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>wcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>17</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
reg_int2 = regular_df.groupby(['season','lteamid'])['daynum'].count().reset_index()
reg_int2.columns = [['season','teamid','lcount']]
reg_int2.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>lcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
reg_winrate =(
        pd.merge(reg_int2,reg_int,how='outer',on=['season','teamid']).fillna(0)
        .pipe(lambda x: x.assign(winrate=x.wcount/(x.wcount+x.lcount)))
)
reg_winrate.columns = [['season','teamid','lcount','wcount','reg_winrate']]
```


```python
reg_winrate.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>lcount</th>
      <th>wcount</th>
      <th>reg_winrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2003</td>
      <td>1102</td>
      <td>16.0</td>
      <td>12.0</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1103</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>0.481481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1104</td>
      <td>11.0</td>
      <td>17.0</td>
      <td>0.607143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2003</td>
      <td>1105</td>
      <td>19.0</td>
      <td>7.0</td>
      <td>0.269231</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003</td>
      <td>1106</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>0.464286</td>
    </tr>
  </tbody>
</table>
</div>




```python
post_int = (
            post_df
            .groupby(['season','wteamid'])['daynum']
            .count()
            .reset_index() 
)
post_int.columns = [['season','teamid','wcount']]
```


```python
post_int.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>wcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1104</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1116</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1120</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1130</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1181</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
post_int2 = (
            post_df
            .groupby(['season','lteamid'])['daynum']
            .count()
            .reset_index() 
)
post_int2.columns = [['season','teamid','lcount']]
```


```python
post_winrate= (
                post_int
                .merge(post_int2,how='outer',on=['season','teamid'])
                .fillna(0)
                .pipe(lambda x: x.assign(winrate=x.wcount/(x.wcount+x.lcount)))
)
post_winrate.columns = [['season','teamid','wcount','lcount','post_winrate']]
```


```python
post_winrate.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>wcount</th>
      <th>lcount</th>
      <th>post_winrate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985</td>
      <td>1104</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1116</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1120</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1130</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1181</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine3.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>coachname</th>
      <th>daysexp</th>
      <th>champions</th>
      <th>playoff</th>
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
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>0</td>
      <td>154</td>
      <td>bob_huggins</td>
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>0</td>
      <td>154</td>
      <td>wimp_sanderson</td>
      <td>154</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>0</td>
      <td>154</td>
      <td>james_oliver</td>
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>0</td>
      <td>154</td>
      <td>davey_whitney</td>
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine4 = (
            combine3
            .merge(reg_winrate,how='left',on=['season','teamid'])
            .drop(['lcount','wcount'],axis = 1)
            .merge(post_winrate,how='left',on=['season','teamid'])
            .drop(['lcount','wcount'],axis=1)
            .fillna(0)
)
```


```python
combine4.head()
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
      <th>season</th>
      <th>teamid</th>
      <th>firstdaynum</th>
      <th>lastdaynum</th>
      <th>coachname</th>
      <th>daysexp</th>
      <th>champions</th>
      <th>playoff</th>
      <th>reg_winrate</th>
      <th>post_winrate</th>
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
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1103</td>
      <td>0</td>
      <td>154</td>
      <td>bob_huggins</td>
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1104</td>
      <td>0</td>
      <td>154</td>
      <td>wimp_sanderson</td>
      <td>154</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1106</td>
      <td>0</td>
      <td>154</td>
      <td>james_oliver</td>
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1108</td>
      <td>0</td>
      <td>154</td>
      <td>davey_whitney</td>
      <td>154</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


