

```python
import pandas as pd
import numpy as np
```


```python
NCAA_results=pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')
```


```python
NCAA_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
seedings = pd.read_csv('data/DataFiles/NCAATourneySeeds.csv')
```


```python
seedings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
def parse_seed(seed_string):
    
    return int(seed_string[1:3])
```


```python
type(parse_seed('X09'))
```




    int




```python
df = (
    NCAA_results
    .merge(seedings,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"Seed":"W_seed"})
    .merge(seedings,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"Seed":"L_seed"})
    [['Season','WTeamID','W_seed','LTeamID','L_seed']]
    .assign()
    .pipe(lambda x:x.assign(W_seed = x.W_seed.apply(parse_seed)))
    .pipe(lambda x:x.assign(L_seed = x.L_seed.apply(parse_seed)))
)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
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
      <td>1985</td>
      <td>1116</td>
      <td>9</td>
      <td>1234</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1985</td>
      <td>1120</td>
      <td>11</td>
      <td>1345</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
      <td>1207</td>
      <td>1</td>
      <td>1250</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1985</td>
      <td>1229</td>
      <td>9</td>
      <td>1425</td>
      <td>8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1985</td>
      <td>1242</td>
      <td>3</td>
      <td>1325</td>
      <td>14</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.to_csv('input/tour-results-seed.csv',index=False)
```


```python
pd.pipe??
```

    Object `pd.pipe` not found.



```python
"""
WTeamID | Seed_a | LTeamID | Seed_b
1116 | 9 | 1234 | ? 


"""
```
