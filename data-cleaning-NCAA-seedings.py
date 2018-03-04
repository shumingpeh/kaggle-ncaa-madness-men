
# coding: utf-8



import pandas as pd
import numpy as np




NCAA_results=pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')




NCAA_results.head()




seedings = pd.read_csv('data/DataFiles/NCAATourneySeeds.csv')




seedings.head()




def parse_seed(seed_string):
    
    return int(seed_string[1:3])




type(parse_seed('X09'))




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




df.head()




df.to_csv('input/tour-results-seed.csv',index=False)




get_ipython().magic('pinfo2 pd.pipe')




"""
WTeamID | Seed_a | LTeamID | Seed_b
1116 | 9 | 1234 | ? 


"""

