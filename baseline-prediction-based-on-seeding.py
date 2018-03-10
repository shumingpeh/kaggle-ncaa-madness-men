
# coding: utf-8

# # Calculating a baseline prediction solely looking at seeding
# 

# References: 
# 
# http://bracketodds.cs.illinois.edu/2015%20Omega.pdf
# 
# http://nessis.org/nessis11/jacobson.pdf

# We assume the rate at which 1 seed wins another as a Poisson distribution (part of the Exponential distribution family).
# 
# Poisson is used to estimate the waiting time between/how often 1 seed wins over another in a particular round.
# 
# This makes the assumption that a seed's winning in a round occurs continuously and independently at a constant rate.
# 
# (A rather naive assumption but suitable for making a baseline)
# 

# In[107]:


import numpy as np
import pandas as pd


# In[108]:


# 1. Recreate the same winning % table as my reference

# 1a. Get tourney results and add which round these wins occured


# In[109]:


raw = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')


# In[110]:


# It seems the no. of games is uneven in different years

for i in range(1985,2018):
    print len((raw.query("Season == {}".format(i))))


# In[111]:


# What the Brackets should look like
# Round 1: 32 matches
# Round 2: 16 matches
# Round 3: 8 matches (Sweet Sixteen)
# Round 4: 4 matches (Quarters)
# Round 5: 2 matches (Semis)
# Round 5: 1 match (Finals)

# Total = 63 matches


# In[112]:


# If we were to select 1 year, the df is already sorted by DayNum
# That means the last entry is the eventual winner and the match in round 6
# Working backwards


# In[113]:


def add_round_to_df(df):
    df1=df.copy()
    df1['Round']=2 # by default all rounds are 2nd
    
    # Replace Finals with Round 7 (Last row, 8th column)
    df1.iloc[-1,8]=7 
    
    # Replace Semis with Round 6
    df1.iloc[-2,8]=6
    df1.iloc[-3,8]=6
    
    # Replace Quarters with Round 5
    df1.iloc[-4,8]=5
    df1.iloc[-5,8]=5
    df1.iloc[-6,8]=5
    df1.iloc[-7,8]=5
    
    # So on... Round 4
    df1.iloc[-8,8]=4
    df1.iloc[-9,8]=4
    df1.iloc[-10,8]=4
    df1.iloc[-11,8]=4
    df1.iloc[-12,8]=4
    df1.iloc[-13,8]=4
    df1.iloc[-14,8]=4
    df1.iloc[-15,8]=4
    
    # Round 3
    df1.iloc[-16,8]=3
    df1.iloc[-17,8]=3
    df1.iloc[-18,8]=3
    df1.iloc[-19,8]=3
    df1.iloc[-20,8]=3
    df1.iloc[-21,8]=3
    df1.iloc[-22,8]=3
    df1.iloc[-23,8]=3
    df1.iloc[-24,8]=3
    df1.iloc[-25,8]=3
    df1.iloc[-26,8]=3
    df1.iloc[-27,8]=3
    df1.iloc[-28,8]=3
    df1.iloc[-29,8]=3
    df1.iloc[-30,8]=3
    df1.iloc[-31,8]=3

    return df1


# In[114]:


# Applying add_round_to_df to raw DF

df=pd.DataFrame()

for i in range(1985,2018):
    small_df=add_round_to_df(raw.query('Season=={}'.format(i)))
    df=df.append(small_df)
df.head()


# In[115]:


seedings = pd.read_csv('data/DataFiles/NCAATourneySeeds.csv')


# In[116]:


seedings.head()


# In[117]:


def parse_region(string1):
    return string1[0]


# In[118]:


def parse_seeding(string1):
    return int(string1[1:3])


# In[122]:


# Merge df and seedings
# Separate out the seedings (integer) and Region for both the winner and loser

df=(
    df
    .merge(seedings,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"Seed":"W_seed"})
    .merge(seedings,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .rename(columns={"Seed":"L_seed"})
    .pipe(lambda x:x.assign(W_region = x.W_seed.apply(parse_region)))
    .pipe(lambda x:x.assign(W_seed = x.W_seed.apply(parse_seeding)))
    .pipe(lambda x:x.assign(L_region = x.L_seed.apply(parse_region)))
    .pipe(lambda x:x.assign(L_seed = x.L_seed.apply(parse_seeding)))
)


# In[123]:


df.head()


# In[ ]:


"""
Recreate the data we saw in the ref paper
1985 - 2013
For example, in round 2, seed
number 2 teams have played 116 times against seed number 15
teams and won 109 of these games. An exponential random
variable with mean 29/109=0.266 is associated with seed 2 teams
to estimate their success rates in round 2.
"""


# In[133]:



len(df
 .query('Season>=1985 & Season<=2013')
 .query('Round==2')
 .query('W_seed==2')
)


# In[137]:


109/(2013-1985+1.0)


# In[142]:


len(df
 .query('Season>=1985 & Season<=2013')
 .query('Round==3')
 .query('W_seed==7')
)/(2013-1985+1.0)


# In[ ]:


# It Works!


# In[143]:


# Function to build Matrix 1 - Winning Rate Table based on seeds and rounds


# In[174]:


seeds = range(1,17) # 1 to 16
rounds = range(2,8) # 2 to 7
wr = np.zeros((len(seeds),len(rounds)))


# In[177]:


years = 2013-1985+1.0

for s in seeds:
    for r in rounds:
        wr[s-1,r-2]=len(df.query('Season>=1985 & Season<=2013')
         .query('Round=={}'.format(r))
         .query('W_seed=={}'.format(s))
        )/years
        


# In[178]:


wr


# In[180]:


# To get the winning rate of seed 7 in Round 3
wr[7-1,3-2]


# In[181]:


def get_wr(seed,rd):
    return wr[seed-1,rd-2]


# In[186]:


def prob(wseed,lseed,rd):
    return get_wr(wseed,rd)/(get_wr(wseed,rd)+get_wr(lseed,rd))


# In[188]:


# What is the probability that 8th seed wins 9th seed in round 2
prob(8,9,2)


# In[189]:


# What is the probability that 1st seed wins 9th seed in round 3
prob(1,9,3)

