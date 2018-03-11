
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

# In[64]:


import numpy as np
import pandas as pd


# In[65]:


get_ipython().magic(u'qtconsole')


# In[2]:


# 1. Recreate the same winning % table as my reference

# 1a. Get tourney results and add which round these wins occured


# In[3]:


raw = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')


# In[4]:


# What the Brackets should look like
# Round 1: 32 matches
# Round 2: 16 matches
# Round 3: 8 matches (Sweet Sixteen)
# Round 4: 4 matches (Quarters)
# Round 5: 2 matches (Semis)
# Round 5: 1 match (Finals)

# Total = 63 matches


# In[5]:


# If we were to select 1 year, the df is already sorted by DayNum
# That means the last entry is the eventual winner and the match in round 6
# Working backwards


# In[6]:


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


# In[7]:


# Applying add_round_to_df to raw DF

df=pd.DataFrame()

for i in range(1985,2018):
    small_df=add_round_to_df(raw.query('Season=={}'.format(i)))
    df=df.append(small_df)
df.head()


# In[8]:


seedings = pd.read_csv('data/DataFiles/NCAATourneySeeds.csv')


# In[9]:


seedings.head()


# In[10]:


def parse_region(string1):
    return string1[0]


# In[11]:


def parse_seeding(string1):
    return int(string1[1:3])


# In[12]:


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


# In[13]:


df.head()


# In[14]:


"""
Recreate the data we saw in the ref paper
1985 - 2013
For example, in round 2, seed
number 2 teams have played 116 times against seed number 15
teams and won 109 of these games. An exponential random
variable with mean 29/109=0.266 is associated with seed 2 teams
to estimate their success rates in round 2.
"""


# In[15]:



len(df
 .query('Season>=1985 & Season<=2013')
 .query('Round==2')
 .query('W_seed==2')
)


# In[17]:


len(df
 .query('Season>=1985 & Season<=2013')
 .query('Round==3')
 .query('W_seed==7')
)/(2013-1985+1.0)


# In[18]:


# It Works!


# In[19]:


# Function to build Matrix 1 - Winning Rate Table based on seeds and rounds


# In[20]:


seeds = range(1,17) # 1 to 16
rounds = range(2,8) # 2 to 7
wr = np.zeros((len(seeds),len(rounds)))


# In[22]:


years = 2013-1985+1.0

for s in seeds:
    for r in rounds:
        wr[s-1,r-2]=len(df.query('Season>=1985 & Season<=2013')
         .query('Round=={}'.format(r))
         .query('W_seed=={}'.format(s))
        )/years
        


# In[23]:


wr


# In[25]:


def get_wr(seed,rd):
    return wr[seed-1,rd-2]


# In[28]:


# To get the winning rate of seed 7 in Round 3
get_wr(7,3)


# In[154]:


def prob(wseed,lseed,rd):
    num=get_wr(wseed,rd)
    den=(get_wr(wseed,rd)+get_wr(lseed,rd))
    if den==0 or num==0:
        return 0.5
    else:
        return num/den


# In[30]:


# What is the probability that 8th seed wins 9th seed in round 2
prob(8,9,2)


# In[31]:


# What is the probability that 1st seed wins 9th seed in round 3
prob(1,9,3)


# In[138]:


# Load default submissions


# In[155]:


sub=pd.read_csv('data/SampleSubmissionStage1.csv')


# In[140]:


sub.head()


# In[141]:


# Break up submission file into yr, id1, id2, pred


# In[142]:


def parse_yr(string1):
    a,b,c=string1.split('_')
    return int(a)


# In[143]:


def parse_id1(string1):
    a,b,c=string1.split('_')
    return int(b)


# In[144]:


def parse_id2(string1):
    a,b,c=string1.split('_')
    return int(c)


# In[156]:


sub=(sub
 .pipe(lambda x:x.assign(year=x.ID.apply(parse_yr)))
 .pipe(lambda x:x.assign(wid=x.ID.apply(parse_id1)))
 .pipe(lambda x:x.assign(lid=x.ID.apply(parse_id2)))
)


# In[157]:


# For id1, id2 - get seeding


# In[158]:


sub=(sub
 .merge(seedings,how='left',left_on=['year','wid'],right_on=['Season','TeamID'])
 .rename(columns={'Seed':'Wseed'})
 .pipe(lambda x:x.assign(Wseed=x.Wseed.apply(parse_seeding)))
 .merge(seedings,how='left',left_on=['year','lid'],right_on=['Season','TeamID'])
 .rename(columns={'Seed':'Lseed'})
 .pipe(lambda x:x.assign(Lseed=x.Lseed.apply(parse_seeding)))
 [['ID','Wseed','Lseed']]
)
sub.head() # seeds can have repetition if teams are from different regions


# In[45]:


# Calc new prob, replace Pred


# In[97]:


import itertools


# In[112]:


# What are all the bracket combinations?
# Key = round
# Values = tuples of id1,id2 matchups
brackets = {
    2 : [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)],
    3 : list(itertools.product((1,16),(8,9)))\
        +list(itertools.product((5,12),(4,13)))\
        +list(itertools.product((6,11),(3,14)))\
        +list(itertools.product((7,10),(2,15))),
    4 : list(itertools.product((1,8,9,16),(4,5,12,13)))\
        +list(itertools.product((6,11,3,14),(7,10,2,15)))
}


# In[113]:


def get_round(wseed,lseed):
    for k,v in brackets.items():
        if (wseed,lseed) in v:
            return k
        elif (lseed,wseed) in v:
            return k
    return 0


# In[115]:


get_round(1,9)


# In[ ]:


# Recompile a new submission file


# In[159]:


predictions=[]
for i,row in sub.iterrows():
    wseed=row[1]
    lseed=row[2]
    rd=get_round(wseed,lseed) #Wseed,Lseed
    if rd ==0:
        predictions.append(0.5)
    else:
        predictions.append(prob(wseed,lseed,rd))


# In[160]:


sub['Pred']=predictions


# In[161]:


final_submission = (
    sub.drop(columns=['Wseed','Lseed'])
)


# In[166]:


import datetime
timestamp=datetime.datetime.now().strftime('%Y-%m-%d')
final_submission.to_csv('output/baseline-{}.csv'.format(timestamp),index=False)

