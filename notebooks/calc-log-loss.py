
# coding: utf-8

# # Calculate log loss offline
# 
# 

# ## Formula
# 
# LogLoss = -(1/n) * sum of 1..n [( y * log(yhat) + (1-y) * log(1-yhat)]

# 
# ## Definitions
# n is the number of games played
# 
# yhat is the predicted probabiltiy that team 1 beats team 2
# 
# y is 1 if team 1 wins, 0 if team 2 wins
# 
# log() is the natural (base e) logarithm

# In[1]:


import numpy as np

def logloss(y,yhat):
    """
    Calculate the logloss of 1 prediction
    """
    return y*np.log(yhat) + (1-y)*np.log(1-yhat)


# In[2]:


# Truer positives ( when we're more confident that we made the right prediction) should have lower log loss - Correct!
print logloss(1,0.5)
print logloss(1,0.983)


# In[3]:


## Stage 1

import pandas as pd
sub=pd.read_csv('/Users/dtan/Code/kaggle-ncaa-madness-men/output/baseline-2018-03-11.csv')


# In[4]:


sub.head()


# In[5]:


results=pd.read_csv('/Users/dtan/Code/kaggle-ncaa-madness-men/data/stage1/DataFiles/NCAATourneyCompactResults.csv')


# In[6]:


results.head()


# ## Test the accuracy of my logloss script by measuring logloss of Stage1 baseline prediction (1985-2013) on 2017's results

# In[7]:


r17=results.query("Season=='2017'")
r17.head(3)


# In[38]:


## I wish i can find a .query(..) example that does this

s17=sub[sub['ID'].str.contains('2017_')]
s17.head(3)


# In[9]:


def parse_id1(string1):
    a,b,c=string1.split('_')
    return int(b)

def parse_id2(string1):
    a,b,c=string1.split('_')
    return int(c)


# In[39]:


# Parse out wid and lid

s17=(s17
 .pipe(lambda x:x.assign(wid=x.ID.apply(parse_id1)))
 .pipe(lambda x:x.assign(lid=x.ID.apply(parse_id2)))
)
s17.head(3)


# In[12]:


def get_y(wid,lid):
    """
    Return 1 if wid won lid -> querying results df will return row of length 1
    Return 0 if wid did not win lid -> querying results df will return row of length 0
    Return -1 if the 2 teams never met
    """
    if len(r17.query("WTeamID=='{}' & LTeamID=='{}'".format(wid,lid)))==1:
        return 1
    elif len(r17.query("WTeamID=='{}' & LTeamID=='{}'".format(lid,wid)))==1:
        return 0
    else:
        return -1


# In[40]:


r17.head(3)


# In[41]:


y=[]
for index,row in s17.iterrows():
    y.append(get_y(row['wid'],row['lid']))


# In[42]:


s17['y']=y
s17=s17.drop(s17[s17.y<0].index) # Drop all -1

#Check that we have as many y as results
print s17.y.value_counts()
s17.head(3)


# ## Calculating Total Log Loss

# In[43]:


total_log_loss=0.0

for index,row in s17.iterrows():
    log_loss=logloss(row['y'],row['Pred'])
    total_log_loss+=log_loss

score=(-1.0/len(s17))*total_log_loss
print 'Final Log Loss Score={}'.format(score)


# In[ ]:


## Looks accurate


# ## Calculating Log Loss of Baseline-2 model on SBNation's predictions
