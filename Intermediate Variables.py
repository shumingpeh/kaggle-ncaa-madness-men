
# coding: utf-8

# ___
# This notebook decides on the intermediate variables being used



import pandas as pd
import numpy as np
import scipy
from sklearn import *


# ## Read data
# - regularseason detailed results
# - ~~cities~~
# - teams
# - coaches
#     - there is a problem if a coach is new, so to prevent this from happening
#     - coach will have a proxy variables of
#         1. number of years of experience up to that year
#         1. number of championship
#         1. number of playoffs made



raw_data_regularseason = pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv")




raw_data_teams = pd.read_csv("data/DataFiles/Teams.csv")




raw_data_coaches = pd.read_csv("data/DataFiles/TeamCoaches.csv")




raw_data_teams_coaches = (
    raw_data_teams
    .merge(raw_data_coaches, how='left', on=['TeamID'])
)




raw_data_regularseason.head()




raw_data_regularseason.dtypes


# ## Features to be included
# - Season year
# - winning/losing teamid
# - winning/losing score
# - winning/losing field goal percentage
# - winning/losing field goal 3 point percentage
# - winning/losing free throw percentage
# - win rate



winning_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WScore":"sum","WFGM":"sum","WFGA":"sum","WFGM3":"sum","WFGA3":"sum","WFTM":"sum","WFTA":"sum","LScore":"sum","winning_num_counts":"sum"})
    .reset_index()
    .rename(columns={"LScore":"losing_opponent_score"})
)




winning_teams_score_up_to_2013.head()




losing_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"WScore":"sum","LScore":"sum","LFGM":"sum","LFGA":"sum","LFGM3":"sum","LFGA3":"sum","LFTM":"sum","LFTA":"sum","losing_num_counts":"sum"})
    .reset_index()
    .rename(columns={"WScore":"winning_opponent_score"})
)




losing_teams_score_up_to_2013.head()




combine_winning_losing_stats_for_year = (
    winning_teams_score_up_to_2013
    .merge(losing_teams_score_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    .pipe(lambda x:x.assign(total_score = x.WScore + x.LScore))
    .pipe(lambda x:x.assign(total_opponent_score = x.winning_opponent_score + x.losing_opponent_score))
    .pipe(lambda x:x.assign(total_fgm = x.WFGM + x.LFGM))
    .pipe(lambda x:x.assign(total_fga = x.WFGA + x.LFGA))
    .pipe(lambda x:x.assign(total_ftm = x.WFTM + x.LFTM))
    .pipe(lambda x:x.assign(total_fta = x.WFTA + x.LFTA))
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .sort_values(['WTeamID','Season'])
)




combine_winning_losing_stats_for_year.head()




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
    .pipe(lambda x:x.assign())
)




cumulative_stats_for_team_each_year.head()




combine_winning_losing_stats_for_year.dtypes


# ## Some variations to try for features
# - separate winning and losing
# - 

# ## Intermediate Variables
# - Coach stats
# - win rate for home court
# - win rate for away court
# - win rate for neutral court
# - offensive stats
# - defensive stats
# - blocks, steals and personal fouls



train_df = prediction_df.query("Season <= 2013")
test_df = prediction_df.query("Season >= 2014")




train_data_x = train_df[['diff_seed']]
train_data_y = train_df['yhat']

test_data_x = test_df[['diff_seed']]
test_data_y = test_df['yhat']


# ## Initializing Logistics Regression



logreg = linear_model.LogisticRegression()




logreg.fit(train_data_x,train_data_y)




#logreg.predict(test_df[['diff_seed']])




test_results = pd.DataFrame(logreg.predict(test_df[['diff_seed']])).rename(columns={0:"prediction_result"})




logreg.score(test_data_x,test_data_y)




test_df['prediction_results'] = test_results.prediction_result.values




test_df.tail(20)




metrics.confusion_matrix(test_df.yhat,test_df.prediction_results)


# http://blog.yhat.com/posts/roc-curves.html



pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv").head()




pd.read_csv("data/DataFiles/RegularSeasonCompactResults.csv").head()


# ## Features selected
# - season
# - region --> perhaps encode to a number. example: west - east = 1001. west = victor, east = loser
# - wteamid
# - wscore
# - lteamid
# - lscore
# - wloc
# - winning field goal percentage
# - winning three point percentage
# - winning free throw percentage
# - transformed variable for rebounds (offensive and defensive)
# - transformed assist
# - transformed turnovers
# - transformed steals
# - transformed blocks
# - transformed personal fouls
# - repeat for losing team
# 
# *transformed variables exclude first



pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv").dtypes

