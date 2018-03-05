
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
# - overall win rate



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
    .pipe(lambda x:x.assign(total_fg3m = x.WFGM3 + x.LFGM3))
    .pipe(lambda x:x.assign(total_fg3a = x.WFGA3 + x.LFGA3))
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
    .pipe(lambda x:x.assign(LFGP = x.LFGM/x.LFGA))
    .pipe(lambda x:x.assign(LFG3P = x.LFGM3/x.LFGA3))
    .pipe(lambda x:x.assign(LFTP = x.LFTM/x.LFTA))
    .pipe(lambda x:x.assign(fgp = x.total_fgm/x.total_fga))
    .pipe(lambda x:x.assign(fg3p = x.total_fg3m/x.total_fg3a))
    .pipe(lambda x:x.assign(ftp = x.total_ftm/x.total_fta))
)




cumulative_stats_for_team_each_year.head()


# ## Some variations to try for features
# - separate winning and losing
#     - reconcilation of winning and losing will have to be done later
#     - could be diff between percentage --> this might give an insight of when they are losing/winning?

# ## Intermediate Variables
# - Coach stats
#     - number of years till that season
#     - number of championship till that season
#     - number of playoffs made till that season
#     - win rate of total games till that season
#         - consider regular or playoff only?
# - win rate for home court
# - win rate for away court
# - win rate for neutral court
# - offensive stats
#     - offensive rebounds
#     - points scored
#     - might try play by play later?
# - defensive stats
#     - defensive rebounds
#     - points scored by opponents
#     - turn over from play by play???
#     - might try play by play later?
# - blocks, steals and personal fouls
# 
# 
# #### reconcilation of intermediate variables
# - relative scoring method
#      - will have a score of between 0 to 1
# 
# 
# #### features being throw into prediction model
# - test out raw intermediate variables
#     - then test out difference in values
#     - or something else



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

