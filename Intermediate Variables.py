
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
combine_winning_losing_stats_for_year.dtypes




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




cumulative_stats_for_team_each_year.dtypes


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
# - ~~win rate for home court~~
# - ~~win rate for away court~~
# - ~~win rate for neutral court~~
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
# - ~~expectation to win by how many points in a game~~
# - 
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



#win rate for home court
#need to ensure that the joining is from a bigger table
raw_data_regularseason.head()




win_test = (
    raw_data_regularseason
    .groupby(['Season','WTeamID','WLoc'])
    .count()
    .reset_index()
    [['Season','WTeamID','WLoc','DayNum']]
)




lose_test = (
    raw_data_regularseason
    .groupby(['Season','LTeamID','WLoc'])
    .count()
    .reset_index()
    [['Season','LTeamID','WLoc','DayNum']]
)




win_test.head()




lose_test.head()




test = (
    lose_test
    .drop(['DayNum'],1)
    .append(win_test.rename(columns={"WTeamID":"LTeamID"}).drop(['DayNum'],1))
    .groupby(['Season','LTeamID','WLoc'])
    .count()
    .reset_index()
)

win_rate_type_of_court = (
    test
    .merge(win_test,how='left',left_on=['Season','LTeamID','WLoc'], right_on=['Season','WTeamID','WLoc'])
    .merge(lose_test,how='left',left_on=['Season','LTeamID','WLoc'],right_on=['Season','LTeamID','WLoc'])
    .fillna(0)
    .rename(columns={"LTeamID":"TeamID","DayNum_x":"games_won","DayNum_y":"games_lost"})
    .drop(['WTeamID'],1)
    .pipe(lambda x:x.assign(win_rate = x.games_won/(x.games_won + x.games_lost)))
)


win_rate_type_of_court.head()




win_rate_away = (
    win_rate_type_of_court
    .query("WLoc == 'A'")
    .rename(columns={"win_rate":"win_rate_away"})
    [['Season','TeamID','win_rate_away']]
)

win_rate_neutral = (
    win_rate_type_of_court
    .query("WLoc == 'N'")
    .rename(columns={"win_rate":"win_rate_neutral"})
    [['Season','TeamID','win_rate_neutral']]
)

win_rate_home = (
    win_rate_type_of_court
    .query("WLoc == 'H'")
    .rename(columns={"win_rate":"win_rate_home"})
    [['Season','TeamID','win_rate_home']]
)

more_testing = win_rate_type_of_court.sort_values(['TeamID','Season']).query("WLoc=='A'").head().groupby(['TeamID']).cumsum()

whatever = win_rate_away.sort_values(['TeamID','Season']).head()

more_testing.pipe(lambda x:x.assign(TeamID = whatever.TeamID.values))




# combine back with cumulative table
cumulative_stats_for_team_each_year.head()




intermediate_combine_stats_for_team_each_year = (
    cumulative_stats_for_team_each_year
    .merge(win_rate_away,how='left',on=['Season','TeamID'])
    .merge(win_rate_home,how='left',on=['Season','TeamID'])
    .merge(win_rate_neutral,how='left',on=['Season','TeamID'])
)

intermediate_combine_stats_for_team_each_year.head()


# ## offensive stats



# scored 
# offensive rebounds
# percentage of offensive rebounds to total rebounds
# offensive rebounding percentage, field goal missed
# defensive rebounds




# block % from opponent field goal attempted
# assist / turnover ratio
# assist per fgm

# win by how many points
# lose by how many points




# normalization on variables




raw_data_regularseason.head()




#win and lose by how many points




combine_winning_losing_stats_for_year.dtypes




win_rate_df = (
    combine_winning_losing_stats_for_year
    [['Season','WTeamID','winning_num_counts','losing_num_counts','WScore','losing_opponent_score','LScore','winning_opponent_score']]
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
    .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
    .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
    .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
    .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
    .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
    .rename(columns={"WTeamID":"TeamID"})
)

win_rate_df.head()




win_rate_cum_df = (
    cumulative_stats_for_team_each_year
    [['Season','TeamID','winning_num_counts','losing_num_counts','WScore','losing_opponent_score','LScore','winning_opponent_score']]
    .pipe(lambda x:x.assign(win_rate = x.winning_num_counts/(x.winning_num_counts + x.losing_num_counts)))
    .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
    .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
    .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
    .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
    .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
    .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
)

win_rate_cum_df.head()




# rebounds
raw_data_regularseason.dtypes




rebounds_winning_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WOR":"sum","WDR":"sum","WFGA":"sum","WFGM":"sum","LFGM":"sum","LFGA":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(total_winning_rebounds = x.WOR + x.WDR))
    .pipe(lambda x:x.assign(winning_off_rebounds_percent = x.WOR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(winning_def_rebounds_percent = x.WDR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(opp_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_rebound_possession_percent = x.WOR/x.team_missed_attempts))
    .pipe(lambda x:x.assign(winning_rebound_possessiongain_percent = x.WDR/x.opp_team_missed_attempts))
)




rebounds_winning_teams_score_up_to_2013.head()




rebounds_losing_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"LOR":"sum","LDR":"sum","LFGM":"sum","LFGA":"sum","WFGA":"sum","WFGM":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(total_losing_rebounds = x.LOR + x.LDR))
    .pipe(lambda x:x.assign(losing_off_rebounds_percent = x.LOR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_def_rebounds_percent = x.LDR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_opp_team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(losing_rebound_possession_percent = x.LOR/x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(losing_rebound_possessiongain_percent = x.LDR/x.winning_opp_team_missed_attempts))
)

rebounds_losing_teams_score_up_to_2013.head()




combine_winning_losing_rebounds_stats_for_year = (
    rebounds_winning_teams_score_up_to_2013
    .merge(rebounds_losing_teams_score_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    .pipe(lambda x:x.assign(total_rebounds = x.total_winning_rebounds + x.total_losing_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds = x.WOR + x.LOR))
    .pipe(lambda x:x.assign(total_def_rebounds = x.WDR + x.LDR))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_team_missed_attempts = x.team_missed_attempts + x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(total_opp_team_missed_attempts = x.opp_team_missed_attempts + x.winning_opp_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    .rename(columns={"WTeamID":"TeamID"})
    [['Season','TeamID','total_rebounds','total_off_rebounds','total_def_rebounds','total_def_rebounds_percent',
      'total_off_rebounds_percent','total_rebound_possession_percent','total_rebound_possessiongain_percent',
      'total_team_missed_attempts','total_opp_team_missed_attempts']]
)




combine_winning_losing_rebounds_stats_for_year.head()




cumulative_winning_losing_rebounds_stats = (
    combine_winning_losing_rebounds_stats_for_year
    .sort_values(['TeamID','Season'])
    .groupby(['TeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    .pipe(lambda x:x.assign(Season = combine_winning_losing_stats_for_year.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_winning_losing_stats_for_year.WTeamID.values))
)




# blocks, steals, assists




raw_data_regularseason.dtypes




bl_sl_topf_winning_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WAst":"sum","WTO":"sum","WStl":"sum","WBlk":"sum","WPF":"sum","LFGA":"sum","WFGM":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(winning_block_opp_FGA_percent = x.WBlk/x.LFGA))
    .pipe(lambda x:x.assign(winning_assist_per_fgm = x.WAst/x.WFGM))
    .pipe(lambda x:x.assign(winning_assist_turnover_ratio = x.WAst/x.WTO))
)

bl_sl_topf_winning_teams_score_up_to_2013.head()




bl_sl_topf_losing_teams_score_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"LAst":"sum","LTO":"sum","LStl":"sum","LBlk":"sum","LPF":"sum","WFGA":"sum","LFGM":"sum"})
    .reset_index()
    .pipe(lambda x:x.assign(losing_block_opp_FGA_percent = x.LBlk/x.WFGA))
    .pipe(lambda x:x.assign(losing_assist_per_fgm = x.LAst/x.LFGM))
    .pipe(lambda x:x.assign(losing_assist_turnover_ratio = x.LAst/x.LTO))
)

bl_sl_topf_losing_teams_score_up_to_2013.head()




combine_winning_losing_other_stats_for_year = (
    bl_sl_topf_winning_teams_score_up_to_2013
    .merge(bl_sl_topf_losing_teams_score_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    .pipe(lambda x:x.assign(total_blocks = x.WBlk + x.LBlk))
    .pipe(lambda x:x.assign(total_assists = x.WAst + x.LAst))
    .pipe(lambda x:x.assign(total_steals = x.WStl + x.LStl))
    .pipe(lambda x:x.assign(total_turnover = x.WTO + x.LTO))
    .pipe(lambda x:x.assign(total_personalfoul = x.WPF + x.LPF))
    .pipe(lambda x:x.assign(total_opp_fga = x.LFGA + x.WFGA))
    .pipe(lambda x:x.assign(total_fgm = x.WFGM + x.LFGM))
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
    .rename(columns={"WTeamID":"TeamID"})
    [['Season','TeamID','total_blocks','total_assists','total_steals','total_turnover','total_personalfoul','total_block_opp_FGA_percent','total_assist_per_fgm','total_assist_turnover_ratio','total_opp_fga','total_fgm']]
)




combine_winning_losing_other_stats_for_year.head()




combine_winning_losing_other_stats_for_year.dtypes




cumulative_winning_losing_rebounds_stats = (
    combine_winning_losing_other_stats_for_year
    .sort_values(['TeamID','Season'])
    .groupby(['TeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
    .pipe(lambda x:x.assign(Season = combine_winning_losing_stats_for_year.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_winning_losing_stats_for_year.WTeamID.values))
)




cumulative_winning_losing_rebounds_stats.head()




#min max standardization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
minmax_scale = scaler.fit(combine_winning_losing_other_stats_for_year[['total_assists']])
df_minmax = minmax_scale.transform(combine_winning_losing_other_stats_for_year[['total_assists']])




winning_games_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(winning_num_counts = 1))
    .query("Season <= 2013")
    .groupby(['Season','WTeamID'])
    .agg({"WScore":"sum","WFGM":"sum","WFGA":"sum","WFGM3":"sum","WFGA3":"sum","WFTM":"sum","WFTA":"sum","LScore":"sum","winning_num_counts":"sum",
          "WOR":"sum","WDR":"sum","LFGM":"sum","LFGA":"sum",
          "WAst":"sum","WTO":"sum","WStl":"sum","WBlk":"sum","WPF":"sum"})
    .reset_index()
    .rename(columns={"LScore":"losing_opponent_score"})
    # rebounds
    .pipe(lambda x:x.assign(total_winning_rebounds = x.WOR + x.WDR))
    .pipe(lambda x:x.assign(winning_off_rebounds_percent = x.WOR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(winning_def_rebounds_percent = x.WDR/x.total_winning_rebounds))
    .pipe(lambda x:x.assign(team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(opp_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_rebound_possession_percent = x.WOR/x.team_missed_attempts))
    .pipe(lambda x:x.assign(winning_rebound_possessiongain_percent = x.WDR/x.opp_team_missed_attempts))
    # blocks, steals, assists and turnovers
    .pipe(lambda x:x.assign(winning_block_opp_FGA_percent = x.WBlk/x.LFGA))
    .pipe(lambda x:x.assign(winning_assist_per_fgm = x.WAst/x.WFGM))
    .pipe(lambda x:x.assign(winning_assist_turnover_ratio = x.WAst/x.WTO))
    # rename columns to prevent duplication when joining with losing stats. example: WFGM_x
    .rename(columns={"LFGA":"LFGA_opp","LFGM":"LFGM_opp"})
)




winning_games_up_to_2013.head()




losing_games_up_to_2013 = (
    raw_data_regularseason
    .pipe(lambda x:x.assign(losing_num_counts=1))
    .query("Season <= 2013")
    .groupby(['Season','LTeamID'])
    .agg({"WScore":"sum","LScore":"sum","LFGM":"sum","LFGA":"sum","LFGM3":"sum","LFGA3":"sum","LFTM":"sum","LFTA":"sum","losing_num_counts":"sum",
          "LOR":"sum","LDR":"sum","WFGA":"sum","WFGM":"sum",
          "LAst":"sum","LTO":"sum","LStl":"sum","LBlk":"sum","LPF":"sum"})
    .reset_index()
    .rename(columns={"WScore":"winning_opponent_score"})
    # rebounds
    .pipe(lambda x:x.assign(total_losing_rebounds = x.LOR + x.LDR))
    .pipe(lambda x:x.assign(losing_off_rebounds_percent = x.LOR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_def_rebounds_percent = x.LDR/x.total_losing_rebounds))
    .pipe(lambda x:x.assign(losing_team_missed_attempts = x.LFGA - x.LFGM))
    .pipe(lambda x:x.assign(winning_opp_team_missed_attempts = x.WFGA - x.WFGM))
    .pipe(lambda x:x.assign(losing_rebound_possession_percent = x.LOR/x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(losing_rebound_possessiongain_percent = x.LDR/x.winning_opp_team_missed_attempts))
    # blocks, steals, assists and turnovers
    .pipe(lambda x:x.assign(losing_block_opp_FGA_percent = x.LBlk/x.WFGA))
    .pipe(lambda x:x.assign(losing_assist_per_fgm = x.LAst/x.LFGM))
    .pipe(lambda x:x.assign(losing_assist_turnover_ratio = x.LAst/x.LTO))
    # rename columns to prevent duplication when joining with losing stats. example: WFGM_x
    .rename(columns={"WFGA":"WFGA_opp","WFGM":"WFGM_opp"})
)

losing_games_up_to_2013.head()




combine_both_winning_losing_games_stats = (
    winning_games_up_to_2013
    .merge(losing_games_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
    # on field goal percentage and winning counts
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
    # on offensive and defensive rebounds
    .pipe(lambda x:x.assign(total_rebounds = x.total_winning_rebounds + x.total_losing_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds = x.WOR + x.LOR))
    .pipe(lambda x:x.assign(total_def_rebounds = x.WDR + x.LDR))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_team_missed_attempts = x.team_missed_attempts + x.losing_team_missed_attempts))
    .pipe(lambda x:x.assign(total_opp_team_missed_attempts = x.opp_team_missed_attempts + x.winning_opp_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    # on steals, turnovers, assists, blocks and personal fouls
    .pipe(lambda x:x.assign(total_blocks = x.WBlk + x.LBlk))
    .pipe(lambda x:x.assign(total_assists = x.WAst + x.LAst))
    .pipe(lambda x:x.assign(total_steals = x.WStl + x.LStl))
    .pipe(lambda x:x.assign(total_turnover = x.WTO + x.LTO))
    .pipe(lambda x:x.assign(total_personalfoul = x.WPF + x.LPF))
    .pipe(lambda x:x.assign(total_opp_fga = x.LFGA_opp + x.WFGA_opp))
    .pipe(lambda x:x.assign(total_fgm = x.WFGM + x.LFGM))
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
)

combine_both_winning_losing_games_stats.head()




cumulative_stats_for_team_each_year.dtypes[0:33]




cumulative_stats_for_team_each_year.dtypes[34:67]




cumulative_stats_for_team_each_year.dtypes[68:100]




cumulative_stats_for_team_each_year = (
    combine_both_winning_losing_games_stats
    .sort_values(['WTeamID','Season'])
    .groupby(['WTeamID'])
    .cumsum()
    .pipe(lambda x:x.assign(Season = combine_both_winning_losing_games_stats.Season.values))
    .pipe(lambda x:x.assign(TeamID = combine_both_winning_losing_games_stats.WTeamID.values))
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
    # rebounds cumsum stats
    .pipe(lambda x:x.assign(total_def_rebounds_percent = x.total_def_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_off_rebounds_percent = x.total_off_rebounds/x.total_rebounds))
    .pipe(lambda x:x.assign(total_rebound_possession_percent = x.total_off_rebounds/x.total_team_missed_attempts))
    .pipe(lambda x:x.assign(total_rebound_possessiongain_percent = x.total_def_rebounds/x.total_opp_team_missed_attempts))
    # assists, turnovers, steals, blocks and personal fouls
    .pipe(lambda x:x.assign(total_block_opp_FGA_percent = x.total_blocks/x.total_opp_fga))
    .pipe(lambda x:x.assign(total_assist_per_fgm = x.total_assists/x.total_fgm))
    .pipe(lambda x:x.assign(total_assist_turnover_ratio = x.total_assists/x.total_turnover))
    # win or lose by how many points
    .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
    .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
    .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
    .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
    .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
    .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
)




cumulative_stats_for_team_each_year.head()




from aggregate_function import build_features_table, win_rate_type_of_location




test_features = build_features_table.BuildFeaturesTable("data/DataFiles/RegularSeasonDetailedResults.csv")




win_rate_location_test = win_rate_type_of_location.WinRateTypeLocation("data/DataFiles/RegularSeasonDetailedResults.csv")




win_rate_location_test.processed_cumulative_win_rate_df.head()




test = test_features.processed_overall




test.head()




# this combines type of win rate to build features table
win_rate_features_combine = (
    test
    .merge(win_rate_location_test.processed_win_rate_df, how='left',on=['Season','TeamID'])
    .fillna(0)
)




win_rate_features_combine.head()




coach_file = 'data/DataFiles/TeamCoaches.csv'
regularseason_file = 'data/DataFiles/RegularSeasonDetailedResults.csv'
postseason_file = 'data/DataFiles/NCAATourneyCompactResults.csv'




from aggregate_function import coach_stats
testing_df = coach_stats.CoachStats(coach_file,regularseason_file,postseason_file)




testing_df.cumulative_final_coach_stats_table.head()




final_table = (
    win_rate_features_combine
    .merge(testing_df.cumulative_final_coach_stats_table[['Season','TeamID','num_season',
                                               'is_playoff','is_champion','win_rate_post',
                                               'win_rate_regular','win_rate_overall','CoachName']],
          how='left',on=['Season','TeamID'])
)
final_table.head()




final_table_copy = final_table.drop(['Season','TeamID','CoachName','win_rate','fgp','fg3p','ftp',
                                     'total_off_rebounds_percent','total_def_rebounds_percent',
                                     'total_rebound_possession_percent','total_rebound_possessiongain_percent',
                                     'total_block_opp_FGA_percent','win_rate_away','win_rate_home','win_rate_neutral',
                                     'win_rate_post','win_rate_regular','win_rate_overall'],1)
final_table_copy.dtypes




final_table_copy




scaler = MinMaxScaler()
minmax_scale = scaler.fit(final_table_copy)
df_minmax = minmax_scale.transform(final_table_copy)




test_out = pd.DataFrame(df_minmax)




test_out.columns = ['total_score', 'total_opponent_score', 'total_rebounds',
       'total_off_rebounds', 'total_def_rebounds', 'total_blocks',
       'total_assists', 'total_steals', 'total_turnover',
       'total_personalfoul', 'total_assist_per_fgm',
       'total_assist_turnover_ratio', 'expectation_per_game',
       'avg_lose_score_by', 'avg_win_score_by', 'num_season', 'is_playoff',
       'is_champion']




pd.DataFrame(final_table_copy.dtypes).index.values




test_out.columns = pd.DataFrame(final_table_copy.dtypes).index.values




from aggregate_function import combine_features_table




combine_features_table.CombineFeaturesTable(test_features,win_rate_location_test,testing_df)

