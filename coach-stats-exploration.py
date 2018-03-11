
# coding: utf-8

# ___
# this notebook gets the coach stats



import pandas as pd
import numpy as np

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

pd.set_option("display.max_columns",50)


# ## Summary of notebook data manipulation
# 1. years/season of experience
# 1. number of playoff made
# 1. number of championship won
# 1. win rate
#     - regular
#     - post
#     - overall

# ## Read data



raw_data_regularseason = pd.read_csv("data/DataFiles/RegularSeasonDetailedResults.csv")
raw_data_coach = pd.read_csv('data/DataFiles/TeamCoaches.csv')
raw_data_postseason = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')


# ## Get number of season experience



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


# ## Get cumulative number of seasons experience



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


# ## Assign one coach to one season
# - check which teams have more than one coach in one season
#     - the coach with more days of coaching will be credited for the season



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


# ## Get number of playoffs made for coaches
# - check if team made to playoff to season
# - final coach gets the credit



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




# join postseason to final coach
final_coach_with_postseason_each_year = (
    final_coach_for_season
    .merge(teams_for_postseason,how='left',on=['Season','TeamID'])
    .fillna(0)
)


# ## Get number of championships won for coaches
# - check which team won championship
# - final coach gets the credit



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


# ## Get win rate for coach during regular season
# - get up till daynum of the coach in the team of the season
# - get number of games won and lost, so that reconciling with cumulative table will be okay



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




# combine both losing and winning games
combine_regular_games_won_lose = (
    games_lose_for_coaches
    .merge(games_won_for_coaches,how='left',left_on=['Season','LTeamID','CoachName_lose'],right_on=['Season','WTeamID','CoachName_win'])
    .pipe(lambda x:x.assign(win_rate_regular = x.which_coach_for_win/(x.which_coach_for_win + x.which_coach_for_lose)))
    .drop(['CoachName_win','WTeamID'],1)
    .rename(columns={"CoachName_lose":"CoachName","LTeamID":"TeamID","which_coach_for_lose":"games_lost","which_coach_for_win":"games_won"})
)

combine_regular_games_won_lose.head()


# ## Get win rate for coach during post season



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


# ## Get overall win rate for coaches



overall_win_rate_for_coaches = (
    combine_post_games_won_lose
    .merge(combine_regular_games_won_lose,how='left',on=['Season','CoachName','TeamID'])
    .fillna(0)
    .pipe(lambda x:x.assign(overall_games_won = x.post_games_won + x.games_won))
    .pipe(lambda x:x.assign(overall_games_lost = x.post_games_lost + x.games_lost))
    .pipe(lambda x:x.assign(win_rate_overall = x.overall_games_won/(x.overall_games_won + x.overall_games_lost)))
)

overall_win_rate_for_coaches.tail()


# ## Combine all coach stats into one master table



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


# ## Cumulative coach stats for master table



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




coach_file = 'data/DataFiles/TeamCoaches.csv'
regularseason_file = 'data/DataFiles/RegularSeasonDetailedResults.csv'
postseason_file = 'data/DataFiles/NCAATourneyCompactResults.csv'




from aggregate_function import coach_stats




testing_df = coach_stats.CoachStats(coach_file,regularseason_file,postseason_file)




testing_df.cumulative_final_coach_stats_table.head()


# ## Concluding remarks
# - need to clean up the way class file is written, at the moment, its a direct copy and paste into the class file to save time
