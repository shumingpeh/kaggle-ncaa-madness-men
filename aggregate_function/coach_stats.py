# this file builds the feature for coach stats that will be joined with the feature table

import pandas as pd
import numpy as np

class CoachStats():
    def __init__(self, coach_data_file, regularseason_data_file, postseason_data_file):
        super(CoachStats, self).__init__()
        self.coach_data_file = coach_data_file
        self.regularseason_data_file = regularseason_data_file
        self.postseason_data_file = postseason_data_file
        self.raw_data_coach = pd.read_csv(self.coach_data_file)
        self.raw_data_regularseason = pd.read_csv(self.regularseason_data_file)
        self.raw_data_postseason = pd.read_csv(self.postseason_data_file)

        self.get_season_experience_number()
        self.get_final_coach_for_each_season()
        self.get_playoff_made_for_each_coach()
        self.get_championship_won_for_each_coach()
        self.combine_playoff_championship_for_each_coach()
        self.get_win_rate_regular_season_for_each_coach()
        self.get_win_rate_post_season_for_each_coach()
        self.get_overall_win_rate_for_each_coach()
        self.processed_table()
        self.proessed_cumulative_table()

    def get_season_experience_number(self):
        # get season max days
        self.season_max_days = (
            self.raw_data_coach
            .groupby(['Season'])
            .agg({"LastDayNum":"max"})
            .reset_index()
            .rename(columns={"LastDayNum":"season_max_days"})
        )

        # get number of season for each coach for each year
        self.num_days_coach_for_season = (
            self.raw_data_coach
            .pipe(lambda x:x.assign(daysexp = x.LastDayNum-x.FirstDayNum))
            .merge(self.season_max_days, how='left',on=['Season'])
            .pipe(lambda x:x.assign(num_season = x.daysexp/x.season_max_days))
            .sort_values(['CoachName','Season'])
        )

    def get_final_coach_for_each_season(self):
        """ get final coach for each session
            coach with more than half the season will be the credited coach for eventual playoff and championship won
        """
        self.final_coach_for_season = (
            self.num_days_coach_for_season
            .groupby(['Season','TeamID'])
            .agg({"CoachName":"count"})
            .reset_index()
            .rename(columns={"CoachName":"coach_counts"})
            .merge(self.num_days_coach_for_season,how='left',on=['Season','TeamID'])
            .pipe(lambda x:x.assign(final_coach = np.where(x.num_season >= 0.5, x.CoachName, "ignore")))
            [['Season','TeamID','final_coach']]
        )

    def get_playoff_made_for_each_coach(self):
        # get teams for post season
        self.teams_for_postseason = (
            self.raw_data_postseason
            # get winning teams first
            .groupby(['Season','WTeamID'])
            .agg({"NumOT":"count"})
            .reset_index()
            # append losing teams to dataframe
            .append(self.raw_data_postseason[['Season','LTeamID','NumOT']].rename(columns={"LTeamID":"WTeamID"}))
            .groupby(['Season','WTeamID'])
            .agg({"NumOT":"count"})
            .reset_index()
            .rename(columns={"NumOT":"is_playoff"})
            .pipe(lambda x:x.assign(is_playoff = 1))
            .rename(columns={"WTeamID":"TeamID"})
        )

        self.final_coach_with_postseason_each_year = (
            self.final_coach_for_season
            .merge(self.teams_for_postseason,how='left',on=['Season','TeamID'])
            .fillna(0)
        )

    def get_championship_won_for_each_coach(self):
        """ Get teams who won the championship for each year
        """
        self.championship_team = (
            self.raw_data_postseason
            .merge(self.season_max_days,how='left',on=['Season'])
            .query("DayNum == season_max_days")
            .groupby(['Season','WTeamID'])
            .agg({"NumOT":"count"})
            .reset_index()
            .rename(columns={"NumOT":"is_champion","WTeamID":"TeamID"})
        )

    def combine_playoff_championship_for_each_coach(self):
        """ Combine teams who made to playoff and won championship to one dataframe
        """
        self.final_coach_with_postseason_champion_each_year = (
            self.final_coach_with_postseason_each_year
            .merge(self.championship_team,how='left',on=['Season','TeamID'])
            .fillna(0)
        )

    def get_win_rate_regular_season_for_each_coach(self):
        """ Get win rate for regular season for each coach
        """
        self.games_won_for_coaches = (
            self.raw_data_regularseason
            [['Season','DayNum','WTeamID']]
            # merge for winning team
            .merge(self.num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
                   how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
            .rename(columns={"FirstDayNum":"FirstDayNum_win","LastDayNum":"LastDayNum_win","CoachName":"CoachName_win","TeamID":"TeamID_win"})
            .pipe(lambda x:x.assign(which_coach_for_win = np.where((x.FirstDayNum_win <= x.DayNum) & (x.LastDayNum_win >= x.DayNum),1,0)))
            .query("which_coach_for_win != 0")
            .groupby(['Season','CoachName_win','WTeamID'])
            .agg({"which_coach_for_win":"sum"})
            .reset_index()
        )

        self.games_lose_for_coaches = (
            self.raw_data_regularseason
            [['Season','DayNum','LTeamID']]
            # merge for losing team
            .merge(self.num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
                   how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
            .rename(columns={"FirstDayNum":"FirstDayNum_lose","LastDayNum":"LastDayNum_lose","CoachName":"CoachName_lose","TeamID":"TeamID_lose"})
            .pipe(lambda x:x.assign(which_coach_for_lose = np.where((x.FirstDayNum_lose <= x.DayNum) & (x.LastDayNum_lose >= x.DayNum),1,0)))
            .query("which_coach_for_lose != 0")
            .groupby(['Season','CoachName_lose','LTeamID'])
            .agg({"which_coach_for_lose":"sum"})
            .reset_index()
        )

        # combine games won and lost df
        self.combine_regular_games_won_lose = (
            self.games_lose_for_coaches
            .merge(self.games_won_for_coaches,how='left',left_on=['Season','LTeamID','CoachName_lose'],right_on=['Season','WTeamID','CoachName_win'])
            .pipe(lambda x:x.assign(win_rate_regular = x.which_coach_for_win/(x.which_coach_for_win + x.which_coach_for_lose)))
            .drop(['CoachName_win','WTeamID'],1)
            .rename(columns={"CoachName_lose":"CoachName","LTeamID":"TeamID","which_coach_for_lose":"games_lost","which_coach_for_win":"games_won"})
        )

    def get_win_rate_post_season_for_each_coach(self):
        """ Get win rate for post season for each coach
        """
        # get winning games for coaches
        self.post_games_won_for_coaches = (
            self.raw_data_postseason
            [['Season','DayNum','WTeamID']]
            # merge for winning team
            .merge(self.num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
                   how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
            .rename(columns={"FirstDayNum":"FirstDayNum_win","LastDayNum":"LastDayNum_win","CoachName":"CoachName_win","TeamID":"TeamID_win"})
            .pipe(lambda x:x.assign(which_coach_for_win = np.where((x.FirstDayNum_win <= x.DayNum) & (x.LastDayNum_win >= x.DayNum),1,0)))
            .query("which_coach_for_win != 0")
            .groupby(['Season','CoachName_win','WTeamID'])
            .agg({"which_coach_for_win":"sum"})
            .reset_index()
        )

        # get losing games for coaches
        self.post_games_lose_for_coaches = (
            self.raw_data_postseason
            [['Season','DayNum','LTeamID']]
            # merge for losing team
            .merge(self.num_days_coach_for_season[['Season','TeamID','FirstDayNum','LastDayNum','CoachName']],
                   how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
            .rename(columns={"FirstDayNum":"FirstDayNum_lose","LastDayNum":"LastDayNum_lose","CoachName":"CoachName_lose","TeamID":"TeamID_lose"})
            .pipe(lambda x:x.assign(which_coach_for_lose = np.where((x.FirstDayNum_lose <= x.DayNum) & (x.LastDayNum_lose >= x.DayNum),1,0)))
            .query("which_coach_for_lose != 0")
            .groupby(['Season','CoachName_lose','LTeamID'])
            .agg({"which_coach_for_lose":"sum"})
            .reset_index()
        )

        # combine games won and lost df for post season
        self.combine_post_games_won_lose = (
            self.post_games_lose_for_coaches
            .merge(self.post_games_won_for_coaches,how='left',left_on=['Season','LTeamID','CoachName_lose'],right_on=['Season','WTeamID','CoachName_win'])
            .pipe(lambda x:x.assign(win_rate_post = x.which_coach_for_win/(x.which_coach_for_win + x.which_coach_for_lose)))
            .drop(['CoachName_win','WTeamID'],1)
            .rename(columns={"CoachName_lose":"CoachName","LTeamID":"TeamID","which_coach_for_lose":"post_games_lost","which_coach_for_win":"post_games_won"})
            .fillna(0)
        )

    def get_overall_win_rate_for_each_coach(self):
        self.overall_win_rate_for_coaches = (
            self.combine_post_games_won_lose
            .merge(self.combine_regular_games_won_lose,how='left',on=['Season','CoachName','TeamID'])
            .fillna(0)
            .pipe(lambda x:x.assign(overall_games_won = x.post_games_won + x.games_won))
            .pipe(lambda x:x.assign(overall_games_lost = x.post_games_lost + x.games_lost))
            .pipe(lambda x:x.assign(win_rate_overall = x.overall_games_won/(x.overall_games_won + x.overall_games_lost)))
        )

    def processed_table(self):
        self.final_coach_stats_table = (
            self.num_days_coach_for_season
            .merge(self.final_coach_with_postseason_champion_each_year,how='left',left_on=['Season','TeamID','CoachName'],right_on=['Season','TeamID','final_coach'])
            .fillna(0)
            .merge(self.overall_win_rate_for_coaches,how='left',on=['Season','TeamID','CoachName'])
            .fillna(0)
            .drop(['final_coach','FirstDayNum','LastDayNum'],1)
            .sort_values(['CoachName','Season'])
        )

    def proessed_cumulative_table(self):
        self.cumulative_final_coach_stats_table = (
            self.final_coach_stats_table
            .groupby(['CoachName'])
            .cumsum()
            .pipe(lambda x:x.assign(Season = self.final_coach_stats_table.Season.values))
            .pipe(lambda x:x.assign(TeamID = self.final_coach_stats_table.TeamID.values))
            .pipe(lambda x:x.assign(CoachName = self.final_coach_stats_table.CoachName.values))
            .pipe(lambda x:x.assign(win_rate_post = x.post_games_won/(x.post_games_won + x.post_games_lost)))
            .fillna(0)
            .pipe(lambda x:x.assign(win_rate_regular = x.games_won/(x.games_won + x.games_lost)))
            .fillna(0)
            .pipe(lambda x:x.assign(win_rate_overall = x.overall_games_won/(x.overall_games_won + x.overall_games_lost)))
            .fillna(0)
        )