# this file builds the features required for the table that will be using for training data

import pandas as pd
import numpy as np

class WinRateTypeLocation():
    """Getting all the initial features required for the training data"""
    def __init__(self, regularseason_data_file):
        super(WinRateTypeLocation, self).__init__()
        self.regularseason_data_file = regularseason_data_file
        self.df = pd.read_csv(self.regularseason_data_file)

        self.games_won()
        self.games_lost()
        self.combine_both_lost_and_won()
        self.win_rate_away()
        self.win_rate_home()
        self.win_rate_neutral()
        self.win_rate_cum_away()
        self.win_rate_cum_home()
        self.win_rate_cum_neutral()

    def games_won(self):
        """Number of games won on each type of location"""
        self.games_won_df = (
            self.df
            .groupby(['Season','WTeamID','WLoc'])
            .count()
            .reset_index()
            [['Season','WTeamID','WLoc','DayNum']]
        )

    def games_lost(self):
        """Number of games lost on each type of location"""
        self.games_lost_df = (
            self.df
            .groupby(['Season','LTeamID','WLoc'])
            .count()
            .reset_index()
            [['Season','LTeamID','WLoc','DayNum']]
        )

    def combine_both_lost_and_won(self):
        """Combine both win and lost df to get stats"""
        self.intermediate_df = (
            self.games_lost_df
            .drop(['DayNum'],1)
            .append(self.games_won_df.rename(columns={"WTeamID":"LTeamID"}).drop(['DayNum'],1))
            .groupby(['Season','LTeamID','WLoc'])
            .count()
            .reset_index()
        )

        self.combine_both_lost_and_won_df = (
            self.intermediate_df
            .merge(self.games_won_df,how='left',left_on=['Season','LTeamID','WLoc'], right_on=['Season','WTeamID','WLoc'])
            .merge(self.games_lost_df,how='left',left_on=['Season','LTeamID','WLoc'],right_on=['Season','LTeamID','WLoc'])
            .fillna(0)
            .rename(columns={"LTeamID":"TeamID","DayNum_x":"games_won","DayNum_y":"games_lost"})
            .drop(['WTeamID'],1)
            .pipe(lambda x:x.assign(win_rate = x.games_won/(x.games_won + x.games_lost)))
        )

    def win_rate_away(self):
        """Win rate for away games, non cumulative"""
        self.win_rate_away_df = (
            self.combine_both_lost_and_won_df
            .query("WLoc == 'A'")
            .rename(columns={"win_rate":"win_rate_away"})
            [['Season','TeamID','win_rate_away']]
            .sort_values(['TeamID','Season'])
        )

    def win_rate_neutral(self):
        """Win rate for neutral games, non cumulative"""
        self.win_rate_neutral_df = (
            self.combine_both_lost_and_won_df
            .query("WLoc == 'N'")
            .rename(columns={"win_rate":"win_rate_neutral"})
            [['Season','TeamID','win_rate_neutral']]
            .sort_values(['TeamID','Season'])
        )

    def win_rate_home(self):
        """Win rate for home games, non cumulative"""
        self.win_rate_home_df = (
            self.combine_both_lost_and_won_df
            .query("WLoc == 'H'")
            .rename(columns={"win_rate":"win_rate_home"})
            [['Season','TeamID','win_rate_home']]
            .sort_values(['TeamID','Season'])
        )

    def win_rate_cum_away(self):
        """Win rate for away games, cumulative"""
        self.win_rate_cum_away_df = (
            self.combine_both_lost_and_won_df
            .sort_values(['TeamID','Season'])
            .query("WLoc == 'A'")
            .groupby(['TeamID'])
            .cumsum()
            .pipe(lambda x:x.assign(Season = self.win_rate_away_df.Season.values))
            .pipe(lambda x:x.assign(TeamID = self.win_rate_away_df.TeamID.values))
            .pipe(lambda x:x.assign(win_rate_away = x.games_won/(x.games_won+x.games_lost)))
        )

    def win_rate_cum_home(self):
        """Win rate for home games, cumulative"""
        self.win_rate_cum_home_df = (
            self.combine_both_lost_and_won_df
            .sort_values(['TeamID','Season'])
            .query("WLoc == 'H'")
            .groupby(['TeamID'])
            .cumsum()
            .pipe(lambda x:x.assign(Season = self.win_rate_home_df.Season.values))
            .pipe(lambda x:x.assign(TeamID = self.win_rate_home_df.TeamID.values))
            .pipe(lambda x:x.assign(win_rate_home = x.games_won/(x.games_won+x.games_lost)))
        )

    def win_rate_cum_neutral(self):
        """Win rate for neutral games, cumulative"""
        self.win_rate_cum_neutral_df = (
            self.combine_both_lost_and_won_df
            .sort_values(['TeamID','Season'])
            .query("WLoc == 'N'")
            .groupby(['TeamID'])
            .cumsum()
            .pipe(lambda x:x.assign(Season = self.win_rate_neutral_df.Season.values))
            .pipe(lambda x:x.assign(TeamID = self.win_rate_neutral_df.TeamID.values))
            .pipe(lambda x:x.assign(win_rate_neutral = x.games_won/(x.games_won+x.games_lost)))
        )