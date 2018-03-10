# this file builds the features required for the table that will be using for training data

import pandas as pd
import numpy as np

class BuildFeaturesTable():
	"""Getting all the initial features required for the training data"""
	def __init__(self, regularseason_data_file):
        super(BuildFeaturesTable, self).__init__()
        self.regularseason_data_file = regularseason_data_file
        self.df = pd.read_csv(self.regularseason_data_file)

        self.winning_games_stats()
        self.losing_games_stats()
        self.combine_both_winning_losing_games_stats()
        self.cumulative_stats_for_teams_each_year()
        self.processed_cum_overall()
        self.processed_cum_split_winning_losing()
        self.processed_overall()
        self.processed_split_winning_losing()

    def winning_games_stats(self):
        """Get winning games stats for each team"""
        self.winning_games_up_to_2013 = (
            self.df
            .pipe(lambda x:x.assign(winning_num_counts = 1))
            .query("Season <= 2013")
            .groupby(['Season','WTeamID'])
            .agg({"WScore":"sum","WFGM":"sum","WFGA":"sum","WFGM3":"sum","WFGA3":"sum","WFTM":"sum","WFTA":"sum","LScore":"sum","winning_num_counts":"sum"})
            .reset_index()
            .rename(columns={"LScore":"losing_opponent_score"})
        )

    def losing_games_stats(self):
        """Get losing games stats for each team"""
        self.losing_games_up_to_2013 = (
            self.df
            .pipe(lambda x:x.assign(losing_num_counts=1))
            .query("Season <= 2013")
            .groupby(['Season','LTeamID'])
            .agg({"WScore":"sum","LScore":"sum","LFGM":"sum","LFGA":"sum","LFGM3":"sum","LFGA3":"sum","LFTM":"sum","LFTA":"sum","losing_num_counts":"sum"})
            .reset_index()
            .rename(columns={"WScore":"winning_opponent_score"})
        )

    def combine_both_winning_losing_games_stats(self):
        """Combine winning and losing games for each team"""
        self.combine_both_winning_losing_games_stats = (
            self.winning_games_up_to_2013
            .merge(self.losing_games_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
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

    def cumulative_stats_for_teams_each_year(self):
        """Cumulative stats for each team every year"""
        self.cumulative_stats_for_team_each_year = (
            self.combine_both_winning_losing_games_stats
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

    def processed_cum_overall(self):
        """Cumulative stats for overall fgp, fg3p, ftp"""
        self.processed_cum_overall = (
            self.cumulative_stats_for_teams_each_year
            [['Season','TeamID','win_rate','total_score','total_opponent_score','fgp','fg3p','ftp']]
        )

    def processed_cum_split_winning_losing(self):
        """Cumulative stats for fgp, fg3p, ftp by splitting them into winning and losing games of the same team"""
        self.processed_cum_split_winning_losing = (
            self.cumulative_stats_for_teams_each_year
            [['Season','TeamID','win_rate','total_score','total_opponent_score','WFGP','WFG3P','WFTP','LFGP','LFG3P','LFTP']]
    )

    def processed_overall(self):
        """Stats for overall fgp, fg3p, ftp"""
        self.processed_overall = (
            self.combine_both_winning_losing_games_stats
            .rename(columns={"WTeamID":"TeamID"})
            .pipe(lambda x:x.assign(fgp = x.total_fgm/x.total_fga))
            .pipe(lambda x:x.assign(fg3p = x.total_fg3m/x.total_fg3a))
            .pipe(lambda x:x.assign(ftp = x.total_ftm/x.total_fta))
            [['Season','TeamID','win_rate','total_score','total_opponent_score','fgp','fg3p','ftp']]
    )

    def processed_split_winning_losing(self):
        """Stats for fgp, fg3p, ftp by splitting them into winning and losing games of the same team"""
        self.processed_split_winning_losing = (
            self.combine_both_winning_losing_games_stats
            .rename(columns={"WTeamID":"TeamID"})
            .pipe(lambda x:x.assign(WFGP = x.WFGM/x.WFGA))
            .pipe(lambda x:x.assign(WFG3P = x.WFGM3/x.WFGA3))
            .pipe(lambda x:x.assign(WFTP = x.WFTM/x.WFTA))
            .pipe(lambda x:x.assign(LFGP = x.LFGM/x.LFGA))
            .pipe(lambda x:x.assign(LFG3P = x.LFGM3/x.LFGA3))
            .pipe(lambda x:x.assign(LFTP = x.LFTM/x.LFTA))
            [['Season','TeamID','win_rate','total_score','total_opponent_score','WFGP','WFG3P','WFTP','LFGP','LFG3P','LFTP']]
    )