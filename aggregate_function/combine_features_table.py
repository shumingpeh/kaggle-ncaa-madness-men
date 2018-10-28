# this file combines all the features into a finalized table for training data

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from aggregate_function import (build_features_table, 
                               coach_stats, 
                               win_rate_type_of_location)

class CombineFeaturesTable():
    """Combine all required features for training data"""
    # def __init__(self, initial_features_table, win_rate_type_location_table, coach_stats_table):
    def __init__(self, coach_data_file, regularseason_data_file, postseason_data_file):
        super(CombineFeaturesTable, self).__init__()
        self.coach_data_file = coach_data_file
        self.regularseason_data_file = regularseason_data_file
        self.postseason_data_file = postseason_data_file
        self.initial_features_table = build_features_table.BuildFeaturesTable(self.regularseason_data_file)
        self.win_rate_type_location_table = win_rate_type_of_location.WinRateTypeLocation(self.regularseason_data_file)
        self.coach_stats_table = coach_stats.CoachStats(self.coach_data_file,self.regularseason_data_file,self.postseason_data_file)


        self.combine_feature_table()
        self.combine_feature_cumulative_table()
        self.min_max_standardization()
        self.final_input_training_data()

    def combine_feature_table(self):
        """ Combine all tables into one finalized table as training data
        """
        self.final_table = (
            self.initial_features_table.processed_overall
            .merge(self.win_rate_type_location_table.processed_win_rate_df, how='left',on=['Season','TeamID'])
            .merge(self.coach_stats_table.cumulative_final_coach_stats_table[['Season','TeamID','num_season',
                                               'is_playoff','is_champion','win_rate_post',
                                               'win_rate_regular','win_rate_overall','CoachName']],
                    how='left',on=['Season','TeamID'])
            .fillna(0)
            .drop(['CoachName'],1)
        )

    def combine_feature_cumulative_table(self):
        """ Combine all tables into one finalize cumulative table as training data
        """
        self.final_cumulative_table = (
            self.initial_features_table.processed_cum_overall
            .merge(self.win_rate_type_location_table.processed_cumulative_win_rate_df, how='left',on=['Season','TeamID'])
            .merge(self.coach_stats_table.cumulative_final_coach_stats_table[['Season','TeamID','num_season',
                                               'is_playoff','is_champion','win_rate_post',
                                               'win_rate_regular','win_rate_overall','CoachName']],
                    how='left',on=['Season','TeamID'])
            .fillna(0)
            .drop(['CoachName'],1)
        )

    def min_max_standardization(self):
        """ Convert non percentage values to min max standardization values
        """
        self.final_table_min_max = (
            self.final_table
            .drop(['Season','TeamID','win_rate','fgp','fg3p','ftp', #'CoachName'
                   'total_off_rebounds_percent','total_def_rebounds_percent',
                   'total_rebound_possession_percent','total_rebound_possessiongain_percent',
                   'total_block_opp_FGA_percent','win_rate_away','win_rate_home','win_rate_neutral',
                   'win_rate_post','win_rate_regular','win_rate_overall'],1)
        )
        
        self.final_table_cumulative_min_max = (
            self.final_cumulative_table
            .drop(['Season','TeamID','win_rate','fgp','fg3p','ftp', #'CoachName'
                   'total_off_rebounds_percent','total_def_rebounds_percent',
                   'total_rebound_possession_percent','total_rebound_possessiongain_percent',
                   'total_block_opp_FGA_percent','win_rate_away','win_rate_home','win_rate_neutral',
                   'win_rate_post','win_rate_regular','win_rate_overall'],1)
        )

        scaler = MinMaxScaler()
        minmax_scale = scaler.fit(self.final_table_min_max)
        self.df_minmax = pd.DataFrame(minmax_scale.transform(self.final_table_min_max))
        self.df_minmax.columns = pd.DataFrame(self.final_table_min_max.dtypes).index.values

        minmax_cum_scale = scaler.fit(self.final_table_cumulative_min_max)
        self.df_minmax_cum = pd.DataFrame(minmax_cum_scale.transform(self.final_table_cumulative_min_max))
        self.df_minmax_cum.columns = pd.DataFrame(self.final_table_cumulative_min_max.dtypes).index.values

    def final_input_training_data(self):
        """ final input training data of values, all features are bounded between 0 and 1
        """
        self.final_table_processed = (
            self.df_minmax
            .assign(Season = self.final_table.Season.values)
            .assign(TeamID = self.final_table.TeamID.values)
            # .assign(CoachName = self.final_table.CoachName.values)
            .assign(win_rate = self.final_table.win_rate.values)
            .assign(fgp = self.final_table.fgp.values)
            .assign(fg3p = self.final_table.fg3p.values)
            .assign(total_off_rebounds_percent = self.final_table.total_off_rebounds_percent.values)
            .assign(total_def_rebounds_percent = self.final_table.total_def_rebounds_percent.values)
            .assign(total_rebound_possession_percent = self.final_table.total_rebound_possession_percent.values)
            .assign(total_rebound_possessiongain_percent = self.final_table.total_rebound_possessiongain_percent.values)
            .assign(total_block_opp_FGA_percent = self.final_table.total_block_opp_FGA_percent.values)
            .assign(win_rate_away = self.final_table.win_rate_away.values)
            .assign(win_rate_home = self.final_table.win_rate_home.values)
            .assign(win_rate_neutral = self.final_table.win_rate_neutral.values)
            .assign(win_rate_post = self.final_table.win_rate_post.values)
            .assign(win_rate_regular = self.final_table.win_rate_regular.values)
            .assign(win_rate_overall = self.final_table.win_rate_overall.values)
        )

        self.final_table_cum_processed = (
            self.df_minmax_cum
            .assign(Season = self.final_cumulative_table.Season.values)
            .assign(TeamID = self.final_cumulative_table.TeamID.values)
            # .assign(CoachName = final_cumulative_table.final_table_cumulative_min_max.CoachName.values)
            .assign(win_rate = self.final_cumulative_table.win_rate.values)
            .assign(fgp = self.final_cumulative_table.fgp.values)
            .assign(fg3p = self.final_cumulative_table.fg3p.values)
            .assign(total_off_rebounds_percent = self.final_cumulative_table.total_off_rebounds_percent.values)
            .assign(total_def_rebounds_percent = self.final_cumulative_table.total_def_rebounds_percent.values)
            .assign(total_rebound_possession_percent = self.final_cumulative_table.total_rebound_possession_percent.values)
            .assign(total_rebound_possessiongain_percent = self.final_cumulative_table.total_rebound_possessiongain_percent.values)
            .assign(total_block_opp_FGA_percent = self.final_cumulative_table.total_block_opp_FGA_percent.values)
            .assign(win_rate_away = self.final_cumulative_table.win_rate_away.values)
            .assign(win_rate_home = self.final_cumulative_table.win_rate_home.values)
            .assign(win_rate_neutral = self.final_cumulative_table.win_rate_neutral.values)
            .assign(win_rate_post = self.final_cumulative_table.win_rate_post.values)
            .assign(win_rate_regular = self.final_cumulative_table.win_rate_regular.values)
            .assign(win_rate_overall = self.final_cumulative_table.win_rate_overall.values)
        )

