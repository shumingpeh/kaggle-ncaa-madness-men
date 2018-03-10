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
        self.processed_overall()

    def winning_games_stats(self):
        """Get winning games stats for each team"""
        self.winning_games_up_to_2013 = (
            self.df
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

    def losing_games_stats(self):
        """Get losing games stats for each team"""
        self.losing_games_up_to_2013 = (
            self.df
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

    def combine_both_winning_losing_games_stats(self):
        """Combine winning and losing games for each team"""
        self.combine_both_winning_losing_games_stats = (
            self.winning_games_up_to_2013
            .merge(self.losing_games_up_to_2013, how='left',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
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
            # win by how many points
            .pipe(lambda x:x.assign(lose_rate = 1-x.win_rate))
            .pipe(lambda x:x.assign(win_score_by = x.WScore - x.losing_opponent_score))
            .pipe(lambda x:x.assign(lose_score_by = x.LScore - x.winning_opponent_score))
            .pipe(lambda x:x.assign(expectation_per_game = x.win_rate * x.win_score_by/x.winning_num_counts + x.lose_rate * x.lose_score_by/x.losing_num_counts))
            .pipe(lambda x:x.assign(avg_win_score_by = x.win_score_by/x.winning_num_counts))
            .pipe(lambda x:x.assign(avg_lose_score_by = x.lose_score_by/x.losing_num_counts))
        )

    def cumulative_stats_for_teams_each_year(self):
        """Cumulative stats for each team every year"""
        self.cumulative_stats_for_team_each_year = (
            self.combine_both_winning_losing_games_stats
            .sort_values(['WTeamID','Season'])
            .groupby(['WTeamID'])
            .cumsum()
            .pipe(lambda x:x.assign(Season = self.combine_both_winning_losing_games_stats.Season.values))
            .pipe(lambda x:x.assign(TeamID = self.combine_both_winning_losing_games_stats.WTeamID.values))
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

    def processed_cum_overall(self):
        """Cumulative stats for overall fgp, fg3p, ftp"""
        self.processed_cum_overall = (
            self.cumulative_stats_for_team_each_year
            [['Season','TeamID','win_rate','total_score','total_opponent_score','fgp','fg3p','ftp', 'total_rebounds','total_off_rebounds','total_def_rebounds',
              'total_off_rebounds_percent','total_def_rebounds_percent','total_rebound_possession_percent','total_rebound_possessiongain_percent','total_blocks',
              'total_assists','total_steals','total_turnover','total_personalfoul','total_block_opp_FGA_percent','total_assist_per_fgm','total_assist_turnover_ratio',
              'expectation_per_game','avg_lose_score_by','avg_win_score_by']]
        )

    def processed_overall(self):
        """Stats for overall fgp, fg3p, ftp"""
        self.processed_overall = (
            self.combine_both_winning_losing_games_stats
            .rename(columns={"WTeamID":"TeamID"})
            .pipe(lambda x:x.assign(fgp = x.total_fgm/x.total_fga))
            .pipe(lambda x:x.assign(fg3p = x.total_fg3m/x.total_fg3a))
            .pipe(lambda x:x.assign(ftp = x.total_ftm/x.total_fta))
            [['Season','TeamID','win_rate','total_score','total_opponent_score','fgp','fg3p','ftp', 'total_rebounds','total_off_rebounds','total_def_rebounds',
              'total_off_rebounds_percent','total_def_rebounds_percent','total_rebound_possession_percent','total_rebound_possessiongain_percent','total_blocks',
              'total_assists','total_steals','total_turnover','total_personalfoul','total_block_opp_FGA_percent','total_assist_per_fgm','total_assist_turnover_ratio',
              'expectation_per_game','avg_lose_score_by','avg_win_score_by']]
    )