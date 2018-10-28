
# coding: utf-8

# ___
# this notebook select features that are important before throwing into the model



get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from sklearn.feature_selection import *
from sklearn import *
from scipy import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel, SelectPercentile,RFECV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from aggregate_function import build_features_table, combine_features_table, coach_stats, win_rate_type_of_location




coach_file = 'data/DataFiles/TeamCoaches.csv'
regularseason_file = 'data/DataFiles/RegularSeasonDetailedResults.csv'
postseason_file = 'data/DataFiles/NCAATourneyCompactResults.csv'




initial_features = build_features_table.BuildFeaturesTable(regularseason_file)
win_rate_features = win_rate_type_of_location.WinRateTypeLocation(regularseason_file)
coach_features = coach_stats.CoachStats(coach_file,regularseason_file,postseason_file)

features = combine_features_table.CombineFeaturesTable(initial_features,win_rate_features,coach_features)


# ## Data Transformation for recency data
# - going to apply a flat weightage of 
#     - 85% to current year
#     - 15% to previous year



features_table = features.final_table

features_table.head()




features_table (
    features.final_table
    .pipe(lambda x:x.assign(shifted_team = x.TeamID.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate = x.win_rate.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_score = x.total_score.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_opponent_score = x.total_opponent_score.shift(+1)))
    .pipe(lambda x:x.assign(shifted_fgp = x.fgp.shift(+1)))
    .pipe(lambda x:x.assign(shifted_fg3p = x.fg3p.shift(+1)))
    .pipe(lambda x:x.assign(shifted_ftp = x.ftp.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_rebounds = x.total_rebounds.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_off_rebounds = x.total_off_rebounds.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_def_rebounds = x.total_def_rebounds.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_off_rebounds_percent = x.total_off_rebounds_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_def_rebounds_percent = x.total_def_rebounds_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_rebound_possession_percent = x.total_rebound_possession_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_rebound_possessiongain_percent = x.total_rebound_possessiongain_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_blocks = x.total_blocks.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assists = x.total_assists.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_steals = x.total_steals.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_turnover = x.total_turnover.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_personalfoul = x.total_personalfoul.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_block_opp_FGA_percent = x.total_block_opp_FGA_percent.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assist_per_fgm = x.total_assist_per_fgm.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assist_turnover_ratio = x.total_assist_turnover_ratio.shift(+1)))
    .pipe(lambda x:x.assign(shifted_expectation_per_game = x.expectation_per_game.shift(+1)))
    .pipe(lambda x:x.assign(shifted_total_assists = x.avg_lose_score_by.shift(+1)))
    .pipe(lambda x:x.assign(shifted_avg_lose_score_by = x.avg_win_score_by.shift(+1)))
    .pipe(lambda x:x.assign(shifted_avg_win_score_by = x.avg_win_score_by.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate_away = x.win_rate_away.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate_home = x.win_rate_home.shift(+1)))
    .pipe(lambda x:x.assign(shifted_win_rate_neutral = x.win_rate_neutral.shift(+1)))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_total_score = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_total_opponent_score = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_fgp = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_fg3p = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_ftp = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_total_rebounds = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
    .pipe(lambda x:x.assign(weighted_win_rate = np.where(
        (x.shifted_team == x.TeamID), 0.85*x.win_rate + 0.15 * x.shifted_win_rate
    )))
)

