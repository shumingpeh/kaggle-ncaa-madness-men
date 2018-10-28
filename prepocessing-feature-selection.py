
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
from sklearn.feature_selection import RFE, SelectFromModel, SelectPercentile
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

from aggregate_function import build_features_table, combine_features_table, coach_stats, win_rate_type_of_location




coach_file = 'data/DataFiles/TeamCoaches.csv'
regularseason_file = 'data/DataFiles/RegularSeasonDetailedResults.csv'
postseason_file = 'data/DataFiles/NCAATourneyCompactResults.csv'




# initial_features = build_features_table.BuildFeaturesTable(regularseason_file)
# win_rate_features = win_rate_type_of_location.WinRateTypeLocation(regularseason_file)
# coach_features = coach_stats.CoachStats(coach_file,regularseason_file,postseason_file)

features = combine_features_table.CombineFeaturesTable(coach_file,regularseason_file,postseason_file)


# ## Feature Selection on Correlation Matrix
# - remove features that are highly correlated



features_table = (
    features.final_table_cum_processed
    .drop(['Season','TeamID'],1)
)

corr = features_table.corr()
features_table.corr(method='pearson').to_csv("output/cum_correlation_matrix.csv")


# columns to be excluded
# - total score
# - total rebounds
# - total blocks
# - total assist turnover ratio
# - E(X) per game
# - win rate
# - total rebound possession per game
# - win_rate_overall coach



mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})




features_table = (
    features.final_table_cum_processed
    .drop(['Season','TeamID','total_score','total_opponent_score','total_rebounds','total_blocks','total_assist_turnover_ratio','expectation_per_game',
           'win_rate','fg3p','win_rate_overall','total_off_rebounds_percent','total_def_rebounds_percent','total_rebound_possession_percent','total_rebound_possessiongain_percent'
          ],1)
    .fillna(0)
)


# ## Feature selection from collinearity
# - remove features that are collinearity
# - features are then passed through variance threshold before passing into models for feature selection



vif = pd.DataFrame()
vif['VIF_factor'] = [variance_inflation_factor(features_table.values, i) for i in range(features_table.shape[1])]
vif['features'] = features_table.columns


# features to further remove
# - total_opponent_score
# - total_rebound_possessiongain_percent 
# - fg3p



features_table = (
    features.final_table_cum_processed
    .drop(['Season','TeamID','total_score','total_opponent_score','total_rebounds','total_blocks',
           'total_assist_turnover_ratio','expectation_per_game', 'win_rate','fg3p','win_rate_overall',
           'total_off_rebounds_percent','total_def_rebounds_percent','total_rebound_possession_percent',
           'total_rebound_possessiongain_percent'
          ],1)
    .fillna(0)
)

vif = pd.DataFrame()
vif['VIF_factor'] = [variance_inflation_factor(features_table.values, i) for i in range(features_table.shape[1])]
vif['features'] = features_table.columns
vif


# ## Feature selection from PCA explained variance
# - use PCA to see how much variance does the feature account for, selecting up to 99% variance would be good enough
# - 15 features are enough for variance
#     - this doesnt tell us which features to discard but in the model selection of feature importance
#     - we can tell the model to choose up to 9 features max



covar_matrix = PCA(n_components = 19)

covar_matrix.fit(features_table)
variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
var #cumulative sum of variance explained with [n] features


# ## Final data transformation for feature table
# - post season is only what we care about
# - post season match ups will be what we are joining all the features table to
# - additional variable of seeding differential



features_table = (
    features.final_table_cum_processed
    .drop(['total_score','total_opponent_score','total_rebounds','total_blocks',
           'total_assist_turnover_ratio','expectation_per_game', 'win_rate','fg3p','win_rate_overall',
           'total_off_rebounds_percent','total_def_rebounds_percent','total_rebound_possession_percent',
           'total_rebound_possessiongain_percent'
          ],1)
    .fillna(0)
)




seeding_data = pd.read_csv("input/tour-results-seed.csv")




winning_team_perspective_df = (
    seeding_data
    .pipe(lambda x:x.assign(diff_seed = x.L_seed - x.W_seed))
    .pipe(lambda x:x.assign(outcome = 1))
    .merge(features_table,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
)




losing_team_perspective_df = (
    seeding_data
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    .pipe(lambda x:x.assign(outcome = 0))
    .merge(features_table,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
)




prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)




train_df = prediction_df.query("Season >= 2003 & Season <= 2016")
test_df = prediction_df.query("Season == 2017")




train_df.head()


# ## Feature selection of logistics regression
# - RFE
# - SelectFromModel



## RFE
train_data_x = train_df[['diff_seed','total_off_rebounds','total_def_rebounds','total_assists',
                         'total_steals','total_turnover','total_personalfoul','total_assist_per_fgm',
                         'avg_lose_score_by','avg_win_score_by','num_season','is_playoff','is_champion',
                         'fgp','total_block_opp_FGA_percent','win_rate_away','win_rate_home','win_rate_neutral',
                         'win_rate_post','win_rate_regular']]
train_data_y = train_df['outcome']
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 13)
fit = rfe.fit(train_data_x, train_data_y)
print("Num Features: "+ str(fit.n_features_))
print("Selected Features: " + str(fit.support_))
print("Feature Ranking: " + str(fit.ranking_))




test_data_x = test_df[['diff_seed','total_off_rebounds','total_def_rebounds','total_assists',
                         'total_steals','total_turnover','total_personalfoul','total_assist_per_fgm',
                         'avg_lose_score_by','avg_win_score_by','num_season','is_playoff','is_champion',
                         'fgp','total_block_opp_FGA_percent','win_rate_away','win_rate_home','win_rate_neutral',
                         'win_rate_post','win_rate_regular']]
test_data_y = test_df['outcome']




rfe.score(test_data_x,test_data_y)




new_train_data_x = train_df[['diff_seed','total_off_rebounds','total_assists',
                         'total_turnover','total_assist_per_fgm',
                         'avg_lose_score_by','avg_win_score_by','is_playoff','is_champion',
                         'win_rate_away','win_rate_home',
                         'win_rate_post','win_rate_regular']]

new_test_data_x = test_df[['diff_seed','total_off_rebounds','total_assists',
                         'total_turnover','total_assist_per_fgm',
                         'avg_lose_score_by','avg_win_score_by','is_playoff','is_champion',
                         'win_rate_away','win_rate_home',
                         'win_rate_post','win_rate_regular']]


## use features and run on RF
rf = RandomForestClassifier(random_state=0)
param_grid = {
         'n_estimators': [5,10,50,100,150,200,500,1000],
         'max_depth': [2,5,10]
     }

grid_rf = GridSearchCV(rf, param_grid, cv=5, verbose=2)
grid_rf.fit(new_train_data_x, train_data_y)

rf_model = grid_rf.best_estimator_
model = rf_model

model.score(new_test_data_x,test_data_y)




model.predict(new_test_data_x)[:67]




## SVM


# ## Feature selection of RF



def fx_test(input):
    print("test")
    if input == 1:
        return "true"
    else:
        return "false"




fx_test(1)




features_table.query("TeamID == 1455 & Season == 2014")

