
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


# ## Final data transformation for feature table
# - post season is only what we care about
# - post season match ups will be what we are joining all the features table to
# - additional variable of seeding differential



features_table = (
    features.final_table_processed
    .drop(['total_score','total_rebounds','total_blocks','total_assist_turnover_ratio','expectation_per_game',
           'win_rate','total_rebound_possession_percent','win_rate_overall','total_off_rebounds_percent','total_def_rebounds_percent',
           'total_opponent_score','total_rebound_possessiongain_percent','fg3p'
          ],1)
    .fillna(0)
)




seeding_data = pd.read_csv("input/tour-results-seed.csv")




winning_team_perspective_df = (
    seeding_data
    .pipe(lambda x:x.assign(diff_seed = x.L_seed - x.W_seed))
    .pipe(lambda x:x.assign(outcome = 1))
    .merge(features_table,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .merge(features_table,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .pipe(lambda x:x.assign(diff_total_off_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_off_rebounds_x - x.total_off_rebounds_y,
                                                                x.total_off_rebounds_y - x.total_off_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_def_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_def_rebounds_x - x.total_def_rebounds_y,
                                                                x.total_def_rebounds_y - x.total_def_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_assists = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assists_x - x.total_assists_y,
                                                                x.total_assists_y - x.total_assists_x)))
    .pipe(lambda x:x.assign(diff_total_steals = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_steals_x - x.total_steals_y,
                                                                x.total_steals_y - x.total_steals_x)))
    .pipe(lambda x:x.assign(diff_total_turnover = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_turnover_x - x.total_turnover_y,
                                                                x.total_turnover_y - x.total_turnover_x)))
    .pipe(lambda x:x.assign(diff_total_personalfoul = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_personalfoul_x - x.total_personalfoul_y,
                                                                x.total_personalfoul_y - x.total_personalfoul_x)))
    .pipe(lambda x:x.assign(diff_total_assist_per_fgm = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assist_per_fgm_x - x.total_assist_per_fgm_y,
                                                                x.total_assist_per_fgm_y - x.total_assist_per_fgm_x)))
    .pipe(lambda x:x.assign(diff_avg_lose_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_lose_score_by_x - x.avg_lose_score_by_y,
                                                                x.avg_lose_score_by_y - x.avg_lose_score_by_x)))
    .pipe(lambda x:x.assign(diff_avg_win_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_win_score_by_x - x.avg_win_score_by_y,
                                                                x.avg_win_score_by_y - x.avg_win_score_by_x)))
    .pipe(lambda x:x.assign(diff_num_season = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.num_season_x - x.num_season_y,
                                                                x.num_season_y - x.num_season_x)))
    .pipe(lambda x:x.assign(diff_is_playoff = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_playoff_x - x.is_playoff_y,
                                                                x.is_playoff_y - x.is_playoff_x)))
    .pipe(lambda x:x.assign(diff_is_champion = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_champion_x - x.is_champion_y,
                                                                x.is_champion_y - x.is_champion_x)))
    .pipe(lambda x:x.assign(diff_fgp = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.fgp_x - x.fgp_y,
                                                                x.fgp_y - x.fgp_x)))
    .pipe(lambda x:x.assign(diff_total_block_opp_FGA_percent = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_block_opp_FGA_percent_x - x.total_block_opp_FGA_percent_y,
                                                                x.total_block_opp_FGA_percent_y - x.total_block_opp_FGA_percent_x)))
    .pipe(lambda x:x.assign(diff_win_rate_away = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_away_x - x.win_rate_away_y,
                                                                x.win_rate_away_y - x.win_rate_away_x)))
    .pipe(lambda x:x.assign(diff_win_rate_home = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_home_x - x.win_rate_home_y,
                                                                x.win_rate_home_y - x.win_rate_home_x)))
    .pipe(lambda x:x.assign(diff_win_rate_neutral = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_neutral_x - x.win_rate_neutral_y,
                                                                x.win_rate_neutral_y - x.win_rate_neutral_x)))
    .pipe(lambda x:x.assign(diff_win_rate_post = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_post_x - x.win_rate_post_y,
                                                                x.win_rate_post_y - x.win_rate_post_x)))
    .pipe(lambda x:x.assign(diff_win_rate_regular = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_regular_x - x.win_rate_regular_y,
                                                                x.win_rate_regular_y - x.win_rate_regular_x)))
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed
    ))
)

winning_team_perspective_df.head()




losing_team_perspective_df = (
    seeding_data
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed))
    .pipe(lambda x:x.assign(outcome = 0))
    .merge(features_table,how='left',left_on=['Season','LTeamID'],right_on=['Season','TeamID'])
    .merge(features_table,how='left',left_on=['Season','WTeamID'],right_on=['Season','TeamID'])
    .pipe(lambda x:x.assign(diff_total_off_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_off_rebounds_x - x.total_off_rebounds_y,
                                                                x.total_off_rebounds_y - x.total_off_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_def_rebounds = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_def_rebounds_x - x.total_def_rebounds_y,
                                                                x.total_def_rebounds_y - x.total_def_rebounds_x)))
    .pipe(lambda x:x.assign(diff_total_assists = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assists_x - x.total_assists_y,
                                                                x.total_assists_y - x.total_assists_x)))
    .pipe(lambda x:x.assign(diff_total_steals = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_steals_x - x.total_steals_y,
                                                                x.total_steals_y - x.total_steals_x)))
    .pipe(lambda x:x.assign(diff_total_turnover = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_turnover_x - x.total_turnover_y,
                                                                x.total_turnover_y - x.total_turnover_x)))
    .pipe(lambda x:x.assign(diff_total_personalfoul = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_personalfoul_x - x.total_personalfoul_y,
                                                                x.total_personalfoul_y - x.total_personalfoul_x)))
    .pipe(lambda x:x.assign(diff_total_assist_per_fgm = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_assist_per_fgm_x - x.total_assist_per_fgm_y,
                                                                x.total_assist_per_fgm_y - x.total_assist_per_fgm_x)))
    .pipe(lambda x:x.assign(diff_avg_lose_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_lose_score_by_x - x.avg_lose_score_by_y,
                                                                x.avg_lose_score_by_y - x.avg_lose_score_by_x)))
    .pipe(lambda x:x.assign(diff_avg_win_score_by = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.avg_win_score_by_x - x.avg_win_score_by_y,
                                                                x.avg_win_score_by_y - x.avg_win_score_by_x)))
    .pipe(lambda x:x.assign(diff_num_season = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.num_season_x - x.num_season_y,
                                                                x.num_season_y - x.num_season_x)))
    .pipe(lambda x:x.assign(diff_is_playoff = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_playoff_x - x.is_playoff_y,
                                                                x.is_playoff_y - x.is_playoff_x)))
    .pipe(lambda x:x.assign(diff_is_champion = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.is_champion_x - x.is_champion_y,
                                                                x.is_champion_y - x.is_champion_x)))
    .pipe(lambda x:x.assign(diff_fgp = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.fgp_x - x.fgp_y,
                                                                x.fgp_y - x.fgp_x)))
    .pipe(lambda x:x.assign(diff_total_block_opp_FGA_percent = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.total_block_opp_FGA_percent_x - x.total_block_opp_FGA_percent_y,
                                                                x.total_block_opp_FGA_percent_y - x.total_block_opp_FGA_percent_x)))
    .pipe(lambda x:x.assign(diff_win_rate_away = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_away_x - x.win_rate_away_y,
                                                                x.win_rate_away_y - x.win_rate_away_x)))
    .pipe(lambda x:x.assign(diff_win_rate_home = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_home_x - x.win_rate_home_y,
                                                                x.win_rate_home_y - x.win_rate_home_x)))
    .pipe(lambda x:x.assign(diff_win_rate_neutral = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_neutral_x - x.win_rate_neutral_y,
                                                                x.win_rate_neutral_y - x.win_rate_neutral_x)))
    .pipe(lambda x:x.assign(diff_win_rate_post = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_post_x - x.win_rate_post_y,
                                                                x.win_rate_post_y - x.win_rate_post_x)))
    .pipe(lambda x:x.assign(diff_win_rate_regular = np.where(
                                                                x.W_seed >= x.L_seed,
                                                                x.win_rate_regular_x - x.win_rate_regular_y,
                                                                x.win_rate_regular_y - x.win_rate_regular_x)))
    .pipe(lambda x:x.assign(diff_seed = x.W_seed - x.L_seed
    ))
)




prediction_df = (
    winning_team_perspective_df.append(losing_team_perspective_df)
)

train_df = prediction_df.query("Season >= 2003 & Season <= 2016")
test_df = prediction_df.query("Season == 2017")

train_df.head()

test_df.head()




train_data_x = train_df[['diff_seed','diff_total_off_rebounds','diff_total_def_rebounds','diff_total_assists',
                         'diff_total_steals','diff_total_turnover','diff_total_personalfoul',
                         'diff_total_assist_per_fgm','diff_avg_lose_score_by',
                         'diff_avg_win_score_by','diff_num_season','diff_is_playoff','diff_is_champion',
                         'diff_fgp','diff_total_block_opp_FGA_percent','diff_win_rate_away','diff_win_rate_home',
                         'diff_win_rate_neutral','diff_win_rate_post','diff_win_rate_regular']]
train_data_y = train_df['outcome']

test_data_x = test_df[['diff_seed','diff_total_off_rebounds','diff_total_def_rebounds','diff_total_assists',
                       'diff_total_steals','diff_total_turnover','diff_total_personalfoul',
                       'diff_total_assist_per_fgm','diff_avg_lose_score_by',
                       'diff_avg_win_score_by','diff_num_season','diff_is_playoff','diff_is_champion',
                       'diff_fgp','diff_total_block_opp_FGA_percent','diff_win_rate_away','diff_win_rate_home',
                       'diff_win_rate_neutral','diff_win_rate_post','diff_win_rate_regular']]
test_data_y = test_df['outcome']


# ## Feature selection for logistics regression
# - Univariate selection
# - SelectFromModel from RF, lassoCV



# univariate selection
percentile_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for i in percentile_list:
    select = SelectPercentile(percentile=i)

    select.fit(train_data_x, train_data_y)

    train_data_x_selected = select.transform(train_data_x)
    test_data_x_selected = select.transform(test_data_x)

    mask = select.get_support()    
#     print(mask)
    logreg = LogisticRegression()
    logreg.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal logreg: {}".format(logreg.score(test_data_x,test_data_y)))

    logreg.fit(train_data_x_selected,train_data_y)
    print("feature selection logreg: {}".format(logreg.score(test_data_x_selected,test_data_y)))
    print(Counter(logreg.predict(test_data_x_selected)[:67]))

    
# 2o percentile is the best FE for logistics regression




# based on the output of the univariate, we can narrow to 10, 25, 80
select_90 = SelectPercentile(percentile=90)
select_85 = SelectPercentile(percentile=85)




select_90.fit(train_data_x, train_data_y)

train_data_x_selected_90 = select_90.transform(train_data_x)
test_data_x_selected_90 = select_90.transform(test_data_x)

mask = select_90.get_support()    
#     print(mask)
logreg_90 = LogisticRegression()
logreg_90.fit(train_data_x_selected_90,train_data_y)

logreg_90.score(test_data_x_selected_90,test_data_y)

logreg_90.predict(test_data_x_selected_90)[:67]




select_85.fit(train_data_x, train_data_y)

train_data_x_selected_85 = select_85.transform(train_data_x)
test_data_x_selected_85 = select_85.transform(test_data_x)

mask = select_85.get_support()    
#     print(mask)
logreg_85 = LogisticRegression()
logreg_85.fit(train_data_x_selected_85,train_data_y)

logreg_85.score(test_data_x_selected_85,test_data_y)

logreg_85.predict(test_data_x_selected_85)[:67]




## selectfrommodel RF
select_rf = SelectFromModel(RandomForestClassifier(n_estimators =100, max_depth = 10, random_state=0),threshold=0.04)




select_rf.fit(train_data_x,train_data_y)




train_data_x_selected = select_rf.transform(train_data_x)
test_data_x_selected = select_rf.transform(test_data_x)




LogisticRegression().fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)




## selectfrommodel lassoCV --> same as univariate




select_lcv = SelectFromModel(LassoCV(max_iter=100,n_alphas=10,eps=1e-05),threshold=0.01)

select_lcv.fit(train_data_x,train_data_y)




train_data_x_selected_lcv = select_lcv.transform(train_data_x)
test_data_x_selected_lcv = select_lcv.transform(test_data_x)




LogisticRegression().fit(train_data_x_selected_lcv,train_data_y).score(test_data_x_selected_lcv,test_data_y)




# select from model lassocv
lassocv = LassoCV(random_state=0)
param_grid_lasso = {
    'n_alphas': [1,5,10,25,50,100,150,200,500,1000],
    'max_iter': [100,500,1000,1500,2000,3000],
    'eps': [0.00001,0.0001,0.001,0.01]
}
grid_lcv = GridSearchCV(lassocv, param_grid_lasso, cv=5, verbose=2)
grid_lcv.fit(train_data_x_selected, train_data_y)

lcv_model = grid_lcv.best_estimator_




lcv_model




select = SelectFromModel(LassoCV(max_iter=100,n_alphas=1,eps=0.001),threshold=0.04)

select.fit(train_data_x,train_data_y)




train_data_x_selected = select.transform(train_data_x)
train_data_x_selected.shape




test_data_x_selected = select.transform(test_data_x)




LogisticRegression().fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)




rf = RandomForestClassifier(random_state=0)
param_grid = {
    'n_estimators': [5,10,50,100,150,500,1000],
    'max_depth': [1,2,5,10,15,50,100]
}
grid_rf = GridSearchCV(rf, param_grid, scoring='accuracy', cv=5, verbose=0)
grid_rf.fit(train_data_x_selected, train_data_y)

rf_model = grid_rf.best_estimator_
rf_model


# ## Feature selection for RF
# - univariate
# - SelectFromModel SVM
# - RFE(CV)



model = LogisticRegression()
rfe = RFE(model, 9)
fit = rfe.fit(train_data_x, train_data_y)
print("Num Features: "+ str(fit.n_features_))
print("Selected Features: " + str(fit.support_))
print("Feature Ranking: " + str(fit.ranking_))




train_data_x_selected = fit.transform(train_data_x)
test_data_x_selected = fit.transform(test_data_x)




model.fit(train_data_x_selected,train_data_y).score(test_data_x_selected,test_data_y)




# univariate selection
percentile_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for i in percentile_list:
    select_rf = SelectPercentile(percentile=i)
    select_rf.fit(train_data_x, train_data_y)
    
    train_data_x_selected = select_rf.transform(train_data_x)
    test_data_x_selected = select_rf.transform(test_data_x)
    
    mask = select_rf.get_support()
#     print(mask)
    rf = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0)
    rf.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal rf: {}".format(rf.score(test_data_x,test_data_y)))
    
    rf.fit(train_data_x_selected,train_data_y)
    print("feature selection rf: {}".format(rf.score(test_data_x_selected,test_data_y)))
    print(Counter(rf.predict(test_data_x_selected)[:67]))
    




# based on the output of the univariate, we can narrow to 60, 80
select_70_rf = SelectPercentile(percentile=70)
select_80_rf = SelectPercentile(percentile=80)




select_70_rf.fit(train_data_x, train_data_y)

train_data_x_selected_70_rf = select_70_rf.transform(train_data_x)
test_data_x_selected_70_rf = select_70_rf.transform(test_data_x)

mask = select_70_rf.get_support()        
# print(mask)
rf_70 = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0,warm_start=True)
rf_70.fit(train_data_x_selected_70_rf,train_data_y)

rf_70.score(test_data_x_selected_70_rf,test_data_y)




rf_70.predict(test_data_x_selected_70_rf)[:67]




select_80_rf.fit(train_data_x, train_data_y)

train_data_x_selected_80_rf = select_80_rf.transform(train_data_x)
test_data_x_selected_80_rf = select_80_rf.transform(test_data_x)

mask = select_80_rf.get_support()        
# print(mask)
rf_80 = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0,warm_start=True)
rf_80.fit(train_data_x_selected_80_rf,train_data_y)

rf_80.score(test_data_x_selected_80_rf,test_data_y)




rf_80.predict(test_data_x_selected_80_rf)[:67]




Counter(rf_80.predict(test_data_x_selected_80_rf)[:67])




Counter(rf_70.predict(test_data_x_selected_70_rf)[:67])




## RFE CV from LR
# for i in [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20]:
model_rfe = LogisticRegression()
rfe = RFECV(model_rfe, step=1, cv=5)
fit = rfe.fit(train_data_x, train_data_y)
print("Num Features: "+ str(fit.n_features_))
print("Selected Features: " + str(fit.support_))
print("Feature Ranking: " + str(fit.ranking_))

train_data_x_selected_rfe = fit.transform(train_data_x)
test_data_x_selected_rfe = fit.transform(test_data_x)




rf_model = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0,warm_start=False)




rf_model.fit(train_data_x_selected_rfe,train_data_y).score(test_data_x_selected_rfe,test_data_y)


# ## SVM



## SVM
# univariate selection
percentile_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for i in percentile_list:
    select = SelectPercentile(percentile=i)

    select.fit(train_data_x, train_data_y)

    train_data_x_selected = select.transform(train_data_x)
    test_data_x_selected = select.transform(test_data_x)

    mask = select.get_support()    
#     print(mask)
    svm = SVC(probability=True)
    svm.fit(train_data_x,train_data_y)
    
    print("\nWhich percentile : " + str(i))
    print("normal svm: {}".format(svm.score(test_data_x,test_data_y)))

    svm.fit(train_data_x_selected,train_data_y)
    print("feature selection svm: {}".format(svm.score(test_data_x_selected,test_data_y)))
    print(Counter(svm.predict(test_data_x_selected)[:67]))
#     print(svm.predict(test_data_x_selected)[:67])
#     print(svm.predict_proba(test_data_x_selected)[0:10])

    
# 2o percentile is the best FE for logistics regression




# based on the output of the univariate, we can narrow to 100
select_100_svm = SelectPercentile(percentile=100)

select_100_svm.fit(train_data_x, train_data_y)

train_data_x_selected_100_svm = select_100_svm.transform(train_data_x)
test_data_x_selected_100_svm = select_100_svm.transform(test_data_x)

mask = select_100_svm.get_support()        
# print(mask)
svm_100 = SVC(probability=True)
svm_100.fit(train_data_x_selected_100_svm,train_data_y)

svm_100.score(test_data_x_selected_100_svm,test_data_y)

svm_100.predict(test_data_x_selected_100_svm)[:67]




rf_fi = RandomForestClassifier(n_estimators = 1000, max_depth=10,random_state=0)




rf_fi_values = (
    pd.DataFrame(rf_fi.fit(train_data_x,train_data_y).feature_importances_,index=train_data_x.columns)
    .rename(columns={0:"feature_importance_values"})
    .reset_index()
    .rename(columns={"index":"features"})
    .sort_values(['feature_importance_values'],ascending=False)
    .pipe(lambda x:x.assign(fi_cumsum = x.feature_importance_values.cumsum()))
    .query("fi_cumsum <= 0.95")
)




rf_fi_values.features.unique()




svm_train_data_x = train_df[['diff_win_rate_home', 'diff_is_playoff', 'diff_win_rate_post',
       'diff_avg_win_score_by', 'diff_total_def_rebounds', 'diff_seed',
       'diff_num_season', 'diff_win_rate_away', 'diff_win_rate_regular',
       'diff_total_assists', 'diff_avg_lose_score_by', 'diff_fgp',
       'diff_total_off_rebounds', 'diff_total_block_opp_FGA_percent',
       'diff_total_steals', 'diff_total_turnover',
       'diff_total_personalfoul']]
svm_train_data_y = train_df[['outcome']]

svm_test_data_x = test_df[['diff_win_rate_home', 'diff_is_playoff', 'diff_win_rate_post',
       'diff_avg_win_score_by', 'diff_total_def_rebounds', 'diff_seed',
       'diff_num_season', 'diff_win_rate_away', 'diff_win_rate_regular',
       'diff_total_assists', 'diff_avg_lose_score_by', 'diff_fgp',
       'diff_total_off_rebounds', 'diff_total_block_opp_FGA_percent',
       'diff_total_steals', 'diff_total_turnover',
       'diff_total_personalfoul']]
svm_test_data_y = test_df[['outcome']]





svm_fs = SVC()




svm_fs.fit(svm_train_data_x,svm_train_data_y)




svm_fs.score(svm_test_data_x,svm_test_data_y)




svm_fs_df = pd.DataFrame(svm_fs.predict(svm_test_data_x)[:67]).rename(columns={0:"svm_fs_df"})




log_rf_fs_df = pd.DataFrame(LogisticRegression().fit(train_data_x_selected,train_data_y).predict(test_data_x_selected)[:67]).rename(columns={0:"log_rf_fs_df"})




rf_70_df = pd.DataFrame(rf_70.predict(test_data_x_selected_70_rf)[:67]).rename(columns={0:"rf_70_fs"})




rf_80_df = pd.DataFrame(rf_80.predict(test_data_x_selected_80_rf)[:67]).rename(columns={0:"rf_80_fs"})




rf_rfe_df = pd.DataFrame(rf_model.fit(train_data_x_selected_rfe,train_data_y).predict(test_data_x_selected_rfe)[:67]).rename(columns={0:"rf_rfe"})




log_85_df = pd.DataFrame(logreg_85.predict(test_data_x_selected_85)[:67]).rename(columns={0:"log_85_fs"})




log_90_df = pd.DataFrame(logreg_90.predict(test_data_x_selected_90)[:67]).rename(columns={0:"log_90_fs"})




(
    svm_fs_df
    .merge(log_rf_fs_df,how='outer', left_index=True, right_index=True)
    .merge(rf_70_df,how='outer',left_index=True, right_index=True)
    .merge(rf_80_df,how='outer', left_index=True, right_index=True)
    .merge(rf_rfe_df,how='outer', left_index=True, right_index=True)
    .merge(log_85_df,how='outer', left_index=True, right_index=True)
    .merge(log_90_df,how='outer', left_index=True, right_index=True)
).to_csv("output/final_results_static_year_improved.csv",index=False)
